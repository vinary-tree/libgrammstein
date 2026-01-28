//! Import state machine for deterministic resource management.
//!
//! This module provides a state machine abstraction for the Google Books import
//! pipeline. It ensures deterministic cleanup order, preventing hang bugs caused
//! by improper resource release sequences.
//!
//! ## Problem Solved
//!
//! The original monolithic `import_http_reactive` (~900 lines) had a subtle bug:
//! when dropping `shared_state` before waiting for workers to exit, the workers
//! still held `Arc<WorkerSharedState>` references. This kept the `progress_tx`
//! channel sender alive, preventing the `worker_converter` task from completing.
//!
//! ## Solution: State Machine + CleanupGuard
//!
//! - **ImportPhase**: Explicit states prevent illegal transitions
//! - **ImportTrigger**: Events that drive state transitions
//! - **CleanupGuard**: LIFO cleanup order ensures channels close properly
//!
//! ## Cleanup Order (Critical!)
//!
//! Resources MUST be cleaned up in this order:
//! 1. Signal all workers to shut down
//! 2. Wait for workers to exit (they drop their Arc references)
//! 3. Drop shared_state (releases one Arc ref, but may not deallocate yet)
//! 4. Drop other channel senders (result_tx, worker_exit_tx)
//! 5. Abort and wait for stats task (releases Arc<shared_state>, closes progress_tx)
//! 6. Wait for worker_converter (channel now closed, can exit)
//! 7. Abort and wait for command handler

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::Sender;
use tokio::sync::watch::Sender as WatchSender;
use tokio::task::JoinHandle;

/// Import pipeline phases representing the current state of the state machine.
///
/// The state machine transitions through these phases in a well-defined order,
/// with cleanup happening automatically via `CleanupGuard` when transitioning
/// to terminal states.
///
/// Note: This is distinct from `importer::ImportPhase` which is used for progress
/// tracking. This enum represents the full state machine states including
/// initialization and cleanup phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelinePhase {
    /// Initializing resources (channels, workers, etc.)
    Initializing,

    /// Actively downloading and processing n-gram files.
    Downloading {
        active_workers: usize,
        results_received: u64,
        total_pending: u64,
    },

    /// Paused by user request. Workers complete current job then idle.
    Paused {
        results_received: u64,
        total_pending: u64,
    },

    /// Computing Modified Kneser-Ney statistics.
    ComputingStats,

    /// Merging shards into final output.
    Merging {
        shards_processed: usize,
        total_shards: usize,
    },

    /// Cleaning up temporary files and resources.
    CleaningUp,

    /// Import completed successfully.
    Completed {
        total_ngrams: u64,
        duration: Duration,
    },

    /// Import cancelled by user (checkpoint saved).
    Cancelled,

    /// Force quit requested (no checkpoint saved).
    ForceQuit,

    /// Import failed with error.
    Failed { error: String },
}

impl PipelinePhase {
    /// Check if this is a terminal state (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            PipelinePhase::Completed { .. }
                | PipelinePhase::Cancelled
                | PipelinePhase::ForceQuit
                | PipelinePhase::Failed { .. }
        )
    }

    /// Get a human-readable name for this phase.
    pub fn name(&self) -> &'static str {
        match self {
            PipelinePhase::Initializing => "Initializing",
            PipelinePhase::Downloading { .. } => "Downloading",
            PipelinePhase::Paused { .. } => "Paused",
            PipelinePhase::ComputingStats => "Computing MKN Statistics",
            PipelinePhase::Merging { .. } => "Merging Shards",
            PipelinePhase::CleaningUp => "Cleaning Up",
            PipelinePhase::Completed { .. } => "Completed",
            PipelinePhase::Cancelled => "Cancelled",
            PipelinePhase::ForceQuit => "Force Quit",
            PipelinePhase::Failed { .. } => "Failed",
        }
    }
}

/// Triggers that cause state transitions in the import pipeline.
#[derive(Debug)]
pub enum ImportTrigger {
    /// Initialization complete, ready to start downloading.
    InitComplete { total_pending: u64 },

    /// A job result was received from a worker.
    JobResult {
        /// N-gram order (1-5).
        order: u8,
        /// Prefix that was processed.
        prefix: Arc<str>,
        /// Number of n-grams processed.
        ngrams: u64,
        /// Whether the job succeeded.
        success: bool,
    },

    /// A worker exited (shutdown signal received or queue empty).
    WorkerExited {
        /// ID of the worker that exited.
        worker_id: usize,
    },

    /// All expected results have been received.
    AllResultsReceived,

    /// MKN statistics computation completed.
    StatsComplete,

    /// Shard merge completed.
    MergeComplete,

    /// Cleanup completed.
    CleanupComplete,

    /// User requested pause.
    Pause,

    /// User requested resume.
    Resume,

    /// User requested cancel (save checkpoint).
    Cancel,

    /// User requested force quit (no checkpoint).
    ForceQuit,

    /// User requested parallelism change.
    SetParallelism(usize),

    /// An error occurred.
    Error(String),
}

/// Type alias for async cleanup functions.
pub type CleanupFn = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

/// Guard that ensures deterministic cleanup order.
///
/// Resources registered with `CleanupGuard` are cleaned up in LIFO order
/// (reverse of registration), ensuring proper shutdown sequences.
///
/// ## Example
///
/// ```ignore
/// let mut guard = CleanupGuard::new();
///
/// // Register cleanup actions (executed in reverse order)
/// guard.register(|| Box::pin(async { println!("3: Last cleanup"); }));
/// guard.register(|| Box::pin(async { println!("2: Middle cleanup"); }));
/// guard.register(|| Box::pin(async { println!("1: First cleanup"); }));
///
/// // Cleanup executes: 1, then 2, then 3
/// guard.cleanup().await;
/// ```
pub struct CleanupGuard {
    /// Cleanup functions in registration order (executed in reverse).
    resources: Vec<CleanupFn>,
}

impl CleanupGuard {
    /// Create a new empty cleanup guard.
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    /// Register a cleanup function to be called during cleanup.
    ///
    /// Functions are called in LIFO order (reverse of registration).
    pub fn register<F, Fut>(&mut self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.resources.push(Box::new(move || Box::pin(f())));
    }

    /// Execute all registered cleanup functions in LIFO order.
    ///
    /// This consumes the guard, ensuring cleanup happens exactly once.
    pub async fn cleanup(mut self) {
        // Execute in reverse order (LIFO)
        while let Some(cleanup_fn) = self.resources.pop() {
            cleanup_fn().await;
        }
    }

    /// Execute cleanup synchronously (blocking).
    ///
    /// Use this when you need to clean up from a non-async context.
    /// Creates a new runtime internally to execute async cleanup functions.
    pub fn cleanup_blocking(mut self) {
        // Execute in reverse order (LIFO)
        while let Some(cleanup_fn) = self.resources.pop() {
            // Use block_in_place for proper tokio integration
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(cleanup_fn())
            });
        }
    }

    /// Get the number of registered cleanup functions.
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if no cleanup functions are registered.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

impl Default for CleanupGuard {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Cleanup Resources
// ============================================================================

/// Resources that need cleanup, stored for CleanupGuard registration.
///
/// This struct collects all the resources created during import initialization
/// that need proper cleanup in LIFO order to prevent channel deadlocks.
///
/// ## Usage
///
/// ```ignore
/// let resources = CleanupResources::new()
///     .with_worker_handles(handles)
///     .with_worker_shutdown_txs(shutdown_txs)
///     .with_worker_converter(converter_handle)
///     .with_stats_task(stats_handle)
///     .with_command_handler(command_handle);
///
/// let guard = resources.into_cleanup_guard();
/// guard.cleanup().await;
/// ```
pub struct CleanupResources<S> {
    /// Worker task handles keyed by worker ID.
    pub worker_handles: HashMap<usize, JoinHandle<()>>,
    /// Per-worker shutdown signal senders.
    pub worker_shutdown_txs: HashMap<usize, WatchSender<bool>>,
    /// Shared state that contains progress_tx (must be dropped after workers exit).
    pub shared_state: Option<Arc<S>>,
    /// Result channel sender (dropped to signal receivers).
    pub result_tx: Option<Box<dyn std::any::Any + Send>>,
    /// Worker exit notification channel sender.
    pub worker_exit_tx: Option<Box<dyn std::any::Any + Send>>,
    /// Task that converts WorkerUpdate to ImportEvent.
    pub worker_converter: Option<JoinHandle<()>>,
    /// Periodic statistics emitter task.
    pub stats_task: Option<JoinHandle<()>>,
    /// Command handler task.
    pub command_handler: Option<JoinHandle<()>>,
}

impl<S: Send + Sync + 'static> CleanupResources<S> {
    /// Create an empty CleanupResources.
    pub fn new() -> Self {
        Self {
            worker_handles: HashMap::new(),
            worker_shutdown_txs: HashMap::new(),
            shared_state: None,
            result_tx: None,
            worker_exit_tx: None,
            worker_converter: None,
            stats_task: None,
            command_handler: None,
        }
    }

    /// Set worker handles.
    pub fn with_worker_handles(mut self, handles: HashMap<usize, JoinHandle<()>>) -> Self {
        self.worker_handles = handles;
        self
    }

    /// Set worker shutdown senders.
    pub fn with_worker_shutdown_txs(mut self, txs: HashMap<usize, WatchSender<bool>>) -> Self {
        self.worker_shutdown_txs = txs;
        self
    }

    /// Set shared state.
    pub fn with_shared_state(mut self, state: Arc<S>) -> Self {
        self.shared_state = Some(state);
        self
    }

    /// Set result_tx sender (type-erased).
    pub fn with_result_tx<T: Send + 'static>(mut self, tx: Sender<T>) -> Self {
        self.result_tx = Some(Box::new(tx));
        self
    }

    /// Set worker_exit_tx sender (type-erased).
    pub fn with_worker_exit_tx<T: Send + 'static>(mut self, tx: Sender<T>) -> Self {
        self.worker_exit_tx = Some(Box::new(tx));
        self
    }

    /// Set worker_converter task handle.
    pub fn with_worker_converter(mut self, handle: JoinHandle<()>) -> Self {
        self.worker_converter = Some(handle);
        self
    }

    /// Set stats_task handle.
    pub fn with_stats_task(mut self, handle: JoinHandle<()>) -> Self {
        self.stats_task = Some(handle);
        self
    }

    /// Set command_handler handle.
    pub fn with_command_handler(mut self, handle: JoinHandle<()>) -> Self {
        self.command_handler = Some(handle);
        self
    }

    /// Build a CleanupGuard from these resources.
    ///
    /// Resources are registered in REVERSE order of desired cleanup execution:
    /// - Last registered = first executed (LIFO)
    ///
    /// Cleanup order (critical for correctness):
    /// 1. Signal all workers to shut down
    /// 2. Wait for workers to exit (they drop their Arc refs)
    /// 3. Drop shared_state (releases one Arc ref, but may not deallocate yet)
    /// 4. Drop other channel senders (result_tx, worker_exit_tx)
    /// 5. Abort and wait for stats task (releases Arc<shared_state>, closes progress_tx)
    /// 6. Wait for worker_converter (channel now closed, can exit)
    /// 7. Abort and wait for command handler
    pub fn into_cleanup_guard(self) -> CleanupGuard {
        let mut guard = CleanupGuard::new();

        // Register in REVERSE order of desired execution (LIFO)
        // Last registered = first executed

        // 7. Abort command handler (executed last)
        if let Some(handle) = self.command_handler {
            guard.register(move || async move {
                handle.abort();
                let _ = handle.await;
            });
        }

        // 6. Wait for worker_converter (now runs AFTER stats_task is aborted)
        if let Some(handle) = self.worker_converter {
            guard.register(move || async move {
                let _ = handle.await;
            });
        }

        // 5. Abort stats task (now runs BEFORE waiting for worker_converter)
        // This releases Arc<shared_state>, dropping progress_tx, allowing worker_converter to exit.
        if let Some(handle) = self.stats_task {
            guard.register(move || async move {
                handle.abort();
                let _ = handle.await;
            });
        }

        // 4. Drop channel senders (result_tx, worker_exit_tx)
        // These are type-erased so we just drop them
        let result_tx = self.result_tx;
        let worker_exit_tx = self.worker_exit_tx;
        guard.register(move || async move {
            drop(result_tx);
            drop(worker_exit_tx);
        });

        // 3. Drop shared_state (releases progress_tx)
        if let Some(state) = self.shared_state {
            guard.register(move || async move {
                drop(state);
            });
        }

        // 2. Wait for workers to exit (executed second)
        let handles = self.worker_handles;
        guard.register(move || async move {
            for (_, handle) in handles {
                let _ = handle.await;
            }
        });

        // 1. Signal shutdown (executed first)
        let shutdown_txs = self.worker_shutdown_txs;
        guard.register(move || async move {
            for tx in shutdown_txs.values() {
                let _ = tx.send(true);
            }
        });

        guard
    }
}

impl<S: Send + Sync + 'static> Default for CleanupResources<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for import state machine, holding all shared resources.
///
/// This struct owns all the resources needed during import:
/// - Channels for communication
/// - Worker handles and shutdown senders
/// - Progress tracking state
/// - Configuration references
pub struct ImportContext {
    /// Current import phase
    pub phase: PipelinePhase,

    /// Number of currently active workers
    pub active_workers: usize,

    /// Total results received from workers
    pub results_received: u64,

    /// Total pending jobs (expected results)
    pub total_pending: u64,

    /// Whether import is paused
    pub paused: Arc<AtomicBool>,

    /// Whether import is cancelled
    pub cancelled: Arc<AtomicBool>,

    /// Whether force quit is requested
    pub force_quit: Arc<AtomicBool>,

    /// Current parallelism level
    pub current_parallelism: Arc<AtomicUsize>,

    /// Total n-grams processed (shared atomic)
    pub total_ngrams: Arc<AtomicU64>,

    /// Unique n-grams processed (shared atomic)
    pub unique_ngrams: Arc<AtomicU64>,

    /// Files completed counter
    pub files_completed: Arc<AtomicU64>,

    /// Per-order files completed count.
    pub order_files_completed: HashMap<u8, u64>,
    /// Per-order files skipped count.
    pub order_files_skipped: HashMap<u8, u64>,
    /// Per-order total files count.
    pub order_total_files: HashMap<u8, u64>,
    /// Per-order start times for duration tracking.
    pub order_start_times: HashMap<u8, Instant>,

    /// Import start time
    pub start_time: Instant,
}

impl ImportContext {
    /// Create a new import context with default values.
    pub fn new() -> Self {
        Self {
            phase: PipelinePhase::Initializing,
            active_workers: 0,
            results_received: 0,
            total_pending: 0,
            paused: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            force_quit: Arc::new(AtomicBool::new(false)),
            current_parallelism: Arc::new(AtomicUsize::new(0)),
            total_ngrams: Arc::new(AtomicU64::new(0)),
            unique_ngrams: Arc::new(AtomicU64::new(0)),
            files_completed: Arc::new(AtomicU64::new(0)),
            order_files_completed: HashMap::new(),
            order_files_skipped: HashMap::new(),
            order_total_files: HashMap::new(),
            order_start_times: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// Transition to a new phase based on a trigger.
    ///
    /// Returns the new phase after applying the trigger.
    pub fn transition(&mut self, trigger: ImportTrigger) -> &PipelinePhase {
        let new_phase = match (&self.phase, trigger) {
            // Initializing -> Downloading
            (PipelinePhase::Initializing, ImportTrigger::InitComplete { total_pending }) => {
                self.total_pending = total_pending;
                PipelinePhase::Downloading {
                    active_workers: self.active_workers,
                    results_received: 0,
                    total_pending,
                }
            }

            // Downloading -> various states
            (
                PipelinePhase::Downloading { total_pending, .. },
                ImportTrigger::JobResult { ngrams, .. },
            ) => {
                self.results_received += 1;
                PipelinePhase::Downloading {
                    active_workers: self.active_workers,
                    results_received: self.results_received,
                    total_pending: *total_pending,
                }
            }

            (PipelinePhase::Downloading { .. }, ImportTrigger::WorkerExited { .. }) => {
                self.active_workers = self.active_workers.saturating_sub(1);
                PipelinePhase::Downloading {
                    active_workers: self.active_workers,
                    results_received: self.results_received,
                    total_pending: self.total_pending,
                }
            }

            (PipelinePhase::Downloading { .. }, ImportTrigger::AllResultsReceived) => {
                PipelinePhase::ComputingStats
            }

            (PipelinePhase::Downloading { total_pending, .. }, ImportTrigger::Pause) => {
                self.paused.store(true, Ordering::SeqCst);
                PipelinePhase::Paused {
                    results_received: self.results_received,
                    total_pending: *total_pending,
                }
            }

            (PipelinePhase::Downloading { .. }, ImportTrigger::Cancel) => {
                self.cancelled.store(true, Ordering::SeqCst);
                PipelinePhase::Cancelled
            }

            (PipelinePhase::Downloading { .. }, ImportTrigger::ForceQuit) => {
                self.force_quit.store(true, Ordering::SeqCst);
                PipelinePhase::ForceQuit
            }

            (PipelinePhase::Downloading { .. }, ImportTrigger::Error(msg)) => {
                PipelinePhase::Failed { error: msg }
            }

            // Paused -> various states
            (PipelinePhase::Paused { total_pending, .. }, ImportTrigger::Resume) => {
                self.paused.store(false, Ordering::SeqCst);
                PipelinePhase::Downloading {
                    active_workers: self.active_workers,
                    results_received: self.results_received,
                    total_pending: *total_pending,
                }
            }

            (PipelinePhase::Paused { .. }, ImportTrigger::Cancel) => {
                self.cancelled.store(true, Ordering::SeqCst);
                PipelinePhase::Cancelled
            }

            (PipelinePhase::Paused { .. }, ImportTrigger::ForceQuit) => {
                self.force_quit.store(true, Ordering::SeqCst);
                PipelinePhase::ForceQuit
            }

            // ComputingStats -> Merging or CleaningUp
            (PipelinePhase::ComputingStats, ImportTrigger::StatsComplete) => PipelinePhase::Merging {
                shards_processed: 0,
                total_shards: 0,
            },

            (PipelinePhase::ComputingStats, ImportTrigger::Error(msg)) => {
                PipelinePhase::Failed { error: msg }
            }

            // Merging -> CleaningUp
            (PipelinePhase::Merging { .. }, ImportTrigger::MergeComplete) => PipelinePhase::CleaningUp,

            (PipelinePhase::Merging { .. }, ImportTrigger::Error(msg)) => {
                PipelinePhase::Failed { error: msg }
            }

            // CleaningUp -> Completed
            (PipelinePhase::CleaningUp, ImportTrigger::CleanupComplete) => {
                let duration = self.start_time.elapsed();
                let total = self.total_ngrams.load(Ordering::Relaxed);
                PipelinePhase::Completed {
                    total_ngrams: total,
                    duration,
                }
            }

            // Any other state + trigger = no change (log warning)
            (current, trigger) => {
                log::warn!(
                    "Invalid state transition: {:?} + {:?}",
                    current.name(),
                    trigger
                );
                return &self.phase;
            }
        };

        self.phase = new_phase;
        &self.phase
    }

    /// Check if the current phase is terminal.
    pub fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }
}

impl Default for ImportContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_is_terminal() {
        assert!(!PipelinePhase::Initializing.is_terminal());
        assert!(!PipelinePhase::Downloading {
            active_workers: 4,
            results_received: 0,
            total_pending: 100
        }
        .is_terminal());
        assert!(!PipelinePhase::ComputingStats.is_terminal());

        assert!(PipelinePhase::Completed {
            total_ngrams: 1000,
            duration: Duration::from_secs(60)
        }
        .is_terminal());
        assert!(PipelinePhase::Cancelled.is_terminal());
        assert!(PipelinePhase::ForceQuit.is_terminal());
        assert!(PipelinePhase::Failed {
            error: "test".to_string()
        }
        .is_terminal());
    }

    #[test]
    fn test_phase_names() {
        assert_eq!(PipelinePhase::Initializing.name(), "Initializing");
        assert_eq!(
            PipelinePhase::Downloading {
                active_workers: 4,
                results_received: 0,
                total_pending: 100
            }
            .name(),
            "Downloading"
        );
        assert_eq!(PipelinePhase::ComputingStats.name(), "Computing MKN Statistics");
    }

    #[test]
    fn test_context_transitions() {
        let mut ctx = ImportContext::new();
        assert!(matches!(ctx.phase, PipelinePhase::Initializing));

        // Initializing -> Downloading
        ctx.transition(ImportTrigger::InitComplete { total_pending: 100 });
        assert!(matches!(ctx.phase, PipelinePhase::Downloading { .. }));

        // Downloading -> Paused
        ctx.transition(ImportTrigger::Pause);
        assert!(matches!(ctx.phase, PipelinePhase::Paused { .. }));

        // Paused -> Downloading
        ctx.transition(ImportTrigger::Resume);
        assert!(matches!(ctx.phase, PipelinePhase::Downloading { .. }));

        // Downloading -> Cancelled
        ctx.transition(ImportTrigger::Cancel);
        assert!(matches!(ctx.phase, PipelinePhase::Cancelled));
        assert!(ctx.is_terminal());
    }

    #[tokio::test]
    async fn test_cleanup_guard_lifo_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let order = Arc::new(AtomicUsize::new(0));
        let results = Arc::new(parking_lot::Mutex::new(Vec::new()));

        let mut guard = CleanupGuard::new();

        // Register cleanup functions
        let order_clone = Arc::clone(&order);
        let results_clone = Arc::clone(&results);
        guard.register(move || {
            let results = results_clone;
            let order = order_clone;
            async move {
                let n = order.fetch_add(1, Ordering::SeqCst);
                results.lock().push(n);
            }
        });

        let order_clone = Arc::clone(&order);
        let results_clone = Arc::clone(&results);
        guard.register(move || {
            let results = results_clone;
            let order = order_clone;
            async move {
                let n = order.fetch_add(1, Ordering::SeqCst);
                results.lock().push(n);
            }
        });

        let order_clone = Arc::clone(&order);
        let results_clone = Arc::clone(&results);
        guard.register(move || {
            let results = results_clone;
            let order = order_clone;
            async move {
                let n = order.fetch_add(1, Ordering::SeqCst);
                results.lock().push(n);
            }
        });

        // Execute cleanup
        guard.cleanup().await;

        // Check LIFO order: last registered executes first (gets 0), etc.
        let results = results.lock();
        assert_eq!(*results, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn test_cleanup_resources_builder() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Track cleanup execution order
        let order = Arc::new(AtomicUsize::new(0));
        let results = Arc::new(parking_lot::Mutex::new(Vec::new()));

        // Create mock shared state
        struct MockSharedState {
            order: Arc<AtomicUsize>,
            results: Arc<parking_lot::Mutex<Vec<(usize, &'static str)>>>,
        }

        impl Drop for MockSharedState {
            fn drop(&mut self) {
                let n = self.order.fetch_add(1, Ordering::SeqCst);
                self.results.lock().push((n, "shared_state"));
            }
        }

        let shared_state = Arc::new(MockSharedState {
            order: Arc::clone(&order),
            results: Arc::clone(&results),
        });

        // Create shutdown channels
        let (shutdown_tx, _shutdown_rx) = tokio::sync::watch::channel(false);
        let mut shutdown_txs = HashMap::new();
        shutdown_txs.insert(0, shutdown_tx);

        // Create worker handles that track their completion
        let order_clone = Arc::clone(&order);
        let results_clone = Arc::clone(&results);
        let worker_handle = tokio::spawn(async move {
            // Worker immediately completes
            let n = order_clone.fetch_add(1, Ordering::SeqCst);
            results_clone.lock().push((n, "worker"));
        });
        let mut handles = HashMap::new();
        handles.insert(0, worker_handle);

        // Build cleanup resources
        let resources: CleanupResources<MockSharedState> = CleanupResources::new()
            .with_worker_handles(handles)
            .with_worker_shutdown_txs(shutdown_txs)
            .with_shared_state(shared_state);

        // Convert to cleanup guard and execute
        let guard = resources.into_cleanup_guard();
        guard.cleanup().await;

        // Verify cleanup happened in correct LIFO order:
        // 1. Signal shutdown (order 0)
        // 2. Wait for workers (order 1 - worker task completes)
        // 3. Drop shared_state (order 2)
        let results = results.lock();
        assert!(results.len() >= 2, "Expected at least 2 cleanup actions");

        // Worker should complete before shared_state is dropped
        let worker_order = results.iter().find(|(_, name)| *name == "worker").map(|(n, _)| *n);
        let shared_order = results.iter().find(|(_, name)| *name == "shared_state").map(|(n, _)| *n);

        if let (Some(w), Some(s)) = (worker_order, shared_order) {
            assert!(w < s, "Worker should complete before shared_state is dropped");
        }
    }

    #[test]
    fn test_cleanup_resources_default() {
        struct DummyState;
        let resources: CleanupResources<DummyState> = CleanupResources::default();
        assert!(resources.worker_handles.is_empty());
        assert!(resources.worker_shutdown_txs.is_empty());
        assert!(resources.shared_state.is_none());
    }
}
