//! Lock-free reactive state machine scheduler for periodic tasks.
//!
//! This module implements a cron-like task scheduler using a reactive state machine
//! design with non-blocking algorithms and atomics.
//!
//! ## Design Goals
//!
//! 1. **Explicit states with clear transitions** - States and events are enumerated
//! 2. **Event-driven architecture** - No polling loops with scattered conditionals
//! 3. **Lock-free task submission** - MPSC channel for concurrent task submission
//! 4. **Thread-local min-heap** - Priority queue owned by the scheduler thread
//! 5. **Graceful shutdown** - Termination signal checked at every state transition
//!
//! ## State Machine Design
//!
//! ```text
//!                               ┌─────────────────────────────────────────┐
//!                               │                                         │
//!                               ▼                                         │
//!     ┌─────────┐  channel has  ┌──────────────┐  task due    ┌──────────────────┐
//!     │  Idle   │ ────────────▶ │ DrainChannel │ ───────────▶ │ ExecutingTask    │
//!     └─────────┘   messages    └──────────────┘              └──────────────────┘
//!          │                          │                              │
//!          │                          │ channel empty                │ task complete
//!          │                          │ & no tasks due               │ (requeue if
//!          │                          ▼                              │  recurring)
//!          │                    ┌──────────────┐                     │
//!          │                    │   Sleeping   │◀────────────────────┘
//!          │                    └──────────────┘
//!          │                          │
//!          │                          │ timer expired
//!          │                          ▼
//!          │                    ┌──────────────┐
//!          └───────────────────▶│ CheckEvents  │◀──── (loop back)
//!                               └──────────────┘
//!                                     │
//!                                     │ terminating == true
//!                                     ▼
//!                               ┌──────────────┐
//!                               │  Terminated  │
//!                               └──────────────┘
//! ```
//!
//! ## Lock-Free Guarantees
//!
//! | Component           | Synchronization            | Lock-Free? |
//! |---------------------|----------------------------|------------|
//! | Task submission     | crossbeam-channel (MPSC)   | ✅ Yes     |
//! | Termination flag    | AtomicBool                 | ✅ Yes     |
//! | Statistics          | AtomicU64                  | ✅ Yes     |
//! | State machine       | Thread-local, no sharing   | ✅ Yes     |
//! | Task queue          | BinaryHeap (thread-local)  | ✅ Yes     |
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::util::cron::{spawn_cron, TaskMetadata};
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
//!
//! let terminating = Arc::new(AtomicBool::new(false));
//! let (handle, thread, stats, _ready) = spawn_cron(Arc::clone(&terminating));
//!
//! // Schedule a recurring task every 5 seconds
//! let counter = Arc::new(AtomicU64::new(0));
//! let counter_clone = Arc::clone(&counter);
//! handle.schedule_recurring(0, 5000, "checkpoint", move || {
//!     counter_clone.fetch_add(1, Ordering::Relaxed);
//!     true
//! });
//!
//! // Wait some time...
//! std::thread::sleep(std::time::Duration::from_secs(20));
//!
//! // Graceful shutdown
//! handle.request_shutdown();
//! thread.join().unwrap();
//!
//! println!("Task executed {} times", counter.load(Ordering::Relaxed));
//! ```

use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};

/// Unix timestamp in milliseconds for scheduling precision.
pub type UnixTimestampMs = u64;

/// Get current Unix timestamp in milliseconds.
#[inline]
pub fn now_ms() -> UnixTimestampMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System time went backwards")
        .as_millis() as u64
}

/// State machine states for the cron scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronState {
    /// Initial state - evaluate event sources.
    CheckEvents,
    /// Draining incoming tasks from the channel.
    DrainChannel,
    /// Executing a task that is due.
    ExecutingTask,
    /// Sleeping until next task or poll interval.
    Sleeping,
    /// Terminal state - graceful shutdown complete.
    Terminated,
}

/// Events that drive state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronEvent {
    /// New task(s) available in channel.
    TaskReceived,
    /// Sleep timer expired.
    TimerExpired,
    /// A task in the queue is past its deadline.
    TaskDue,
    /// Task execution completed (with success flag).
    TaskCompleted {
        /// Whether the task returned true (success).
        success: bool,
        /// Whether the task should be requeued (recurring).
        should_requeue: bool,
    },
    /// External termination signal.
    TerminationRequested,
    /// All senders dropped, channel closed.
    ChannelDisconnected,
    /// No events pending (idle).
    NoEvents,
}

/// Metadata for a scheduled task.
#[derive(Clone, Debug)]
pub enum TaskMetadata {
    /// One-shot task - executed once and discarded.
    OneShot,

    /// Recurring task - re-queued after completion.
    Recurring {
        /// Interval in milliseconds between executions.
        interval_ms: u64,
    },

    /// Named task with custom metadata.
    Named {
        /// Task name for logging/debugging.
        name: String,
        /// Recurrence interval (None = one-shot).
        recurring_interval_ms: Option<u64>,
    },
}

impl TaskMetadata {
    /// Get the recurrence interval if this task recurs.
    #[inline]
    pub fn recurrence_interval(&self) -> Option<u64> {
        match self {
            TaskMetadata::OneShot => None,
            TaskMetadata::Recurring { interval_ms } => Some(*interval_ms),
            TaskMetadata::Named {
                recurring_interval_ms,
                ..
            } => *recurring_interval_ms,
        }
    }

    /// Get the task name for logging.
    pub fn name(&self) -> &str {
        match self {
            TaskMetadata::OneShot => "one-shot",
            TaskMetadata::Recurring { .. } => "recurring",
            TaskMetadata::Named { name, .. } => name,
        }
    }
}

/// A scheduled task with execution time and callback.
pub struct ScheduledTask {
    /// Unix timestamp (ms) when task should execute.
    pub scheduled_time_ms: UnixTimestampMs,

    /// Task metadata (one-shot, recurring, etc.).
    pub metadata: TaskMetadata,

    /// The task to execute. Returns `true` if successful.
    pub task: Box<dyn FnMut() -> bool + Send>,
}

impl std::fmt::Debug for ScheduledTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScheduledTask")
            .field("scheduled_time_ms", &self.scheduled_time_ms)
            .field("metadata", &self.metadata)
            .field("task", &"<fn>")
            .finish()
    }
}

// Implement ordering for min-heap (earliest timestamp first)
impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (BinaryHeap is max-heap by default)
        other.scheduled_time_ms.cmp(&self.scheduled_time_ms)
    }
}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.scheduled_time_ms == other.scheduled_time_ms
    }
}

impl Eq for ScheduledTask {}

/// Statistics for the cron manager (lock-free).
#[derive(Default)]
pub struct CronStats {
    /// Total tasks executed.
    pub tasks_executed: AtomicU64,
    /// Tasks that returned false (failed).
    pub tasks_failed: AtomicU64,
    /// Tasks that panicked.
    pub tasks_panicked: AtomicU64,
    /// State transitions performed.
    pub transitions: AtomicU64,
}

impl CronStats {
    /// Record a successful task execution.
    #[inline]
    fn record_success(&self) {
        self.tasks_executed.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record a failed task execution.
    #[inline]
    fn record_failure(&self) {
        self.tasks_executed.fetch_add(1, AtomicOrdering::Relaxed);
        self.tasks_failed.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record a panicked task.
    #[inline]
    fn record_panic(&self) {
        self.tasks_executed.fetch_add(1, AtomicOrdering::Relaxed);
        self.tasks_panicked.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record a state transition.
    #[inline]
    fn record_transition(&self) {
        self.transitions.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Get snapshot of current statistics.
    pub fn snapshot(&self) -> CronStatsSnapshot {
        CronStatsSnapshot {
            tasks_executed: self.tasks_executed.load(AtomicOrdering::Relaxed),
            tasks_failed: self.tasks_failed.load(AtomicOrdering::Relaxed),
            tasks_panicked: self.tasks_panicked.load(AtomicOrdering::Relaxed),
            transitions: self.transitions.load(AtomicOrdering::Relaxed),
        }
    }
}

/// Immutable snapshot of cron statistics.
#[derive(Debug, Clone, Copy)]
pub struct CronStatsSnapshot {
    /// Total tasks executed.
    pub tasks_executed: u64,
    /// Tasks that returned false (failed).
    pub tasks_failed: u64,
    /// Tasks that panicked.
    pub tasks_panicked: u64,
    /// State transitions performed.
    pub transitions: u64,
}

/// Lock-free reactive state machine scheduler.
///
/// Explicit states and transitions make control flow clear:
/// - No scattered termination checks
/// - No nested conditionals
/// - Easy to reason about and test
///
/// Uses only lock-free primitives:
/// - AtomicBool for termination
/// - Lock-free MPSC channel for task submission
/// - Thread-local BinaryHeap (no sharing)
pub struct CronStateMachine {
    /// Current state.
    state: CronState,

    /// Task queue (owned by this thread - no sharing).
    queue: BinaryHeap<ScheduledTask>,

    /// Receiver for new tasks (lock-free MPSC).
    task_rx: Receiver<ScheduledTask>,

    /// Poll interval in milliseconds.
    poll_interval_ms: u64,

    /// Termination flag (atomic - no lock).
    terminating: Arc<AtomicBool>,

    /// Channel disconnected flag (set once, never cleared).
    channel_disconnected: bool,

    /// Statistics (atomic counters - no lock).
    stats: Arc<CronStats>,

    /// One-shot ready signal sender (sent at start of run()).
    ready_tx: Option<Sender<()>>,
}

impl CronStateMachine {
    /// Default poll interval: 100ms
    pub const DEFAULT_POLL_INTERVAL_MS: u64 = 100;

    /// Create a new CronStateMachine.
    ///
    /// # Arguments
    /// * `task_rx` - Receiver for new tasks from the channel
    /// * `terminating` - Atomic flag for graceful shutdown
    /// * `stats` - Shared statistics counters
    /// * `poll_interval_ms` - Maximum sleep duration between polls
    /// * `ready_tx` - Optional one-shot channel to signal when event loop starts
    pub fn new(
        task_rx: Receiver<ScheduledTask>,
        terminating: Arc<AtomicBool>,
        stats: Arc<CronStats>,
        poll_interval_ms: u64,
        ready_tx: Option<Sender<()>>,
    ) -> Self {
        Self {
            state: CronState::CheckEvents,
            queue: BinaryHeap::new(),
            task_rx,
            poll_interval_ms,
            terminating,
            channel_disconnected: false,
            stats,
            ready_tx,
        }
    }

    /// Run the state machine until termination (call from dedicated thread).
    ///
    /// This is the main entry point - drives the state machine to completion.
    ///
    /// **Important**: The ready signal is sent at the START of this method,
    /// INSIDE the event loop, ensuring that any tasks scheduled after the
    /// caller receives the ready signal will be processed by this event loop.
    pub fn run(&mut self) {
        log::info!(
            "CronStateMachine started (poll={}ms, lock-free reactive design)",
            self.poll_interval_ms
        );

        // Signal ready INSIDE the event loop (not before it starts).
        // This ensures tasks scheduled after receiving the ready signal
        // will be processed by this event loop iteration.
        if let Some(tx) = self.ready_tx.take() {
            let _ = tx.send(());
        }

        // Drive state machine until terminal state
        while self.state != CronState::Terminated {
            let event = self.poll_event();
            self.transition(event);
        }

        log::info!(
            "CronStateMachine terminated (executed={}, failed={}, panicked={}, transitions={})",
            self.stats.tasks_executed.load(AtomicOrdering::Relaxed),
            self.stats.tasks_failed.load(AtomicOrdering::Relaxed),
            self.stats.tasks_panicked.load(AtomicOrdering::Relaxed),
            self.stats.transitions.load(AtomicOrdering::Relaxed),
        );
    }

    /// Poll for the next event based on current state.
    ///
    /// This is the "sense" phase of the reactive loop.
    ///
    /// **Important**: Due tasks are checked BEFORE termination to ensure
    /// that tasks which are already due get executed even if termination
    /// is requested while the scheduler was sleeping.
    fn poll_event(&mut self) -> CronEvent {
        // Check for due tasks FIRST - they have priority over termination
        // This ensures tasks that became due while sleeping get executed
        if let Some(task) = self.queue.peek() {
            if task.scheduled_time_ms <= now_ms() {
                return CronEvent::TaskDue;
            }
        }

        // Now check termination (atomic load)
        if self.terminating.load(AtomicOrdering::Acquire) {
            return CronEvent::TerminationRequested;
        }

        match self.state {
            CronState::CheckEvents => self.poll_check_events(),
            CronState::DrainChannel => self.poll_drain_channel(),
            CronState::ExecutingTask => unreachable!("ExecutingTask polls internally"),
            CronState::Sleeping => CronEvent::TimerExpired, // Just woke up
            CronState::Terminated => unreachable!("Cannot poll from Terminated"),
        }
    }

    /// Poll events in CheckEvents state.
    fn poll_check_events(&mut self) -> CronEvent {
        // Priority: termination > channel messages > due tasks > sleep
        if self.terminating.load(AtomicOrdering::Acquire) {
            return CronEvent::TerminationRequested;
        }

        // Check channel (non-blocking)
        match self.task_rx.try_recv() {
            Ok(task) => {
                self.queue.push(task);
                return CronEvent::TaskReceived;
            }
            Err(TryRecvError::Disconnected) if !self.channel_disconnected => {
                self.channel_disconnected = true;
                return CronEvent::ChannelDisconnected;
            }
            _ => {}
        }

        // Check for due tasks
        if let Some(task) = self.queue.peek() {
            if task.scheduled_time_ms <= now_ms() {
                return CronEvent::TaskDue;
            }
        }

        // Nothing to do
        CronEvent::NoEvents
    }

    /// Poll in DrainChannel state - continue draining or signal done.
    fn poll_drain_channel(&mut self) -> CronEvent {
        match self.task_rx.try_recv() {
            Ok(task) => {
                self.queue.push(task);
                CronEvent::TaskReceived
            }
            Err(TryRecvError::Empty) => {
                // Done draining, check for due tasks
                if let Some(task) = self.queue.peek() {
                    if task.scheduled_time_ms <= now_ms() {
                        return CronEvent::TaskDue;
                    }
                }
                CronEvent::NoEvents
            }
            Err(TryRecvError::Disconnected) => {
                self.channel_disconnected = true;
                CronEvent::ChannelDisconnected
            }
        }
    }

    /// State transition function: (State, Event) → State
    ///
    /// This is the core of the reactive design - a pure function
    /// from (state, event) to new state with side effects.
    fn transition(&mut self, event: CronEvent) {
        self.stats.record_transition();
        let old_state = self.state;

        self.state = match (self.state, event) {
            // === Termination (highest priority, from any state) ===
            (_, CronEvent::TerminationRequested) => {
                log::debug!(
                    "Transition: {:?} + TerminationRequested → Terminated",
                    old_state
                );
                CronState::Terminated
            }

            // === CheckEvents transitions ===
            (CronState::CheckEvents, CronEvent::TaskReceived) => {
                log::trace!("Transition: CheckEvents + TaskReceived → DrainChannel");
                CronState::DrainChannel
            }
            (CronState::CheckEvents, CronEvent::TaskDue) => {
                self.execute_one_task();
                CronState::CheckEvents // Re-check after execution
            }
            (CronState::CheckEvents, CronEvent::NoEvents) => CronState::Sleeping,
            (CronState::CheckEvents, CronEvent::ChannelDisconnected) => {
                log::debug!("Channel disconnected, continuing with existing tasks");
                if self.queue.is_empty() {
                    CronState::Terminated
                } else {
                    CronState::CheckEvents
                }
            }

            // === DrainChannel transitions ===
            (CronState::DrainChannel, CronEvent::TaskReceived) => {
                // Stay in drain state until channel is empty
                CronState::DrainChannel
            }
            (CronState::DrainChannel, CronEvent::TaskDue) => {
                self.execute_one_task();
                CronState::CheckEvents
            }
            (CronState::DrainChannel, CronEvent::NoEvents) => CronState::Sleeping,
            (CronState::DrainChannel, CronEvent::ChannelDisconnected) => {
                log::debug!("Channel disconnected during drain");
                CronState::CheckEvents
            }

            // === Sleeping transitions ===
            (CronState::Sleeping, CronEvent::TimerExpired) => CronState::CheckEvents,
            (CronState::Sleeping, CronEvent::TaskDue) => {
                // Task became due while sleeping - execute it
                self.execute_one_task();
                CronState::CheckEvents
            }

            // === ExecutingTask transitions ===
            (CronState::ExecutingTask, CronEvent::TaskCompleted { success, .. }) => {
                if success {
                    self.stats.record_success();
                } else {
                    self.stats.record_failure();
                }
                // Requeuing handled in execute_one_task
                CronState::CheckEvents
            }

            // === Invalid transitions ===
            (CronState::Terminated, _) => {
                unreachable!("Cannot transition from Terminated")
            }

            // Catch-all for unexpected combinations
            (state, event) => {
                log::warn!("Unexpected transition: {:?} + {:?}", state, event);
                CronState::CheckEvents
            }
        };

        // Handle sleeping state entry (side effect)
        if self.state == CronState::Sleeping {
            self.do_sleep();
        }
    }

    /// Execute one due task (exception-safe).
    fn execute_one_task(&mut self) {
        let Some(mut task) = self.queue.pop() else {
            return;
        };

        log::trace!("Executing task: {}", task.metadata.name());

        // Execute with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (task.task)()));

        match result {
            Ok(true) => {
                self.stats.record_success();
                // Re-queue if recurring
                if let Some(interval) = task.metadata.recurrence_interval() {
                    task.scheduled_time_ms = now_ms() + interval;
                    self.queue.push(task);
                    log::trace!("Task requeued with interval {}ms", interval);
                }
            }
            Ok(false) => {
                self.stats.record_failure();
                log::warn!(
                    "Task '{}' returned false, not re-queuing",
                    task.metadata.name()
                );
            }
            Err(e) => {
                self.stats.record_panic();
                log::error!("Task '{}' panicked: {:?}", task.metadata.name(), e);
            }
        }
    }

    /// Sleep for the poll interval.
    fn do_sleep(&self) {
        // Calculate sleep duration: min(poll_interval, time_to_next_task)
        let sleep_ms = if let Some(task) = self.queue.peek() {
            let now = now_ms();
            if task.scheduled_time_ms <= now {
                0 // Task already due
            } else {
                (task.scheduled_time_ms - now).min(self.poll_interval_ms)
            }
        } else {
            self.poll_interval_ms
        };

        if sleep_ms > 0 {
            std::thread::sleep(Duration::from_millis(sleep_ms));
        }
    }

    /// Get number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Get current state (for testing/debugging).
    pub fn current_state(&self) -> CronState {
        self.state
    }
}

/// Handle for submitting tasks to a CronStateMachine (thread-safe, lock-free).
///
/// Uses a lock-free MPSC channel - no mutexes or locks.
/// Clone this handle to submit tasks from multiple threads.
#[derive(Clone)]
pub struct CronHandle {
    /// Sender end of lock-free task channel.
    task_tx: Sender<ScheduledTask>,

    /// Reference to termination flag for shutdown coordination.
    terminating: Arc<AtomicBool>,
}

impl CronHandle {
    /// Schedule a task at a specific Unix timestamp (ms).
    ///
    /// This is a lock-free operation using crossbeam-channel.
    /// Returns `true` if the task was submitted, `false` if channel disconnected.
    pub fn schedule_at<F>(&self, time_ms: UnixTimestampMs, metadata: TaskMetadata, task: F) -> bool
    where
        F: FnMut() -> bool + Send + 'static,
    {
        let scheduled_task = ScheduledTask {
            scheduled_time_ms: time_ms,
            metadata,
            task: Box::new(task),
        };
        self.task_tx.send(scheduled_task).is_ok()
    }

    /// Schedule a task after a delay (ms).
    pub fn schedule_after<F>(&self, delay_ms: u64, metadata: TaskMetadata, task: F) -> bool
    where
        F: FnMut() -> bool + Send + 'static,
    {
        self.schedule_at(now_ms() + delay_ms, metadata, task)
    }

    /// Schedule a recurring task.
    ///
    /// The task will execute after `initial_delay_ms` and then every `interval_ms`
    /// as long as it returns `true`. If the task returns `false`, it will not be
    /// rescheduled.
    pub fn schedule_recurring<F>(
        &self,
        initial_delay_ms: u64,
        interval_ms: u64,
        name: &str,
        task: F,
    ) -> bool
    where
        F: FnMut() -> bool + Send + 'static,
    {
        let metadata = TaskMetadata::Named {
            name: name.to_string(),
            recurring_interval_ms: Some(interval_ms),
        };
        self.schedule_after(initial_delay_ms, metadata, task)
    }

    /// Schedule a one-shot task after a delay.
    pub fn schedule_once<F>(&self, delay_ms: u64, name: &str, task: F) -> bool
    where
        F: FnMut() -> bool + Send + 'static,
    {
        let metadata = TaskMetadata::Named {
            name: name.to_string(),
            recurring_interval_ms: None,
        };
        self.schedule_after(delay_ms, metadata, task)
    }

    /// Request graceful shutdown of the state machine.
    ///
    /// This is a lock-free atomic store.
    pub fn request_shutdown(&self) {
        self.terminating.store(true, AtomicOrdering::Release);
    }

    /// Check if shutdown has been requested.
    pub fn is_shutting_down(&self) -> bool {
        self.terminating.load(AtomicOrdering::Acquire)
    }
}

/// Spawn the cron state machine on a dedicated thread.
///
/// Returns:
/// - `CronHandle` for submitting tasks (clone-able, thread-safe, lock-free)
/// - `JoinHandle` for the cron thread
/// - `Arc<CronStats>` for reading statistics (lock-free)
/// - `Receiver<()>` that signals when the scheduler is ready (receives one `()` when event loop starts)
pub fn spawn_cron(
    terminating: Arc<AtomicBool>,
) -> (
    CronHandle,
    std::thread::JoinHandle<()>,
    Arc<CronStats>,
    Receiver<()>,
) {
    spawn_cron_with_interval(terminating, CronStateMachine::DEFAULT_POLL_INTERVAL_MS)
}

/// Spawn with custom poll interval.
///
/// Returns:
/// - `CronHandle` for submitting tasks (clone-able, thread-safe, lock-free)
/// - `JoinHandle` for the cron thread
/// - `Arc<CronStats>` for reading statistics (lock-free)
/// - `Receiver<()>` that signals when the scheduler is ready (receives one `()` when event loop starts)
///
/// # Ready Signal
///
/// The returned `Receiver<()>` provides a one-shot signal that the scheduler is ready.
/// Call `ready_rx.recv()` to block until the cron thread has entered its event loop.
/// This prevents race conditions where tasks are scheduled before the scheduler is ready.
pub fn spawn_cron_with_interval(
    terminating: Arc<AtomicBool>,
    poll_interval_ms: u64,
) -> (
    CronHandle,
    std::thread::JoinHandle<()>,
    Arc<CronStats>,
    Receiver<()>,
) {
    // Lock-free unbounded MPSC channel for tasks
    let (task_tx, task_rx) = unbounded::<ScheduledTask>();

    // One-shot channel to signal when the scheduler is ready
    let (ready_tx, ready_rx) = unbounded::<()>();

    let stats = Arc::new(CronStats::default());
    let stats_clone = Arc::clone(&stats);
    let terminating_clone = Arc::clone(&terminating);

    let thread_handle = std::thread::Builder::new()
        .name("cron-state-machine".to_string())
        .spawn(move || {
            // Pass ready_tx to state machine - signal will be sent inside run()
            let mut sm = CronStateMachine::new(
                task_rx,
                terminating_clone,
                stats_clone,
                poll_interval_ms,
                Some(ready_tx),
            );

            sm.run();
        })
        .expect("Failed to spawn cron state machine thread");

    let handle = CronHandle {
        task_tx,
        terminating,
    };

    (handle, thread_handle, stats, ready_rx)
}

#[cfg(test)]
mod tests;
