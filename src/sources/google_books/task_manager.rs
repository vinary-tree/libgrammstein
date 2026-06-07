//! Dual-queue task manager for Google Books N-gram imports.
//!
//! This module provides a `TaskManager` abstraction that manages two queues:
//! 1. **Regular queue (FIFO)** - for new download tasks
//! 2. **Retry queue (min-heap)** - priority queue ordered by earliest retry time
//!
//! Workers call `get_next_task()` which checks the retry queue first (if head is ready),
//! otherwise polls the regular queue. This enables non-blocking retry handling where
//! workers can process other tasks while waiting for rate-limited retries.
//!
//! ## Features
//!
//! - **Non-blocking retries**: Workers don't block waiting for backoff timers
//! - **Retry-After support**: Respects HTTP Retry-After headers (seconds or HTTP-date)
//! - **Exponential backoff with jitter**: Configurable retry behavior
//! - **Observability**: Metrics for queue depths, retry counts, and failures
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::task_manager::{TaskManager, TaskManagerConfig};
//! use tokio::sync::watch;
//!
//! // Create shutdown signal
//! let (shutdown_tx, shutdown_rx) = watch::channel(false);
//!
//! // Create task manager
//! let config = TaskManagerConfig::default();
//! let (manager, submitter) = TaskManager::new(1000, config, shutdown_rx);
//!
//! // Submit tasks
//! submitter.submit(Job::new("https://...", "th", 1)).await?;
//!
//! // Workers get tasks
//! while let Some(job) = manager.get_next_task().await {
//!     // Process job...
//!     if should_retry {
//!         manager.schedule_retry(job.with_retry(), Some(retry_after));
//!     }
//! }
//! ```

#[cfg(feature = "google-books")]
use std::cmp::Reverse;
#[cfg(feature = "google-books")]
use std::collections::BinaryHeap;
#[cfg(feature = "google-books")]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
#[cfg(feature = "google-books")]
use std::sync::Arc;
#[cfg(feature = "google-books")]
use std::time::{Duration, Instant};

#[cfg(feature = "google-books")]
use chrono::{DateTime, Utc};
#[cfg(feature = "google-books")]
use tokio::sync::{mpsc, watch, Mutex};

// ============================================================================
// RetryAfter Parsing
// ============================================================================

/// Parsed Retry-After header value.
///
/// The HTTP Retry-After header can specify either:
/// - Seconds: "120" (retry after 120 seconds)
/// - HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT"
#[cfg(feature = "google-books")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryAfter {
    /// Retry after a number of seconds from now.
    Seconds(u64),
    /// Retry after a specific date/time.
    DateTime(DateTime<Utc>),
}

#[cfg(feature = "google-books")]
impl RetryAfter {
    /// Parse a Retry-After header value.
    ///
    /// Supports two formats:
    /// - Seconds: "120" (numeric seconds)
    /// - HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT"
    ///
    /// Returns `None` if the value cannot be parsed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let seconds = RetryAfter::parse("120");
    /// assert_eq!(seconds, Some(RetryAfter::Seconds(120)));
    ///
    /// let date = RetryAfter::parse("Wed, 21 Oct 2015 07:28:00 GMT");
    /// assert!(matches!(date, Some(RetryAfter::DateTime(_))));
    /// ```
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();

        // Try parsing as seconds first (most common)
        if let Ok(seconds) = value.parse::<u64>() {
            return Some(RetryAfter::Seconds(seconds));
        }

        // Try parsing as HTTP-date (RFC 7231 format)
        // Format: "Wed, 21 Oct 2015 07:28:00 GMT"
        if let Ok(dt) = DateTime::parse_from_rfc2822(value) {
            return Some(RetryAfter::DateTime(dt.with_timezone(&Utc)));
        }

        // Try parsing as RFC 3339 (ISO 8601) as a fallback
        if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
            return Some(RetryAfter::DateTime(dt.with_timezone(&Utc)));
        }

        None
    }

    /// Convert to a Duration from now.
    ///
    /// For `Seconds`, returns the duration directly.
    /// For `DateTime`, calculates the duration until that time.
    /// If the datetime is in the past, returns `Duration::ZERO`.
    pub fn to_duration(&self) -> Duration {
        match self {
            RetryAfter::Seconds(s) => Duration::from_secs(*s),
            RetryAfter::DateTime(dt) => {
                let now = Utc::now();
                if *dt > now {
                    let diff = (*dt - now).num_milliseconds();
                    Duration::from_millis(diff.max(0) as u64)
                } else {
                    Duration::ZERO
                }
            }
        }
    }

    /// Convert to an Instant for priority queue ordering.
    pub fn to_instant(&self) -> Instant {
        Instant::now() + self.to_duration()
    }
}

// ============================================================================
// Job Structure
// ============================================================================

/// Maximum retry attempts for transient failures.
#[cfg(feature = "google-books")]
pub const MAX_RETRIES: u8 = 5;

/// Initial backoff delay in milliseconds (doubles each retry).
#[cfg(feature = "google-books")]
pub const INITIAL_BACKOFF_MS: u64 = 1000;

/// A job for the worker pool to process.
#[cfg(feature = "google-books")]
#[derive(Clone, Debug)]
pub struct Job {
    /// URL of the prefix file to download.
    pub url: Arc<str>,
    /// The prefix being downloaded (e.g., "th", "to").
    pub prefix: Arc<str>,
    /// N-gram order for this job (1-5).
    pub order: u8,
    /// Current retry attempt (0 = first attempt).
    pub attempt: u8,
    /// Backoff duration in ms for next retry (doubles each attempt).
    pub backoff_ms: u64,
}

#[cfg(feature = "google-books")]
impl Job {
    /// Create a new job for first attempt.
    pub fn new(url: impl Into<Arc<str>>, prefix: impl Into<Arc<str>>, order: u8) -> Self {
        Self {
            url: url.into(),
            prefix: prefix.into(),
            order,
            attempt: 0,
            backoff_ms: INITIAL_BACKOFF_MS,
        }
    }

    /// Create a retry job with incremented attempt and doubled backoff.
    /// Arc<str> cloning is cheap (just a pointer increment).
    pub fn with_retry(&self) -> Self {
        Self {
            url: Arc::clone(&self.url),
            prefix: Arc::clone(&self.prefix),
            order: self.order,
            attempt: self.attempt + 1,
            backoff_ms: self.backoff_ms.saturating_mul(2),
        }
    }

    /// Check if this job can be retried (hasn't exceeded max retries).
    pub fn can_retry(&self) -> bool {
        self.attempt < MAX_RETRIES
    }
}

// ============================================================================
// Retry Queue
// ============================================================================

/// Entry in the retry priority queue.
#[cfg(feature = "google-books")]
struct RetryEntry {
    /// When this retry is ready to execute.
    ready_at: Instant,
    /// The job to retry.
    job: Job,
}

#[cfg(feature = "google-books")]
impl PartialEq for RetryEntry {
    fn eq(&self, other: &Self) -> bool {
        self.ready_at == other.ready_at
    }
}

#[cfg(feature = "google-books")]
impl Eq for RetryEntry {}

#[cfg(feature = "google-books")]
impl PartialOrd for RetryEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "google-books")]
impl Ord for RetryEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Natural ordering for Instant (earlier times are smaller)
        self.ready_at.cmp(&other.ready_at)
    }
}

/// Min-heap for retry tasks ordered by earliest ready_at time.
#[cfg(feature = "google-books")]
struct RetryQueue {
    /// Binary heap with Reverse wrapper for min-heap behavior.
    heap: BinaryHeap<Reverse<RetryEntry>>,
}

#[cfg(feature = "google-books")]
impl RetryQueue {
    /// Create a new empty retry queue.
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Push a job onto the retry queue with the specified ready time.
    fn push(&mut self, job: Job, ready_at: Instant) {
        self.heap.push(Reverse(RetryEntry { ready_at, job }));
    }

    /// Peek at the earliest ready time without removing.
    fn peek_ready_at(&self) -> Option<Instant> {
        self.heap.peek().map(|Reverse(entry)| entry.ready_at)
    }

    /// Pop the job with the earliest ready time if it's ready now.
    fn pop_if_ready(&mut self) -> Option<Job> {
        if let Some(Reverse(entry)) = self.heap.peek() {
            if entry.ready_at <= Instant::now() {
                return self.heap.pop().map(|Reverse(e)| e.job);
            }
        }
        None
    }

    /// Get the number of jobs in the retry queue.
    fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the retry queue is empty.
    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Time until the next retry is ready, or None if empty.
    fn time_until_next(&self) -> Option<Duration> {
        self.peek_ready_at()
            .and_then(|ready_at| ready_at.checked_duration_since(Instant::now()))
    }
}

// ============================================================================
// Task Manager Configuration
// ============================================================================

/// Configuration for retry behavior.
#[cfg(feature = "google-books")]
#[derive(Debug, Clone)]
pub struct TaskManagerConfig {
    /// Maximum number of retry attempts (default: 5).
    pub max_retries: u8,
    /// Initial backoff delay in milliseconds (default: 1000).
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds (default: 60000 = 1 minute).
    pub max_backoff_ms: u64,
    /// Jitter factor as a fraction (default: 0.25 = 25% jitter).
    /// Applied as +/- this percentage of the backoff value.
    pub jitter_factor: f64,
}

#[cfg(feature = "google-books")]
impl Default for TaskManagerConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60_000,
            jitter_factor: 0.25,
        }
    }
}

#[cfg(feature = "google-books")]
impl TaskManagerConfig {
    /// Create a new configuration with custom values.
    pub fn new(
        max_retries: u8,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        jitter_factor: f64,
    ) -> Self {
        Self {
            max_retries,
            initial_backoff_ms,
            max_backoff_ms,
            jitter_factor: jitter_factor.clamp(0.0, 1.0),
        }
    }

    /// Compute backoff duration with exponential increase and jitter.
    ///
    /// Formula: `min(initial * 2^attempt, max) * (1 +/- jitter)`
    pub fn compute_backoff(&self, attempt: u8) -> Duration {
        // Exponential backoff: initial * 2^attempt, capped at 10 doublings
        let base = self
            .initial_backoff_ms
            .saturating_mul(1u64 << attempt.min(10));
        let capped = base.min(self.max_backoff_ms);

        // Add jitter: +/- jitter_factor percent
        let jitter_range = (capped as f64) * self.jitter_factor;
        let jitter = jitter_range * (rand::random::<f64>() * 2.0 - 1.0);
        let with_jitter = (capped as f64 + jitter).max(0.0) as u64;

        Duration::from_millis(with_jitter)
    }
}

// ============================================================================
// Task Manager Metrics
// ============================================================================

/// Metrics for observability of the task manager.
#[cfg(feature = "google-books")]
#[derive(Debug)]
pub struct TaskManagerMetrics {
    /// Current depth of the regular (FIFO) queue.
    pub regular_queue_depth: AtomicUsize,
    /// Current depth of the retry (min-heap) queue.
    pub retry_queue_depth: AtomicUsize,
    /// Total number of retries scheduled across all jobs.
    pub total_retries: AtomicU64,
    /// Total number of jobs that failed permanently (max retries exceeded).
    pub total_failures: AtomicU64,
    /// Total number of jobs completed successfully.
    pub total_successes: AtomicU64,
}

#[cfg(feature = "google-books")]
impl Default for TaskManagerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "google-books")]
impl TaskManagerMetrics {
    /// Create new zeroed metrics.
    pub fn new() -> Self {
        Self {
            regular_queue_depth: AtomicUsize::new(0),
            retry_queue_depth: AtomicUsize::new(0),
            total_retries: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_successes: AtomicU64::new(0),
        }
    }

    /// Get a snapshot of current metrics values.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            regular_queue_depth: self.regular_queue_depth.load(Ordering::Relaxed),
            retry_queue_depth: self.retry_queue_depth.load(Ordering::Relaxed),
            total_retries: self.total_retries.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            total_successes: self.total_successes.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of metrics values.
#[cfg(feature = "google-books")]
#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    /// Current depth of the regular queue.
    pub regular_queue_depth: usize,
    /// Current depth of the retry queue.
    pub retry_queue_depth: usize,
    /// Total retries scheduled.
    pub total_retries: u64,
    /// Total permanent failures.
    pub total_failures: u64,
    /// Total successes.
    pub total_successes: u64,
}

// ============================================================================
// Task Manager
// ============================================================================

/// Dual-queue task manager for non-blocking retry handling.
///
/// The task manager maintains two queues:
/// 1. **Regular queue**: FIFO queue for new tasks, backed by `mpsc::Receiver`
/// 2. **Retry queue**: Min-heap priority queue ordered by earliest retry time
///
/// When workers call `get_next_task()`:
/// 1. First check if any retry is ready (head of min-heap <= now)
/// 2. If no ready retry, poll the regular queue
/// 3. If regular queue is empty, wait for either a new task or retry to become ready
#[cfg(feature = "google-books")]
pub struct TaskManager {
    /// Receiver for the regular (FIFO) queue.
    regular_rx: Mutex<mpsc::Receiver<Job>>,
    /// Priority queue for retry tasks.
    retry_queue: Mutex<RetryQueue>,
    /// Shutdown signal receiver.
    shutdown: watch::Receiver<bool>,
    /// Shared metrics for observability.
    metrics: Arc<TaskManagerMetrics>,
    /// Configuration for retry behavior.
    config: TaskManagerConfig,
}

#[cfg(feature = "google-books")]
impl TaskManager {
    /// Create a new task manager and submitter handle.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Buffer size for the regular queue
    /// * `config` - Retry behavior configuration
    /// * `shutdown` - Watch receiver for shutdown signal
    ///
    /// # Returns
    ///
    /// A tuple of `(TaskManager, TaskSubmitter)` where the submitter can be cloned
    /// and used to submit tasks from multiple sources.
    pub fn new(
        capacity: usize,
        config: TaskManagerConfig,
        shutdown: watch::Receiver<bool>,
    ) -> (Self, TaskSubmitter) {
        let (regular_tx, regular_rx) = mpsc::channel(capacity);
        let metrics = Arc::new(TaskManagerMetrics::new());

        let manager = Self {
            regular_rx: Mutex::new(regular_rx),
            retry_queue: Mutex::new(RetryQueue::new()),
            shutdown,
            metrics: Arc::clone(&metrics),
            config,
        };

        let submitter = TaskSubmitter {
            regular_tx,
            metrics,
        };

        (manager, submitter)
    }

    /// Get the next ready task.
    ///
    /// Priority:
    /// 1. Ready retry (from min-heap if head.ready_at <= now)
    /// 2. New task from regular queue
    /// 3. Wait for either a new task or retry to become ready
    ///
    /// Returns `None` when:
    /// - Shutdown signal received
    /// - Both queues are empty and closed
    pub async fn get_next_task(&self) -> Option<Job> {
        loop {
            // Check shutdown signal
            if *self.shutdown.borrow() {
                return None;
            }

            // First, check retry queue for ready retries
            {
                let mut retry_queue = self.retry_queue.lock().await;
                if let Some(job) = retry_queue.pop_if_ready() {
                    self.metrics
                        .retry_queue_depth
                        .store(retry_queue.len(), Ordering::Relaxed);
                    return Some(job);
                }
            }

            // Get time until next retry (if any)
            let time_until_retry = {
                let retry_queue = self.retry_queue.lock().await;
                retry_queue.time_until_next()
            };

            // Try to get from regular queue, with timeout if retries are pending
            let mut regular_rx = self.regular_rx.lock().await;

            // Clone shutdown receiver outside select! to avoid temporary borrow issues
            let mut shutdown_rx = self.shutdown.clone();

            match time_until_retry {
                Some(timeout) => {
                    // Wait for regular queue OR timeout for retry
                    tokio::select! {
                        biased;
                        _ = shutdown_rx.changed() => {
                            if *self.shutdown.borrow() {
                                return None;
                            }
                        }
                        result = tokio::time::timeout(timeout, regular_rx.recv()) => {
                            match result {
                                Ok(Some(job)) => {
                                    self.metrics.regular_queue_depth.fetch_sub(1, Ordering::Relaxed);
                                    return Some(job);
                                }
                                Ok(None) => {
                                    // Regular queue closed - check retry queue
                                    drop(regular_rx);
                                    let retry_queue = self.retry_queue.lock().await;
                                    if retry_queue.is_empty() {
                                        return None;
                                    }
                                    // Continue loop to wait for retry
                                }
                                Err(_) => {
                                    // Timeout - retry should be ready now, loop back
                                }
                            }
                        }
                    }
                }
                None => {
                    // No pending retries - just wait for regular queue
                    tokio::select! {
                        biased;
                        _ = shutdown_rx.changed() => {
                            if *self.shutdown.borrow() {
                                return None;
                            }
                        }
                        result = regular_rx.recv() => {
                            match result {
                                Some(job) => {
                                    self.metrics.regular_queue_depth.fetch_sub(1, Ordering::Relaxed);
                                    return Some(job);
                                }
                                None => {
                                    // Regular queue closed and no retries pending
                                    return None;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Schedule a job for retry with the specified backoff.
    ///
    /// The retry time is computed as:
    /// 1. If `retry_after` is provided, use that duration
    /// 2. Otherwise, use exponential backoff based on attempt count
    ///
    /// # Arguments
    ///
    /// * `job` - The job to retry (should be created with `job.with_retry()`)
    /// * `retry_after` - Optional Retry-After header value from server
    pub async fn schedule_retry(&self, job: Job, retry_after: Option<RetryAfter>) {
        let backoff = match retry_after {
            Some(ra) => ra.to_duration(),
            None => self.config.compute_backoff(job.attempt),
        };

        let ready_at = Instant::now() + backoff;

        let mut retry_queue = self.retry_queue.lock().await;
        retry_queue.push(job, ready_at);
        self.metrics
            .retry_queue_depth
            .store(retry_queue.len(), Ordering::Relaxed);
        self.metrics.total_retries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a permanent failure (max retries exceeded).
    pub fn record_failure(&self) {
        self.metrics.total_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful completion.
    pub fn record_success(&self) {
        self.metrics.total_successes.fetch_add(1, Ordering::Relaxed);
    }

    /// Get time until the next retry is ready.
    ///
    /// Returns `None` if the retry queue is empty.
    /// Returns `Duration::ZERO` if a retry is already ready.
    pub async fn time_until_next_retry(&self) -> Option<Duration> {
        let retry_queue = self.retry_queue.lock().await;
        retry_queue.time_until_next().or_else(|| {
            // If there's a retry but time_until_next is None, it's ready now
            if !retry_queue.is_empty() {
                Some(Duration::ZERO)
            } else {
                None
            }
        })
    }

    /// Get reference to metrics for observability.
    pub fn metrics(&self) -> &Arc<TaskManagerMetrics> {
        &self.metrics
    }

    /// Get reference to configuration.
    pub fn config(&self) -> &TaskManagerConfig {
        &self.config
    }

    /// Check if both queues are empty.
    pub async fn is_empty(&self) -> bool {
        let retry_empty = self.retry_queue.lock().await.is_empty();
        let regular_depth = self.metrics.regular_queue_depth.load(Ordering::Relaxed);
        retry_empty && regular_depth == 0
    }
}

// ============================================================================
// Task Submitter
// ============================================================================

/// Cloneable handle for submitting tasks to the TaskManager.
///
/// This handle can be cloned and shared across multiple task sources.
/// It only provides the ability to submit new tasks, not to retrieve them.
#[cfg(feature = "google-books")]
#[derive(Clone)]
pub struct TaskSubmitter {
    /// Sender for the regular queue.
    regular_tx: mpsc::Sender<Job>,
    /// Shared metrics reference.
    metrics: Arc<TaskManagerMetrics>,
}

#[cfg(feature = "google-books")]
impl TaskSubmitter {
    /// Submit a new task to the regular queue.
    ///
    /// Returns `Ok(())` if the task was queued, `Err(job)` if the queue is full or closed.
    pub async fn submit(&self, job: Job) -> Result<(), Job> {
        match self.regular_tx.send(job).await {
            Ok(()) => {
                self.metrics
                    .regular_queue_depth
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => Err(e.0),
        }
    }

    /// Try to submit a task without waiting.
    ///
    /// Returns `Ok(())` if the task was queued, `Err(job)` if the queue is full or closed.
    pub fn try_submit(&self, job: Job) -> Result<(), Job> {
        match self.regular_tx.try_send(job) {
            Ok(()) => {
                self.metrics
                    .regular_queue_depth
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(job)) => Err(job),
            Err(mpsc::error::TrySendError::Closed(job)) => Err(job),
        }
    }

    /// Get reference to shared metrics.
    pub fn metrics(&self) -> &Arc<TaskManagerMetrics> {
        &self.metrics
    }

    /// Check if the queue is closed.
    pub fn is_closed(&self) -> bool {
        self.regular_tx.is_closed()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "google-books"))]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn test_retry_after_parse_seconds() {
        assert_eq!(RetryAfter::parse("120"), Some(RetryAfter::Seconds(120)));
        assert_eq!(RetryAfter::parse("0"), Some(RetryAfter::Seconds(0)));
        assert_eq!(RetryAfter::parse("3600"), Some(RetryAfter::Seconds(3600)));
        assert_eq!(RetryAfter::parse("  60  "), Some(RetryAfter::Seconds(60)));
    }

    #[test]
    fn test_retry_after_parse_http_date() {
        // RFC 2822 format (HTTP-date)
        let result = RetryAfter::parse("Wed, 21 Oct 2015 07:28:00 GMT");
        assert!(matches!(result, Some(RetryAfter::DateTime(_))));

        if let Some(RetryAfter::DateTime(dt)) = result {
            assert_eq!(dt.year(), 2015);
            assert_eq!(dt.month(), 10);
            assert_eq!(dt.day(), 21);
        }
    }

    #[test]
    fn test_retry_after_parse_rfc3339() {
        // RFC 3339 / ISO 8601 format
        let result = RetryAfter::parse("2024-01-15T10:30:00Z");
        assert!(matches!(result, Some(RetryAfter::DateTime(_))));
    }

    #[test]
    fn test_retry_after_parse_invalid() {
        assert_eq!(RetryAfter::parse("invalid"), None);
        assert_eq!(RetryAfter::parse(""), None);
        assert_eq!(RetryAfter::parse("-100"), None);
    }

    #[test]
    fn test_retry_after_to_duration_seconds() {
        let ra = RetryAfter::Seconds(60);
        let duration = ra.to_duration();
        assert_eq!(duration, Duration::from_secs(60));
    }

    #[test]
    fn test_retry_after_to_duration_past_datetime() {
        // Past datetime should return ZERO
        let past = Utc::now() - chrono::Duration::hours(1);
        let ra = RetryAfter::DateTime(past);
        assert_eq!(ra.to_duration(), Duration::ZERO);
    }

    #[test]
    fn test_config_default() {
        let config = TaskManagerConfig::default();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_backoff_ms, 1000);
        assert_eq!(config.max_backoff_ms, 60_000);
        assert!((config.jitter_factor - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_compute_backoff_exponential() {
        let config = TaskManagerConfig {
            max_retries: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60_000,
            jitter_factor: 0.0, // No jitter for deterministic test
        };

        // Attempt 0: 1000ms
        let d0 = config.compute_backoff(0);
        assert_eq!(d0, Duration::from_millis(1000));

        // Attempt 1: 2000ms
        let d1 = config.compute_backoff(1);
        assert_eq!(d1, Duration::from_millis(2000));

        // Attempt 2: 4000ms
        let d2 = config.compute_backoff(2);
        assert_eq!(d2, Duration::from_millis(4000));

        // Attempt 5: 32000ms
        let d5 = config.compute_backoff(5);
        assert_eq!(d5, Duration::from_millis(32000));

        // Attempt 6: 64000ms -> capped at 60000ms
        let d6 = config.compute_backoff(6);
        assert_eq!(d6, Duration::from_millis(60000));
    }

    #[test]
    fn test_config_compute_backoff_with_jitter() {
        let config = TaskManagerConfig {
            max_retries: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60_000,
            jitter_factor: 0.25,
        };

        // Run multiple times to verify jitter range
        for _ in 0..100 {
            let d = config.compute_backoff(0);
            let ms = d.as_millis() as u64;
            // With 25% jitter, 1000ms should be in range [750, 1250]
            assert!(
                (750..=1250).contains(&ms),
                "Backoff {} outside expected range [750, 1250]",
                ms
            );
        }
    }

    #[test]
    fn test_job_new() {
        let job = Job::new("https://example.com/file.gz", "th", 2);
        assert_eq!(&*job.url, "https://example.com/file.gz");
        assert_eq!(&*job.prefix, "th");
        assert_eq!(job.order, 2);
        assert_eq!(job.attempt, 0);
        assert_eq!(job.backoff_ms, INITIAL_BACKOFF_MS);
    }

    #[test]
    fn test_job_with_retry() {
        let job = Job::new("https://example.com/file.gz", "th", 2);
        let retry1 = job.with_retry();

        assert_eq!(&*retry1.url, "https://example.com/file.gz");
        assert_eq!(retry1.attempt, 1);
        assert_eq!(retry1.backoff_ms, 2000);

        let retry2 = retry1.with_retry();
        assert_eq!(retry2.attempt, 2);
        assert_eq!(retry2.backoff_ms, 4000);
    }

    #[test]
    fn test_job_can_retry() {
        let mut job = Job::new("https://example.com/file.gz", "th", 2);
        assert!(job.can_retry());

        for _ in 0..MAX_RETRIES {
            job = job.with_retry();
        }
        assert!(!job.can_retry());
    }

    #[test]
    fn test_retry_queue_ordering() {
        let mut queue = RetryQueue::new();

        let now = Instant::now();
        let job1 = Job::new("url1", "a", 1);
        let job2 = Job::new("url2", "b", 1);
        let job3 = Job::new("url3", "c", 1);

        // Push in reverse order
        queue.push(job3.clone(), now + Duration::from_secs(30));
        queue.push(job1.clone(), now + Duration::from_secs(10));
        queue.push(job2.clone(), now + Duration::from_secs(20));

        assert_eq!(queue.len(), 3);

        // Earliest should be first
        assert_eq!(queue.peek_ready_at(), Some(now + Duration::from_secs(10)));
    }

    #[test]
    fn test_retry_queue_pop_if_ready() {
        let mut queue = RetryQueue::new();
        let now = Instant::now();

        let job1 = Job::new("url1", "a", 1);
        let job2 = Job::new("url2", "b", 1);

        // Push one ready and one not ready
        queue.push(job1.clone(), now - Duration::from_secs(1)); // Ready (past)
        queue.push(job2.clone(), now + Duration::from_secs(60)); // Not ready

        // Should pop the ready one
        let popped = queue.pop_if_ready();
        assert!(popped.is_some());
        assert_eq!(&*popped.unwrap().prefix, "a");

        // Should not pop the non-ready one
        let not_ready = queue.pop_if_ready();
        assert!(not_ready.is_none());

        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = TaskManagerMetrics::new();

        metrics.regular_queue_depth.store(10, Ordering::Relaxed);
        metrics.retry_queue_depth.store(5, Ordering::Relaxed);
        metrics.total_retries.store(100, Ordering::Relaxed);
        metrics.total_failures.store(3, Ordering::Relaxed);
        metrics.total_successes.store(97, Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.regular_queue_depth, 10);
        assert_eq!(snapshot.retry_queue_depth, 5);
        assert_eq!(snapshot.total_retries, 100);
        assert_eq!(snapshot.total_failures, 3);
        assert_eq!(snapshot.total_successes, 97);
    }

    #[tokio::test]
    async fn test_task_manager_basic_flow() {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let config = TaskManagerConfig::default();
        let (manager, submitter) = TaskManager::new(100, config, shutdown_rx);

        // Submit some jobs
        let job1 = Job::new("url1", "a", 1);
        let job2 = Job::new("url2", "b", 1);

        submitter.submit(job1).await.expect("submit job1");
        submitter.submit(job2).await.expect("submit job2");

        // Get jobs (should be FIFO)
        let got1 = manager.get_next_task().await;
        assert!(got1.is_some());
        assert_eq!(&*got1.unwrap().prefix, "a");

        let got2 = manager.get_next_task().await;
        assert!(got2.is_some());
        assert_eq!(&*got2.unwrap().prefix, "b");

        // Shutdown
        shutdown_tx.send(true).expect("send shutdown");

        // Should return None after shutdown
        let got3 = manager.get_next_task().await;
        assert!(got3.is_none());
    }

    #[tokio::test]
    async fn test_task_manager_retry_priority() {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let config = TaskManagerConfig::default();
        let (manager, submitter) = TaskManager::new(100, config, shutdown_rx);

        // Submit a regular job
        let regular_job = Job::new("regular", "regular", 1);
        submitter.submit(regular_job).await.expect("submit regular");

        // Schedule a retry that's ready immediately
        let retry_job = Job::new("retry", "retry", 1).with_retry();
        manager
            .schedule_retry(retry_job, Some(RetryAfter::Seconds(0)))
            .await;

        // Retry should come first (priority)
        let got1 = manager.get_next_task().await;
        assert!(got1.is_some());
        assert_eq!(&*got1.unwrap().prefix, "retry");

        // Then regular job
        let got2 = manager.get_next_task().await;
        assert!(got2.is_some());
        assert_eq!(&*got2.unwrap().prefix, "regular");

        shutdown_tx.send(true).expect("send shutdown");
    }

    #[tokio::test]
    async fn test_task_manager_metrics_tracking() {
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let config = TaskManagerConfig::default();
        let (manager, submitter) = TaskManager::new(100, config, shutdown_rx);

        // Submit jobs
        submitter
            .submit(Job::new("url1", "a", 1))
            .await
            .expect("submit");
        submitter
            .submit(Job::new("url2", "b", 1))
            .await
            .expect("submit");

        // Check regular queue depth
        let snapshot = manager.metrics().snapshot();
        assert_eq!(snapshot.regular_queue_depth, 2);

        // Get one job
        let _ = manager.get_next_task().await;
        let snapshot = manager.metrics().snapshot();
        assert_eq!(snapshot.regular_queue_depth, 1);

        // Schedule a retry
        let retry_job = Job::new("retry", "r", 1).with_retry();
        manager.schedule_retry(retry_job, None).await;

        let snapshot = manager.metrics().snapshot();
        assert_eq!(snapshot.retry_queue_depth, 1);
        assert_eq!(snapshot.total_retries, 1);

        // Record success and failure
        manager.record_success();
        manager.record_failure();

        let snapshot = manager.metrics().snapshot();
        assert_eq!(snapshot.total_successes, 1);
        assert_eq!(snapshot.total_failures, 1);
    }
}
