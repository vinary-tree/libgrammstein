# Cron Manager

Lock-free reactive state machine scheduler for periodic tasks.

**Source**: [`src/util/cron/mod.rs`](../../../src/util/cron/mod.rs)

---

## Overview

The cron manager implements a task scheduler using a reactive state machine design with non-blocking algorithms and atomics. It is designed for periodic checkpointing during long-running imports but can be used for any scheduled task workload.

### Key Design Goals

1. **Explicit states with clear transitions** - States and events are enumerated types
2. **Event-driven architecture** - No polling loops with scattered conditionals
3. **Lock-free task submission** - MPSC channel for concurrent task submission
4. **Thread-local min-heap** - Priority queue owned by the scheduler thread
5. **Graceful shutdown** - Termination signal checked at every state transition

### When to Use

- Periodic checkpointing during long-running operations
- Scheduled maintenance tasks
- Any workload requiring timer-based task execution
- Scenarios requiring lock-free concurrent task submission

---

## Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Cron Manager Architecture                        │
│                                                                              │
│   ┌────────────────┐    crossbeam-channel      ┌─────────────────────────┐  │
│   │  Main Thread   │    (lock-free MPSC)       │     Cron Thread         │  │
│   │                │ ─────────────────────────>│                         │  │
│   │  CronHandle    │                           │  CronStateMachine       │  │
│   │  (task submit) │    Arc<AtomicBool>        │  (owns task heap)       │  │
│   │                │<──────────────────────────│                         │  │
│   └────────────────┘    termination flag       └─────────────────────────┘  │
│          │                                                │                  │
│          │                                                │                  │
│          ▼                                                ▼                  │
│   ┌────────────────┐                           ┌─────────────────────────┐  │
│   │  Arc<CronStats>│◀──────────────────────────│  AtomicU64 counters     │  │
│   │  (read stats)  │    atomic loads           │  (tasks, failures, etc) │  │
│   └────────────────┘                           └─────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### State Machine Diagram

```
                              ┌─────────────────────────────────────────┐
                              │                                         │
                              ▼                                         │
    ┌─────────┐  channel has  ┌──────────────┐  task due    ┌──────────────────┐
    │  Idle   │ ────────────▶ │ DrainChannel │ ───────────▶ │ ExecutingTask    │
    └─────────┘   messages    └──────────────┘              └──────────────────┘
         │                          │                              │
         │                          │ channel empty                │ task complete
         │                          │ & no tasks due               │ (requeue if
         │                          ▼                              │  recurring)
         │                    ┌──────────────┐                     │
         │                    │   Sleeping   │◀────────────────────┘
         │                    └──────────────┘
         │                          │
         │                          │ timer expired
         │                          ▼
         │                    ┌──────────────┐
         └───────────────────▶│ CheckEvents  │◀──── (loop back)
                              └──────────────┘
                                    │
                                    │ terminating == true
                                    ▼
                              ┌──────────────┐
                              │  Terminated  │
                              └──────────────┘
```

### State Descriptions

| State | Description |
|-------|-------------|
| `CheckEvents` | Initial state - evaluate event sources (channel, queue, termination) |
| `DrainChannel` | Draining incoming tasks from the channel into the local queue |
| `ExecutingTask` | Currently executing a due task |
| `Sleeping` | Sleeping until next task is due or poll interval expires |
| `Terminated` | Terminal state - graceful shutdown complete |

---

## Core Types

### CronState

Enumerated state machine states:

```rust
pub enum CronState {
    CheckEvents,    // Evaluate event sources
    DrainChannel,   // Drain incoming tasks
    ExecutingTask,  // Execute a due task
    Sleeping,       // Wait for timer/task
    Terminated,     // Shutdown complete
}
```

### CronEvent

Events that drive state transitions:

```rust
pub enum CronEvent {
    TaskReceived,                              // New task(s) in channel
    TimerExpired,                              // Sleep timer expired
    TaskDue,                                   // Task past deadline
    TaskCompleted { success: bool, should_requeue: bool },
    TerminationRequested,                      // External shutdown signal
    ChannelDisconnected,                       // All senders dropped
    NoEvents,                                  // Idle
}
```

### TaskMetadata

Metadata describing task behavior:

```rust
pub enum TaskMetadata {
    /// One-shot task - executed once and discarded
    OneShot,

    /// Recurring task - re-queued after completion
    Recurring { interval_ms: u64 },

    /// Named task with optional recurrence
    Named {
        name: String,
        recurring_interval_ms: Option<u64>,
    },
}
```

### ScheduledTask

A task with execution time and callback:

```rust
pub struct ScheduledTask {
    pub scheduled_time_ms: UnixTimestampMs,  // When to execute
    pub metadata: TaskMetadata,               // Task type
    pub task: Box<dyn FnMut() -> bool + Send>, // Callback
}
```

The task callback returns `true` to indicate success. For recurring tasks, returning `false` prevents rescheduling.

### CronStats

Lock-free statistics counters:

```rust
pub struct CronStats {
    pub tasks_executed: AtomicU64,   // Total tasks executed
    pub tasks_failed: AtomicU64,     // Tasks that returned false
    pub tasks_panicked: AtomicU64,   // Tasks that panicked
    pub transitions: AtomicU64,      // State transitions performed
}
```

### CronStatsSnapshot

Immutable snapshot of statistics (useful for logging):

```rust
pub struct CronStatsSnapshot {
    pub tasks_executed: u64,
    pub tasks_failed: u64,
    pub tasks_panicked: u64,
    pub transitions: u64,
}
```

### CronStateMachine

The state machine itself (runs on dedicated thread):

```rust
pub struct CronStateMachine {
    state: CronState,
    queue: BinaryHeap<ScheduledTask>,       // Thread-local min-heap
    task_rx: Receiver<ScheduledTask>,       // Lock-free MPSC receiver
    poll_interval_ms: u64,
    terminating: Arc<AtomicBool>,
    channel_disconnected: bool,
    stats: Arc<CronStats>,
}
```

### CronHandle

Lock-free handle for submitting tasks (cloneable, thread-safe):

```rust
#[derive(Clone)]
pub struct CronHandle {
    task_tx: Sender<ScheduledTask>,     // Lock-free MPSC sender
    terminating: Arc<AtomicBool>,
}
```

---

## State Transition Table

Complete mapping of (State, Event) → Next State:

| Current State | Event | Next State | Notes |
|---------------|-------|------------|-------|
| Any | `TerminationRequested` | `Terminated` | Highest priority |
| `CheckEvents` | `TaskReceived` | `DrainChannel` | Start draining channel |
| `CheckEvents` | `TaskDue` | `CheckEvents` | Execute task, re-check |
| `CheckEvents` | `NoEvents` | `Sleeping` | Nothing to do |
| `CheckEvents` | `ChannelDisconnected` | `Terminated` or `CheckEvents` | Terminate if queue empty |
| `DrainChannel` | `TaskReceived` | `DrainChannel` | Continue draining |
| `DrainChannel` | `TaskDue` | `CheckEvents` | Execute task, re-check |
| `DrainChannel` | `NoEvents` | `Sleeping` | Done draining |
| `DrainChannel` | `ChannelDisconnected` | `CheckEvents` | Continue with existing tasks |
| `Sleeping` | `TimerExpired` | `CheckEvents` | Wake and check |
| `ExecutingTask` | `TaskCompleted` | `CheckEvents` | Task done, re-check |

---

## Lock-Free Guarantees

Every component uses lock-free synchronization:

| Component | Synchronization Primitive | Lock-Free? | Notes |
|-----------|--------------------------|------------|-------|
| Task submission | `crossbeam-channel` (MPSC) | ✅ Wait-free send | Unbounded channel |
| Termination flag | `AtomicBool` | ✅ Yes | Acquire/Release ordering |
| Statistics | `AtomicU64` | ✅ Yes | Relaxed ordering (counters) |
| State machine | Thread-local | ✅ N/A | No sharing needed |
| Task queue | `BinaryHeap` (thread-local) | ✅ N/A | Owned by cron thread |

### Memory Ordering

- **Termination flag**: `Acquire` on load, `Release` on store - ensures proper synchronization
- **Statistics**: `Relaxed` ordering - counters don't require ordering guarantees
- **Channel**: Lock-free internally via crossbeam's wait-free algorithms

---

## API Reference

### spawn_cron

Spawn the cron state machine with default 100ms poll interval.

```rust
pub fn spawn_cron(
    terminating: Arc<AtomicBool>,
) -> (CronHandle, JoinHandle<()>, Arc<CronStats>, Receiver<()>)
```

**Parameters:**
- `terminating` - Shared termination flag

**Returns:**
- `CronHandle` - For submitting tasks (clone-able, thread-safe)
- `JoinHandle<()>` - For joining the cron thread
- `Arc<CronStats>` - For reading statistics (lock-free)
- `Receiver<()>` - One-shot channel that signals when the scheduler is ready (see below)

### spawn_cron_with_interval

Spawn with custom poll interval.

```rust
pub fn spawn_cron_with_interval(
    terminating: Arc<AtomicBool>,
    poll_interval_ms: u64,
) -> (CronHandle, JoinHandle<()>, Arc<CronStats>, Receiver<()>)
```

**Parameters:**
- `terminating` - Shared termination flag
- `poll_interval_ms` - Maximum sleep duration between event checks

**Returns:**
- `CronHandle` - For submitting tasks (clone-able, thread-safe)
- `JoinHandle<()>` - For joining the cron thread
- `Arc<CronStats>` - For reading statistics (lock-free)
- `Receiver<()>` - One-shot channel that signals when the scheduler is ready (see below)

#### Ready Signal

The returned `Receiver<()>` provides a one-shot signal that the scheduler is ready.
Call `ready_rx.recv()` to block until the cron thread has entered its event loop.
This prevents race conditions where tasks are scheduled before the scheduler is fully initialized.

```rust
let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, stats, ready_rx) = spawn_cron(Arc::clone(&terminating));

// Wait for scheduler to be ready before scheduling tasks
ready_rx.recv().expect("Cron thread failed to start");

// Now safe to schedule tasks
handle.schedule_once(0, "my-task", || true);
```

For most use cases, you can safely ignore the ready signal if you don't need deterministic startup:

```rust
let (handle, thread, stats, _ready) = spawn_cron(Arc::clone(&terminating));
```

### CronHandle::schedule_at

Schedule a task at a specific Unix timestamp.

```rust
pub fn schedule_at<F>(
    &self,
    time_ms: UnixTimestampMs,
    metadata: TaskMetadata,
    task: F,
) -> bool
where
    F: FnMut() -> bool + Send + 'static
```

**Returns:** `true` if submitted, `false` if channel disconnected.

### CronHandle::schedule_after

Schedule a task after a delay.

```rust
pub fn schedule_after<F>(
    &self,
    delay_ms: u64,
    metadata: TaskMetadata,
    task: F,
) -> bool
where
    F: FnMut() -> bool + Send + 'static
```

### CronHandle::schedule_recurring

Schedule a recurring task with initial delay and interval.

```rust
pub fn schedule_recurring<F>(
    &self,
    initial_delay_ms: u64,
    interval_ms: u64,
    name: &str,
    task: F,
) -> bool
where
    F: FnMut() -> bool + Send + 'static
```

The task continues recurring as long as it returns `true`. Returning `false` stops rescheduling.

### CronHandle::schedule_once

Schedule a one-shot task after a delay.

```rust
pub fn schedule_once<F>(
    &self,
    delay_ms: u64,
    name: &str,
    task: F,
) -> bool
where
    F: FnMut() -> bool + Send + 'static
```

### CronHandle::request_shutdown

Request graceful shutdown of the state machine.

```rust
pub fn request_shutdown(&self)
```

This is a lock-free atomic store. The scheduler will terminate at the next event check.

### CronHandle::is_shutting_down

Check if shutdown has been requested.

```rust
pub fn is_shutting_down(&self) -> bool
```

### CronStats::snapshot

Get an immutable snapshot of current statistics.

```rust
pub fn snapshot(&self) -> CronStatsSnapshot
```

---

## Usage Examples

### Basic One-Shot Task

```rust
use libgrammstein::util::cron::{spawn_cron, TaskMetadata};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, stats, _ready) = spawn_cron(Arc::clone(&terminating));

// Schedule a task to run after 1 second
handle.schedule_once(1000, "delayed-task", || {
    println!("Task executed after 1 second");
    true  // Success
});

// ... do other work ...

// Graceful shutdown
handle.request_shutdown();
thread.join().expect("Cron thread panicked");

println!("Executed {} tasks", stats.tasks_executed.load(std::sync::atomic::Ordering::Relaxed));
```

### Recurring Task

```rust
use libgrammstein::util::cron::spawn_cron;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, _stats, _ready) = spawn_cron(Arc::clone(&terminating));

let counter = Arc::new(AtomicU64::new(0));
let counter_clone = Arc::clone(&counter);

// Schedule task to run every 5 seconds, starting immediately
handle.schedule_recurring(0, 5000, "periodic-counter", move || {
    let count = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;
    println!("Counter incremented to {}", count);
    true  // Continue recurring
});

// Run for 20 seconds
std::thread::sleep(std::time::Duration::from_secs(20));

handle.request_shutdown();
thread.join().expect("Cron thread panicked");

println!("Final count: {}", counter.load(Ordering::Relaxed));
// Expected: ~4-5 executions (0s, 5s, 10s, 15s, potentially 20s)
```

### Self-Terminating Recurring Task

```rust
use libgrammstein::util::cron::spawn_cron;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, _stats, _ready) = spawn_cron(Arc::clone(&terminating));

let counter = Arc::new(AtomicU64::new(0));
let counter_clone = Arc::clone(&counter);

// Schedule task that stops after 5 executions
handle.schedule_recurring(0, 100, "limited-task", move || {
    let count = counter_clone.fetch_add(1, Ordering::Relaxed) + 1;
    println!("Execution #{}", count);
    count < 5  // Return false on 5th execution to stop
});

std::thread::sleep(std::time::Duration::from_secs(1));
handle.request_shutdown();
thread.join().expect("Cron thread panicked");

assert_eq!(counter.load(Ordering::Relaxed), 5);
```

### Concurrent Task Submission

```rust
use libgrammstein::util::cron::spawn_cron;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, stats, _ready) = spawn_cron(Arc::clone(&terminating));

let counter = Arc::new(AtomicU64::new(0));

// Submit tasks from 10 threads concurrently
let handles: Vec<_> = (0..10)
    .map(|_| {
        let h = handle.clone();  // Clone is cheap (Arc + Sender)
        let c = Arc::clone(&counter);
        std::thread::spawn(move || {
            for _ in 0..100 {
                let c = Arc::clone(&c);
                h.schedule_once(0, "concurrent-task", move || {
                    c.fetch_add(1, Ordering::Relaxed);
                    true
                });
            }
        })
    })
    .collect();

// Wait for all submitters to finish
for h in handles {
    h.join().expect("Submitter thread panicked");
}

// Wait for tasks to execute
std::thread::sleep(std::time::Duration::from_millis(500));

handle.request_shutdown();
thread.join().expect("Cron thread panicked");

// All 1000 tasks should have executed
assert_eq!(counter.load(Ordering::Relaxed), 1000);
assert_eq!(stats.tasks_executed.load(Ordering::Relaxed), 1000);
```

### Integration: Google Books Importer

Real-world usage for periodic checkpointing during a long-running import:

```rust
use libgrammstein::util::cron::{spawn_cron_with_interval, TaskMetadata};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Shared state for periodic checkpoint tasks (lock-free reads).
pub struct CheckpointState {
    storage: Arc<Storage>,
    ngrams_processed: Arc<AtomicU64>,
    last_checkpoint_ngrams: AtomicU64,
}

impl CheckpointState {
    /// Perform a checkpoint (called from cron thread).
    pub fn perform_checkpoint(&self) -> Result<(), Error> {
        let current = self.ngrams_processed.load(Ordering::Acquire);
        let last = self.last_checkpoint_ngrams.swap(current, Ordering::AcqRel);

        if current > last {
            log::info!("Checkpointing: {} new n-grams since last checkpoint", current - last);
            self.storage.flush_to_disk()?;
        }

        Ok(())
    }
}

pub async fn run_import_with_periodic_checkpoints(
    storage: Arc<Storage>,
    checkpoint_interval_ms: u64,
    terminating: Arc<AtomicBool>,
) -> Result<ImportStats, Error> {
    let ngrams_processed = Arc::new(AtomicU64::new(0));

    let checkpoint_state = Arc::new(CheckpointState {
        storage: Arc::clone(&storage),
        ngrams_processed: Arc::clone(&ngrams_processed),
        last_checkpoint_ngrams: AtomicU64::new(0),
    });

    // Start cron with 50ms poll interval for responsive shutdown
    let (cron_handle, cron_thread, cron_stats, _ready) =
        spawn_cron_with_interval(Arc::clone(&terminating), 50);

    // Schedule periodic checkpoints
    let checkpoint_state_for_cron = Arc::clone(&checkpoint_state);
    cron_handle.schedule_recurring(
        checkpoint_interval_ms,  // Initial delay
        checkpoint_interval_ms,  // Interval
        "periodic-checkpoint",
        move || {
            match checkpoint_state_for_cron.perform_checkpoint() {
                Ok(()) => true,  // Continue checkpointing
                Err(e) => {
                    log::error!("Checkpoint failed: {}", e);
                    true  // Keep trying
                }
            }
        },
    );

    // Run the import (async)
    let result = run_import(&storage, &ngrams_processed, &terminating).await;

    // Final checkpoint before shutdown
    if let Err(e) = checkpoint_state.perform_checkpoint() {
        log::error!("Final checkpoint failed: {}", e);
    }

    // Signal termination to cron scheduler
    terminating.store(true, Ordering::Release);

    // Wait for cron manager to stop
    log::info!("Stopping periodic checkpoint scheduler...");
    if let Err(e) = cron_thread.join() {
        log::error!("Cron thread panicked: {:?}", e);
    }

    let stats_snapshot = cron_stats.snapshot();
    log::info!(
        "Cron stats: {} checkpoints, {} failures",
        stats_snapshot.tasks_executed,
        stats_snapshot.tasks_failed
    );

    result
}
```

---

## Error Handling

### Task Returns False

When a task returns `false`:
- Logged as a failure (warning level)
- Not rescheduled, even if recurring
- `tasks_failed` counter incremented

```rust
handle.schedule_recurring(0, 1000, "conditional-task", || {
    if some_condition() {
        true  // Continue recurring
    } else {
        log::warn!("Stopping task due to condition");
        false  // Stop recurring
    }
});
```

### Task Panics

When a task panics:
- Panic is caught with `std::panic::catch_unwind`
- Logged as an error
- `tasks_panicked` counter incremented
- Scheduler continues running other tasks

```rust
// This won't crash the scheduler
handle.schedule_once(0, "panicking-task", || {
    panic!("This panic is caught");
});

// This task will still execute
handle.schedule_once(100, "normal-task", || {
    println!("Still running!");
    true
});
```

### Channel Disconnection

When all `CronHandle` instances are dropped:
- If queue is empty: scheduler terminates immediately
- If queue has pending tasks: scheduler continues until queue is drained, then terminates

```rust
let terminating = Arc::new(AtomicBool::new(false));
let (handle, thread, _stats, _ready) = spawn_cron(Arc::clone(&terminating));

// Schedule a delayed task
handle.schedule_once(100, "delayed", || {
    println!("Executed even after handle dropped");
    true
});

// Drop handle immediately
drop(handle);

// Scheduler continues running until delayed task completes
// Then terminates because channel is disconnected and queue is empty
thread.join().expect("Cron thread panicked");
```

---

## Testing

The module includes 14 unit tests covering all aspects of the scheduler:

| Test | Description |
|------|-------------|
| `test_state_transitions` | Verifies initial state is `CheckEvents` |
| `test_termination_from_any_state` | Termination works from any state |
| `test_concurrent_task_submission` | 10 threads × 100 tasks = 1000 executions |
| `test_recurring_task` | Task recurs at specified interval |
| `test_recurring_task_stops_on_false` | Returning false stops recurrence |
| `test_one_shot_task` | One-shot executes exactly once |
| `test_panic_safety` | Panicking task doesn't crash scheduler |
| `test_channel_disconnect_empty_queue` | Terminates when queue empty |
| `test_channel_disconnect_with_tasks` | Continues with pending tasks |
| `test_stats_snapshot` | Statistics tracking is accurate |
| `test_task_metadata` | Metadata types work correctly |
| `test_task_ordering` | Min-heap orders by scheduled time |
| `test_handle_cloning` | Cloned handles submit correctly |
| `test_shutdown_flag` | Shutdown flag propagates correctly |

Run tests with:

```bash
cargo test cron -- --nocapture
```

---

## Performance Considerations

### Poll Interval Selection

| Use Case | Recommended Interval | Notes |
|----------|---------------------|-------|
| Responsive UI | 10-50ms | Low latency shutdown |
| Background tasks | 100-500ms | Default, good balance |
| Battery-sensitive | 1000ms+ | Minimal wake-ups |

### Adaptive Sleep

The scheduler sleeps for `min(poll_interval, time_to_next_task)`, so tasks execute promptly regardless of poll interval setting.

### Memory Usage

- Each `ScheduledTask` contains a boxed closure
- Queue size grows with pending tasks
- Statistics use 4 × 8 = 32 bytes (atomic u64s)

---

## Formal Verification

The cron state machine is formally specified and verified using TLA+ (Temporal Logic of Actions).

**Specification**: [`formal/tla/CronStateMachine.tla`](../../../formal/tla/CronStateMachine.tla)

### What TLA+ Verifies

The specification models the concurrent interaction between:
1. **CronThread** - The state machine (CheckEvents/DrainChannel/Sleeping/ExecutingTask/Terminated)
2. **TestThread** - Client behavior (spawn, wait_ready, schedule_tasks, request_shutdown)

### Safety Invariants

| Property | Description |
|----------|-------------|
| `TypeOK` | All variables have correct types |
| `ReadySignalSafety` | Ready signal received only after sent |
| `PanicIsolation` | Panicked and executed task sets are disjoint |
| `TestCompletionRequiresExecution` | Test completes only after normal task executes |
| `TerminationRequiresRequest` | Cron terminates only when termination is requested |

### Liveness Properties (under fairness)

| Property | Description |
|----------|-------------|
| `TestEventuallyCompletes` | Test eventually reaches done state |
| `CronEventuallyTerminates` | Cron terminates after shutdown request |
| `NormalTaskExecutesAfterPanic` | Normal task executes even after a panicking task |
| `ReadySignalEventuallyReceived` | Ready signal is eventually received |

### Running the Model Checker

```bash
# Navigate to formal specifications
cd formal/tla

# Run TLC model checker (requires TLA+ toolbox or command-line tools)
tlc CronStateMachine.cfg

# For liveness checking (slower), uncomment SPECIFICATION and PROPERTY lines in .cfg
```

### Key Verified Behaviors

1. **Panic Safety**: A panicking task does not prevent subsequent tasks from executing
2. **Ready Signal Correctness**: Tasks scheduled after receiving the ready signal are guaranteed to be processed
3. **Graceful Termination**: The scheduler always terminates when shutdown is requested
4. **No Deadlock**: The system cannot reach a deadlocked state

---

## See Also

- [Threading Model](../../architecture/threading.md) - Concurrency patterns in libgrammstein
- [Data Flow](../../architecture/data-flow.md) - How data moves through the system
- [Google Books Importer](../../components/corpus/streaming.md) - Main user of periodic checkpoints
