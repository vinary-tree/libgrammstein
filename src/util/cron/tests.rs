//! Unit tests for the cron state machine scheduler.

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Test that state transitions work correctly.
#[test]
fn test_state_transitions() {
    let (_, rx) = unbounded::<ScheduledTask>();
    let terminating = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(CronStats::default());

    let sm = CronStateMachine::new(rx, terminating.clone(), stats.clone(), 100, None);

    // Initial state is CheckEvents
    assert_eq!(sm.current_state(), CronState::CheckEvents);
}

/// Test that termination signal works from any state.
#[test]
fn test_termination_from_any_state() {
    let (_, rx) = unbounded::<ScheduledTask>();
    let terminating = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(CronStats::default());

    let mut sm = CronStateMachine::new(rx, terminating.clone(), stats.clone(), 100, None);

    // Request termination
    terminating.store(true, AtomicOrdering::Release);

    // Should immediately transition to Terminated
    sm.run();
    assert_eq!(sm.current_state(), CronState::Terminated);
}

/// Test concurrent task submission from multiple threads.
#[test]
fn test_concurrent_task_submission() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, stats, _ready) = spawn_cron(Arc::clone(&terminating));

    let counter = Arc::new(AtomicU64::new(0));

    // Submit tasks from multiple threads concurrently (lock-free)
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let h = handle.clone();
            let c = Arc::clone(&counter);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    h.schedule_after(0, TaskMetadata::OneShot, {
                        let c = Arc::clone(&c);
                        move || {
                            c.fetch_add(1, Ordering::Relaxed);
                            true
                        }
                    });
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    // Wait for tasks to execute
    std::thread::sleep(Duration::from_millis(500));

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    // All 1000 tasks should have executed
    assert_eq!(counter.load(Ordering::Relaxed), 1000);
    assert_eq!(stats.tasks_executed.load(Ordering::Relaxed), 1000);

    // State machine should have performed many transitions
    assert!(stats.transitions.load(Ordering::Relaxed) > 0);
}

/// Test that recurring tasks are requeued.
#[test]
fn test_recurring_task() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    // Schedule recurring task every 50ms
    handle.schedule_recurring(0, 50, "counter", move || {
        c.fetch_add(1, Ordering::Relaxed);
        true
    });

    // Wait ~275ms - should execute ~5-6 times
    std::thread::sleep(Duration::from_millis(275));

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    let count = counter.load(Ordering::Relaxed);
    assert!(
        count >= 4 && count <= 7,
        "Expected 4-7 executions, got {}",
        count
    );
}

/// Test that recurring tasks stop when returning false.
#[test]
fn test_recurring_task_stops_on_false() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    // Schedule recurring task that stops after 3 executions
    handle.schedule_recurring(0, 20, "limited-counter", move || {
        let count = c.fetch_add(1, Ordering::Relaxed) + 1;
        count < 3 // Return false on 3rd execution
    });

    // Wait longer than needed for all executions
    std::thread::sleep(Duration::from_millis(200));

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    let count = counter.load(Ordering::Relaxed);
    assert_eq!(count, 3, "Expected exactly 3 executions, got {}", count);
}

/// Test that one-shot tasks execute exactly once.
#[test]
fn test_one_shot_task() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    // Schedule one-shot task
    handle.schedule_once(0, "one-shot", move || {
        c.fetch_add(1, Ordering::Relaxed);
        true
    });

    // Wait for execution
    std::thread::sleep(Duration::from_millis(100));

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    let count = counter.load(Ordering::Relaxed);
    assert_eq!(count, 1, "Expected exactly 1 execution, got {}", count);
}

/// Test that panicking tasks are caught and don't crash the scheduler.
#[test]
fn test_panic_safety() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, stats, ready_rx) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    // Wait for scheduler to be ready (prevents race condition where tasks are
    // scheduled before the cron thread has entered its event loop)
    ready_rx.recv().expect("Cron thread failed to start");

    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    // Schedule a task that panics
    handle.schedule_once(0, "panicking", || {
        panic!("This task intentionally panics");
    });

    // Schedule a normal task after the panicking one
    handle.schedule_once(50, "normal", move || {
        c.fetch_add(1, Ordering::Relaxed);
        true
    });

    // Poll until normal task executes (with timeout)
    // This is more robust than a fixed sleep as it adapts to system load
    let deadline = std::time::Instant::now() + Duration::from_millis(500);
    while counter.load(Ordering::Relaxed) == 0 {
        if std::time::Instant::now() > deadline {
            panic!("Timeout waiting for normal task to execute");
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    handle.request_shutdown();
    thread
        .join()
        .expect("Cron thread should not panic from task panic");

    // Normal task should have executed despite the panic
    let count = counter.load(Ordering::Relaxed);
    assert_eq!(count, 1, "Normal task should have executed");

    // Stats should show one panic
    assert_eq!(stats.tasks_panicked.load(Ordering::Relaxed), 1);
}

/// Test that channel disconnection terminates the scheduler when queue is empty.
#[test]
fn test_channel_disconnect_empty_queue() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    // Don't schedule any tasks, just drop the handle
    drop(handle);

    // Scheduler should terminate since queue is empty and channel is disconnected
    thread.join().expect("Cron thread panicked");

    // Should have at least one transition (to check events, then to terminated)
    assert!(stats.transitions.load(Ordering::Relaxed) >= 1);
}

/// Test that channel disconnection keeps scheduler running when tasks remain.
#[test]
fn test_channel_disconnect_with_tasks() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    let counter = Arc::new(AtomicU64::new(0));
    let c = Arc::clone(&counter);

    // Schedule a delayed task
    handle.schedule_once(100, "delayed", move || {
        c.fetch_add(1, Ordering::Relaxed);
        true
    });

    // Drop handle immediately (disconnects channel)
    drop(handle);

    // Scheduler should terminate after the delayed task completes and the
    // already-disconnected channel is observed with an empty queue.
    let deadline = std::time::Instant::now() + Duration::from_millis(500);
    while !thread.is_finished() {
        if std::time::Instant::now() > deadline {
            panic!("Timeout waiting for cron thread to terminate after draining tasks");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    thread.join().expect("Cron thread panicked");

    let count = counter.load(Ordering::Relaxed);
    assert_eq!(count, 1, "Delayed task should have executed");
}

/// Test statistics snapshot.
#[test]
fn test_stats_snapshot() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, stats, _ready) = spawn_cron_with_interval(Arc::clone(&terminating), 10);

    // Schedule tasks
    for _ in 0..5 {
        handle.schedule_once(0, "success", || true);
    }
    for _ in 0..3 {
        handle.schedule_once(0, "failure", || false);
    }

    // Wait for execution
    std::thread::sleep(Duration::from_millis(100));

    // Get snapshot
    let snapshot = stats.snapshot();

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    // Verify snapshot
    assert_eq!(snapshot.tasks_executed, 8);
    assert_eq!(snapshot.tasks_failed, 3);
    assert_eq!(snapshot.tasks_panicked, 0);
}

/// Test task metadata types.
#[test]
fn test_task_metadata() {
    // OneShot
    let one_shot = TaskMetadata::OneShot;
    assert_eq!(one_shot.name(), "one-shot");
    assert_eq!(one_shot.recurrence_interval(), None);

    // Recurring
    let recurring = TaskMetadata::Recurring { interval_ms: 1000 };
    assert_eq!(recurring.name(), "recurring");
    assert_eq!(recurring.recurrence_interval(), Some(1000));

    // Named one-shot
    let named = TaskMetadata::Named {
        name: "custom".to_string(),
        recurring_interval_ms: None,
    };
    assert_eq!(named.name(), "custom");
    assert_eq!(named.recurrence_interval(), None);

    // Named recurring
    let named_recurring = TaskMetadata::Named {
        name: "checkpoint".to_string(),
        recurring_interval_ms: Some(5000),
    };
    assert_eq!(named_recurring.name(), "checkpoint");
    assert_eq!(named_recurring.recurrence_interval(), Some(5000));
}

/// Test scheduled task ordering (min-heap behavior).
#[test]
fn test_task_ordering() {
    let mut heap = BinaryHeap::new();

    // Add tasks in random order
    heap.push(ScheduledTask {
        scheduled_time_ms: 300,
        metadata: TaskMetadata::OneShot,
        task: Box::new(|| true),
    });
    heap.push(ScheduledTask {
        scheduled_time_ms: 100,
        metadata: TaskMetadata::OneShot,
        task: Box::new(|| true),
    });
    heap.push(ScheduledTask {
        scheduled_time_ms: 200,
        metadata: TaskMetadata::OneShot,
        task: Box::new(|| true),
    });

    // Should pop in ascending order (earliest first)
    assert_eq!(heap.pop().expect("task 1").scheduled_time_ms, 100);
    assert_eq!(heap.pop().expect("task 2").scheduled_time_ms, 200);
    assert_eq!(heap.pop().expect("task 3").scheduled_time_ms, 300);
}

/// Test that the scheduler execution path consumes the earliest queued task.
#[test]
fn test_execute_one_task_uses_earliest_due_task() {
    let (_, rx) = unbounded::<ScheduledTask>();
    let terminating = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(CronStats::default());
    let mut sm = CronStateMachine::new(rx, terminating, stats, 100, None);
    let executed = Arc::new(std::sync::Mutex::new(Vec::new()));

    for scheduled_time_ms in [300, 100, 200] {
        let executed = Arc::clone(&executed);
        sm.queue.push(ScheduledTask {
            scheduled_time_ms,
            metadata: TaskMetadata::OneShot,
            task: Box::new(move || {
                executed.lock().expect("lock").push(scheduled_time_ms);
                true
            }),
        });
    }

    sm.execute_one_task();
    sm.execute_one_task();
    sm.execute_one_task();

    let observed = executed.lock().expect("lock").clone();
    assert_eq!(observed, vec![100, 200, 300]);
}

/// Test handle cloning and concurrent use.
#[test]
fn test_handle_cloning() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron(Arc::clone(&terminating));

    let counter = Arc::new(AtomicU64::new(0));

    // Clone handle multiple times
    let handle1 = handle.clone();
    let handle2 = handle.clone();

    let c1 = Arc::clone(&counter);
    let c2 = Arc::clone(&counter);

    // Submit from different handles
    handle1.schedule_once(0, "from-handle1", move || {
        c1.fetch_add(1, Ordering::Relaxed);
        true
    });
    handle2.schedule_once(0, "from-handle2", move || {
        c2.fetch_add(1, Ordering::Relaxed);
        true
    });

    std::thread::sleep(Duration::from_millis(100));

    handle.request_shutdown();
    thread.join().expect("Cron thread panicked");

    assert_eq!(counter.load(Ordering::Relaxed), 2);
}

/// Test shutdown flag propagation.
#[test]
fn test_shutdown_flag() {
    let terminating = Arc::new(AtomicBool::new(false));
    let (handle, thread, _stats, _ready) = spawn_cron(Arc::clone(&terminating));

    // Initially not shutting down
    assert!(!handle.is_shutting_down());

    // Request shutdown
    handle.request_shutdown();

    // Should now be shutting down
    assert!(handle.is_shutting_down());

    thread.join().expect("Cron thread panicked");
}
