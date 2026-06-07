#![cfg(feature = "loom-tests")]

use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

const CLEAN: usize = 0;
const DIRTY: usize = 1;
const SYNCING: usize = 2;
const CLEAN_AFTER_SYNC: usize = 3;

// RETIRED (lock-free overlay migration): the production write-token mechanism
// (try_acquire_write / release_write / generation counter) was removed from
// src/sources/google_books/sharding/shard.rs in favor of lock-free overlay
// writes (increment_cas). With no exclusive per-shard token there is no
// single-writer-exclusion property left to align against, so this loom model --
// and its companion spec formal/tla/ShardWriteToken.tla -- are retired (kept,
// not deleted, per the no-deletion policy). The lock-free replacement's safety
// is covered by `async_shard_sync_has_at_most_one_syncer` below and by
// formal/tla/AsyncShardSync.tla (single-syncer CAS, no writer token).
//
// #[test]
// fn write_token_excludes_double_writer() {
//     loom::model(|| {
//         let locked = Arc::new(AtomicBool::new(false));
//         let generation = Arc::new(AtomicUsize::new(0));
//         let active_writers = Arc::new(AtomicUsize::new(0));
//
//         let mut handles = Vec::new();
//         for _ in 0..2 {
//             let locked = Arc::clone(&locked);
//             let generation = Arc::clone(&generation);
//             let active_writers = Arc::clone(&active_writers);
//
//             handles.push(thread::spawn(move || {
//                 if locked
//                     .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
//                     .is_ok()
//                 {
//                     let token_generation = generation.load(Ordering::Relaxed) + 1;
//                     generation.store(token_generation, Ordering::Relaxed);
//
//                     let previous = active_writers.fetch_add(1, Ordering::SeqCst);
//                     assert_eq!(previous, 0, "two writers held the shard token");
//                     thread::yield_now();
//                     active_writers.fetch_sub(1, Ordering::SeqCst);
//
//                     assert_eq!(generation.load(Ordering::Relaxed), token_generation);
//                     locked.store(false, Ordering::Release);
//                 }
//             }));
//         }
//
//         for handle in handles {
//             handle.join().expect("writer thread panicked");
//         }
//
//         assert_eq!(active_writers.load(Ordering::SeqCst), 0);
//     });
// }

#[test]
fn async_shard_sync_has_at_most_one_syncer() {
    loom::model(|| {
        let state = Arc::new(AtomicUsize::new(CLEAN));
        let active_syncers = Arc::new(AtomicUsize::new(0));

        state.store(DIRTY, Ordering::Release);

        let mut handles = Vec::new();
        for _ in 0..2 {
            let state = Arc::clone(&state);
            let active_syncers = Arc::clone(&active_syncers);

            handles.push(thread::spawn(move || {
                if state
                    .compare_exchange(DIRTY, SYNCING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    let previous = active_syncers.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(previous, 0, "two syncers entered Syncing");
                    thread::yield_now();
                    active_syncers.fetch_sub(1, Ordering::SeqCst);
                    state.store(CLEAN_AFTER_SYNC, Ordering::Release);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("sync thread panicked");
        }

        assert_eq!(active_syncers.load(Ordering::SeqCst), 0);
        // Exactly one thread wins the DIRTY -> SYNCING CAS and stores
        // CLEAN_AFTER_SYNC; the loser no-ops. CLEAN_AFTER_SYNC is the only
        // reachable terminal (DIRTY would require no thread winning, which the
        // initial `state.store(DIRTY)` rules out).
        assert_eq!(state.load(Ordering::Acquire), CLEAN_AFTER_SYNC);
    });
}

#[test]
fn worker_shutdown_sends_claimed_result_before_exit() {
    loom::model(|| {
        let shutdown = Arc::new(AtomicBool::new(false));
        let job_available = Arc::new(AtomicBool::new(true));
        let result_sent = Arc::new(AtomicBool::new(false));
        let exited = Arc::new(AtomicBool::new(false));

        let worker = {
            let shutdown = Arc::clone(&shutdown);
            let job_available = Arc::clone(&job_available);
            let result_sent = Arc::clone(&result_sent);
            let exited = Arc::clone(&exited);

            thread::spawn(move || {
                if !shutdown.load(Ordering::Acquire)
                    && job_available
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    thread::yield_now();
                    result_sent.store(true, Ordering::Release);
                }

                exited.store(true, Ordering::Release);
            })
        };

        shutdown.store(true, Ordering::Release);
        worker.join().expect("worker thread panicked");

        if !job_available.load(Ordering::Acquire) {
            assert!(result_sent.load(Ordering::Acquire));
        }
        assert!(exited.load(Ordering::Acquire));
    });
}

#[test]
fn cron_ready_signal_precedes_post_ready_schedule() {
    loom::model(|| {
        let ready = Arc::new(AtomicBool::new(false));
        let scheduled = Arc::new(AtomicBool::new(false));
        let executed = Arc::new(AtomicBool::new(false));

        let cron = {
            let ready = Arc::clone(&ready);
            let scheduled = Arc::clone(&scheduled);
            let executed = Arc::clone(&executed);

            thread::spawn(move || {
                ready.store(true, Ordering::Release);
                while !scheduled.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                executed.store(true, Ordering::Release);
            })
        };

        while !ready.load(Ordering::Acquire) {
            thread::yield_now();
        }
        scheduled.store(true, Ordering::Release);

        cron.join().expect("cron thread panicked");
        assert!(executed.load(Ordering::Acquire));
    });
}
