# Formal Verification

This directory contains the executable formal models and machine-checked
proofs for correctness-sensitive parts of the Google Books importer.

- TLA+ models concurrency, lifecycle, crash-recovery, and shutdown behavior.
- Rocq proves arithmetic and algebraic properties used by MKN statistics and
  frequency-count aggregation.
- The Rust implementation is kept aligned with the verified models where the
  models expose implementation requirements such as no silent `u64` wrap.

## Directory Structure

```text
formal/
├── README.md
├── Makefile
├── tla/
│   ├── ShardWriteToken.tla
│   ├── ShardWriteTokenProofs.tla
│   ├── MC_ShardWriteToken.cfg
│   ├── MC_ShardWriteToken_Liveness.cfg
│   ├── MC_ShardWriteToken_Stress.cfg
│   ├── CheckpointStateMachine.tla
│   ├── CheckpointStateMachineProofs.tla
│   ├── MC_CheckpointStateMachine.cfg
│   ├── MC_CheckpointStateMachine_Liveness.cfg
│   ├── MC_CheckpointStateMachine_Stress.cfg
│   ├── AsyncShardSync.tla
│   ├── AsyncShardSync.cfg
│   ├── AsyncShardSync_Liveness.cfg
│   ├── AsyncShardSync_Stress.cfg
│   ├── CronStateMachine.tla
│   ├── CronStateMachine.cfg
│   ├── CronStateMachine_Liveness.cfg
│   ├── CronStateMachine_Stress.cfg
│   ├── WorkerShutdown.tla
│   ├── MC_WorkerShutdown.cfg
│   ├── MC_WorkerShutdown_NoSymmetry.cfg
│   └── MC_WorkerShutdown_Stress.cfg
└── rocq/
    ├── _CoqProject
    ├── Makefile
    ├── MknStatistics.v
    ├── FrequencyCountsMerge.v
    └── MknFloatBounds.v
```

## Tools

Installed tools used by this repository:

- `tlc` for bounded TLA+ model checking.
- `tlapm` for TLAPS proof modules.
- `apalache-mc` for TLA+ typechecking.
- `rocq makefile` / `rocq compile` through the local Rocq makefile.
- `cargo test` for source-level alignment checks.

If `tlapm` is installed outside `PATH`, pass it explicitly:

```bash
make complete TLAPM=/home/dylon/.local/tlaps/bin/tlapm
```

Apalache currently needs `--features=no-rows` for these specs because some TLC
friendly record-set idioms are outside Apalache's default row-typing mode.

## Commands

Run the installed verification stack from the top-level formal entrypoint:

```bash
cd formal
make check
```

Run all installed checks plus the TLAPS gate:

```bash
cd formal
make complete
```

`make complete` requires TLAPS in addition to the installed bounded model
checking, typechecking, Rocq, and Rust alignment tools.

When TLAPS is installed outside `PATH`, run:

```bash
make -C formal complete TLAPM=/home/dylon/.local/tlaps/bin/tlapm
```

Run larger bounded TLC stress configs with a timeout budget:

```bash
cd formal
make stress STRESS_TIMEOUT=180s
```

Run TLA+ safety checks:

```bash
cd formal/tla
tlc -metadir /tmp/tlc-shard -config MC_ShardWriteToken.cfg ShardWriteToken.tla
tlc -metadir /tmp/tlc-checkpoint -config MC_CheckpointStateMachine.cfg CheckpointStateMachine.tla
tlc -metadir /tmp/tlc-async -config AsyncShardSync.cfg AsyncShardSync.tla
tlc -metadir /tmp/tlc-cron -config CronStateMachine.cfg CronStateMachine.tla
tlc -metadir /tmp/tlc-worker -config MC_WorkerShutdown_NoSymmetry.cfg WorkerShutdown.tla
```

Run TLA+ liveness checks:

```bash
cd formal/tla
tlc -metadir /tmp/tlc-shard-live -config MC_ShardWriteToken_Liveness.cfg ShardWriteToken.tla
tlc -metadir /tmp/tlc-checkpoint-live -config MC_CheckpointStateMachine_Liveness.cfg CheckpointStateMachine.tla
tlc -metadir /tmp/tlc-async-live -config AsyncShardSync_Liveness.cfg AsyncShardSync.tla
tlc -metadir /tmp/tlc-cron-live -config CronStateMachine_Liveness.cfg CronStateMachine.tla
```

Run Apalache typechecks:

```bash
cd formal/tla
apalache-mc --features=no-rows --out-dir=/tmp/apalache-shard typecheck ShardWriteToken.tla
apalache-mc --features=no-rows --out-dir=/tmp/apalache-checkpoint typecheck CheckpointStateMachine.tla
apalache-mc --features=no-rows --out-dir=/tmp/apalache-async typecheck AsyncShardSync.tla
apalache-mc --features=no-rows --out-dir=/tmp/apalache-cron typecheck CronStateMachine.tla
apalache-mc --features=no-rows --out-dir=/tmp/apalache-worker typecheck WorkerShutdown.tla
```

Run Rocq proof checks:

```bash
cd formal/rocq
make check
```

Expected Rocq completion message:

```text
All proofs complete (no proof holes or trusted declarations found)
```

Run focused Rust alignment checks:

```bash
cargo test --features google-books sources::google_books::sharding::shard::tests::test_write_token_generation_does_not_wrap
cargo test --features google-books sources::google_books::sharding::mkn
```

## TLA+ Models

### ShardWriteToken.tla

Target: `src/sources/google_books/sharding/shard.rs`

Models exclusive shard write ownership, token acquisition/release, generation
matching, and generation exhaustion.

Safety properties:

- `AtMostOneWriter`
- `ValidTokenGenerationMatch`
- `ValidTokenImpliesHolder`
- `GenerationMonotonic`

Liveness properties:

- `NoStarvation`
- `HeldLockEventuallyReleases`

The model exposed a source requirement: write generations must not silently
wrap. Rust now uses checked generation increment and fails acquisition at
`u64::MAX`.

`ShardWriteTokenProofs.tla` proves the decomposed type invariant, exclusive
writer safety, holder/generation consistency, initialization, each action's
preservation obligations, stutter preservation, and `Spec => []Safety` with
TLAPS. Large action proofs are split into helper theorems for function updates
and per-action preservation lemmas, so TLAPS can recheck small obligations
instead of one monolithic invariant theorem.

### CheckpointStateMachine.tla

Target: `src/sources/google_books/checkpoint.rs`

Models prefix lifecycle transitions, explicit reprocessing, crash, restart, and
recovery from in-progress work.

Safety properties:

- `DisjointSets`
- `StateConsistent`
- `CompletedNotFailed`
- `CrashRecoverySound`

Liveness properties:

- completed or failed prefixes remain terminal until explicit reprocessing
- restart/recovery actions make progress under fairness

The model intentionally allows explicit `StartPrefix` on completed or failed
prefixes because the Rust checkpoint API overwrites the prefix state when a
caller explicitly reprocesses a prefix.

The model assumes `NONE \notin Workers`, matching the Rust representation where
`None` is a sentinel distinct from real worker identifiers.

`CheckpointStateMachineProofs.tla` proves initialization, per-action safety
preservation, crash-recovery consistency, stutter preservation, and
`Spec => []Safety` with TLAPS. The largest `StartPrefix` preservation proof is
decomposed into type, disjointness, state-consistency, worker-assignment,
completion, and crash-recovery lemmas, with additional cases for the updated
prefix, same-order unaffected prefixes, and unaffected orders.

### AsyncShardSync.tla

Target: `src/sources/google_books/sharding/shard.rs`,
`src/sources/google_books/sharding/coordinator/sync.rs`, and
`src/sources/google_books/sharding/worker_pool.rs`

Models dirty shard tracking, checkpoint target capture, shard sync, failed sync
abort, deferred jobs, and global checkpoint save.

Safety properties:

- `CheckpointAtomicity`
- `JobPartition`
- `CleanMeansZeroDirty`
- `WorkersSafelyDefer`

Liveness properties require scheduler and retry fairness:

- queued jobs eventually complete
- shard sync completion is strongly fair
- deferred jobs eventually return
- failed checkpoints eventually abort
- checkpoint start and per-shard sync are weakly fair

The Rust implementation uses transactions/write locks that make a worker
already past the pre-check safe even if a checkpoint starts before commit. The
model conservatively represents that as a processing job that must finish before
the target shard is synced.

`WorkersSafelyDefer` is checked with a sticky history flag
`unsafe_write_attempted`, so weakening the worker write precondition later will
turn into an invariant failure instead of a vacuous pass.

### CronStateMachine.tla

Target: `src/util/cron/mod.rs`

Models task scheduling, due-task execution, sleeping, termination requests,
handle drop, channel close, and panic accounting.

Safety properties:

- `TasksNotLost`
- `PanicCountSound`
- `TerminationRequiresRequest`

Liveness properties:

- termination request eventually terminates
- closed empty channel eventually terminates

The liveness model requires the closed-empty channel path to terminate rather
than sleep forever. This matches the Rust `recv_timeout` disconnect handling.

### WorkerShutdown.tla

Target: `src/sources/google_books/sharding/importer/worker_pool.rs` and related
state-machine shutdown code.

Models job ownership, stop requests, worker exit, and shutdown completion.

Safety properties:

- `NoJobDuplicated`
- `NoJobLost`
- `JobUniqueOwnership`
- `NoWorkAfterStopped`
- `ShutdownImpliesAllStopped`

Liveness is checked with `MC_WorkerShutdown_NoSymmetry.cfg` because symmetry can
hide some temporal-property issues in TLC.

## Rocq Proofs

### MknStatistics.v

Target: `src/sources/google_books/sharding/mkn.rs`

Proves the boundedness of Modified Kneser-Ney discount calculations using
rational arithmetic:

- `y_bounded`
- `d1_bounded`
- `d2_clamped_bounded`
- `d3_plus_clamped_bounded`
- `from_counts_safe_valid`
- `from_counts_rust_valid`

The proof model now matches Rust's implementation detail that both `n3` and
`n4` are clamped to at least 1 for `D3+`, and that the public Rust function
returns default discounts when `n1 == 0` or `n2 == 0`.

### MknFloatBounds.v

Target: `src/sources/google_books/sharding/mkn.rs`

Models the Rust `f64` MKN calculation envelope with real arithmetic and Flocq's
binary64 exponent threshold:

- raw MKN expressions remain below a conservative binary64 overflow margin for
  `u64`-derived positive counts
- clamped discount models stay within `[0,1]`, `[0,2]`, and `[0,3]`
- the `Y` parameter stays within `[0,1]`

This does not prove every IEEE-754 rounding bit. It proves the finite magnitude
envelope and clamped output ranges that the Rust implementation relies on.

### FrequencyCountsMerge.v

Target: `src/sources/google_books/sharding/mkn.rs`

Proves algebraic and bounded-arithmetic properties for frequency-count
aggregation:

- `merge_associative`
- `merge_commutative`
- `merge_identity_left`
- `merge_identity_right`
- `merge_is_commutative_monoid`
- `merge_preserves_u64_bounds`
- `observe_preserves_validity`
- `observe_preserves_u64_bounds`

The bounded proofs require checked addition. Rust now panics instead of silently
wrapping if frequency counts overflow `u64`.

## Source Alignment

The verification pass led to two implementation changes:

- `Shard::try_acquire_write` uses checked generation increment and rejects
  acquisition after `u64::MAX` instead of wrapping to zero.
- `FrequencyCounts::merge` and `AtomicFrequencyCounts::observe` use checked
  addition so Rocq's bounded-count model matches Rust behavior.

The checkpoint, async shard sync, cron, and worker shutdown source paths were
reviewed against the final TLA+ abstractions. No further source changes were
needed there.

## Verification Status

| Area | Tool | Status |
| --- | --- | --- |
| Shard write-token safety | TLC | Pass |
| Shard write-token liveness | TLC | Pass |
| Shard write-token typecheck | Apalache | Pass |
| Checkpoint safety | TLC | Pass |
| Checkpoint liveness | TLC | Pass |
| Checkpoint typecheck | Apalache | Pass |
| Async shard sync safety | TLC | Pass |
| Async shard sync liveness | TLC | Pass |
| Async shard sync typecheck | Apalache | Pass |
| Cron safety | TLC | Pass |
| Cron liveness | TLC | Pass |
| Cron typecheck | Apalache | Pass |
| Worker shutdown safety/liveness | TLC | Pass |
| Worker shutdown typecheck | Apalache | Pass |
| MKN arithmetic bounds | Rocq | Pass |
| MKN floating-point envelope | Rocq/Flocq/Interval | Pass |
| Frequency-count algebra and bounds | Rocq | Pass |
| Focused Rust alignment tests | Cargo | Pass |
| Shard write-token inductive safety | TLAPS | Pass |
| Checkpoint inductive safety | TLAPS | Pass |

## Known Limits

- TLC checks bounded finite models. The selected bounds cover contention,
  recovery, retry, shutdown, and failure interleavings, but they are not
  unbounded mathematical proofs.
- TLA+ models atomic operations as linearizable. Rust memory-order details are
  abstracted to their intended linearization points.
- Rocq now proves both exact rational MKN bounds and a binary64 magnitude
  envelope. It still does not prove bit-exact IEEE-754 rounding for each Rust
  operation.
- Async liveness assumes fair scheduling and eventual retry opportunities for
  sync and deferred-job actions.
- TLAPS proves unbounded inductive safety for the write-token and checkpoint
  models. Async shard sync, cron, and worker shutdown still rely on TLC
  bounded safety/liveness plus Apalache typechecking.
