# Formal Verification

This directory contains the executable formal models and machine-checked
proofs for correctness-sensitive parts of the Google Books importer.

Completed, planned, and remaining verification work is tracked in `TASKS.md`.
The recent dependency bridges connect this repository's importer/checkpoint and
query-wrapper models to libdictenstein's persistent trie/WAL/checkpoint/recovery
guarantees and liblevenshtein's trusted dictionary, metric, and query-adjacent
guarantees.

- TLA+ models concurrency, lifecycle, crash-recovery, and shutdown behavior.
- Rocq proves arithmetic and algebraic properties used by MKN statistics and
  frequency-count aggregation.
- The Rust implementation is kept aligned with the verified models where the
  models expose implementation requirements such as no silent `u64` wrap.

## Directory Structure

```text
formal/
├── README.md
├── TASKS.md
├── dependencies/
│   ├── libdictenstein-contracts.md
│   └── liblevenshtein-contracts.md
├── Makefile
├── tla/
│   ├── ShardWriteToken.tla             # deprecated (retired with WriteToken)
│   ├── ShardWriteTokenProofs.tla       # deprecated
│   ├── MC_ShardWriteToken.cfg          # deprecated
│   ├── MC_ShardWriteToken_Liveness.cfg # deprecated
│   ├── MC_ShardWriteToken_Stress.cfg   # deprecated
│   ├── CheckpointStateMachine.tla
│   ├── CheckpointStateMachineProofs.tla
│   ├── MC_CheckpointStateMachine.cfg
│   ├── MC_CheckpointStateMachine_Liveness.cfg
│   ├── MC_CheckpointStateMachine_Stress.cfg
│   ├── AsyncShardSync.tla
│   ├── AsyncShardSyncProofs.tla
│   ├── AsyncShardSync.cfg
│   ├── AsyncShardSync_Liveness.cfg
│   ├── AsyncShardSync_Stress.cfg
│   ├── CronStateMachine.tla
│   ├── CronStateMachineProofs.tla
│   ├── CronStateMachine.cfg
│   ├── CronStateMachine_Liveness.cfg
│   ├── CronStateMachine_Stress.cfg
│   ├── WorkerShutdown.tla
│   ├── WorkerShutdownProofs.tla
│   ├── MC_WorkerShutdown.cfg
│   ├── MC_WorkerShutdown_Safety.cfg
│   ├── MC_WorkerShutdown_NoSymmetry.cfg
│   ├── MC_WorkerShutdown_Stress.cfg
│   ├── ImporterLifecycle.tla
│   ├── ImporterLifecycleProofs.tla
│   ├── ImporterLifecycle.cfg
│   ├── ImporterLifecycle_Liveness.cfg
│   ├── ImporterLifecycle_Stress.cfg
│   ├── PersistentStorageBridge.tla
│   ├── PersistentStorageBridgeProofs.tla
│   ├── PersistentStorageBridge.cfg
│   ├── PersistentStorageBridge_Sharded.cfg
│   ├── PersistentStorageBridge_Stress.cfg
│   ├── QuerySemanticsBridge.tla
│   ├── QuerySemanticsBridgeProofs.tla
│   ├── QuerySemanticsBridge.cfg
│   ├── QuerySemanticsBridge_NoMetadata.cfg
│   └── QuerySemanticsBridge_Stress.cfg
└── rocq/
    ├── _CoqProject
    ├── Makefile
    ├── MknStatistics.v
    ├── FrequencyCountsMerge.v
    ├── MknFloatBounds.v
    ├── VarintTermIdView.v
    └── NoisyChannelDecode.v
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

The Makefile runs TLC and Apalache with conservative local resource caps by
default:

- `TLC_JAVA_OPTS=-XX:+UseParallelGC -Xmx768m`
- `TLC_WORKERS=1`
- `TLA_MODEL_TIMEOUT=60s`
- `TLC_STRESS_WORKERS=1`
- `STRESS_TIMEOUT=60s`
- `DEPENDENCY_TLA_MODEL_TIMEOUT=120s`
- `APALACHE_JVM_ARGS=-Xmx1536m`
- `APALACHE_TIMEOUT=60s`
- `TLAPS_TIMEOUT=120s`
- `MIRI_FLAGS=-Zmiri-strict-provenance -Zmiri-disable-isolation`

Raise these explicitly only on a machine intended for larger model-checking
runs. Direct `tlc` or `apalache-mc` commands bypass the Makefile defaults, so
prefer the Makefile targets for routine local verification.

## Commands

Run the installed verification stack from the top-level formal entrypoint:

```bash
cd formal
make check
```

Run all installed checks plus stress and the TLAPS gate:

```bash
cd formal
make complete
```

`make complete` requires TLAPS in addition to the installed bounded model
checking, stress model checking, typechecking, Rocq, Rust alignment tools, and
the source-hygiene gates. The fast hygiene phase checks formatting, diff
whitespace, and source marker hygiene before expensive model checking. The full
hygiene phase checks all-target all-feature compilation with an automated
local-warning audit and then runs the full all-feature library suite against
this crate's manifest. Dependency-crate warnings remain visible in Cargo output,
and the audit prints a non-fatal dependency warning summary; local
`libgrammstein` warnings fail the gate. Some all-feature Rust tests start local
mock HTTP servers, so sandboxed runners must allow loopback port binding.

When TLAPS is installed outside `PATH`, run:

```bash
make -C formal complete TLAPM=/home/dylon/.local/tlaps/bin/tlapm
```

Run the explicit dependency-contract refresh gate:

```bash
make -C formal dependency-contracts
```

This invokes `../libdictenstein/scripts/verify-formal-correspondence.sh` and
`../liblevenshtein-rust/scripts/verify-formal.sh trusted`. It is kept separate
from the default local gate because it depends on sibling repositories and can
fail due dependency-local inventory drift even when libgrammstein's own models
and correspondence tests pass.

Check whether sibling path dependencies are warning-clean:

```bash
make -C formal dependency-warning-hygiene
```

This target fails until Cargo emits no warnings from `../libdictenstein`,
`../liblevenshtein-rust`, or `../PathMap`. It is intentionally separate from
the local `complete` gate because those repositories are outside this crate's
workspace boundary.

Run the stricter dependency gate, including imported libdictenstein TLC models,
io_uring storage-correspondence tests, and Miri unsafe-boundary checks:

```bash
make -C formal strict-dependency-contracts MIRI_TOOLCHAIN=nightly
```

This target uses the same bounded TLC defaults as the local formal gate and a
120 second timeout for each imported dependency TLA+ model.

Run larger bounded TLC stress configs with a timeout budget:

```bash
cd formal
make stress
```

Stress timeouts are hard failures. The stress configurations are larger than
the standard safety/liveness configurations while remaining complete finite
state-space checks.

Run TLA+ safety checks:

```bash
cd formal/tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-checkpoint -config MC_CheckpointStateMachine.cfg CheckpointStateMachine.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-async -config AsyncShardSync.cfg AsyncShardSync.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-cron -config CronStateMachine.cfg CronStateMachine.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-worker -config MC_WorkerShutdown_Safety.cfg WorkerShutdown.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-importer-lifecycle -config ImporterLifecycle.cfg ImporterLifecycle.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-persistent-storage-bridge -config PersistentStorageBridge.cfg PersistentStorageBridge.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-persistent-storage-bridge-sharded -config PersistentStorageBridge_Sharded.cfg PersistentStorageBridge.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-query-semantics-bridge -config QuerySemanticsBridge.cfg QuerySemanticsBridge.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-query-semantics-bridge-no-metadata -config QuerySemanticsBridge_NoMetadata.cfg QuerySemanticsBridge.tla
```

Run TLA+ liveness checks:

```bash
cd formal/tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-checkpoint-live -config MC_CheckpointStateMachine_Liveness.cfg CheckpointStateMachine.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-async-live -config AsyncShardSync_Liveness.cfg AsyncShardSync.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-cron-live -config CronStateMachine_Liveness.cfg CronStateMachine.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-worker-live -config MC_WorkerShutdown_NoSymmetry.cfg WorkerShutdown.tla
TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m" timeout 60s tlc -workers 1 -metadir /tmp/tlc-importer-lifecycle-live -config ImporterLifecycle_Liveness.cfg ImporterLifecycle.tla
```

Run Apalache typechecks:

```bash
cd formal/tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-checkpoint typecheck CheckpointStateMachine.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-async typecheck AsyncShardSync.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-cron typecheck CronStateMachine.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-worker typecheck WorkerShutdown.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-importer-lifecycle typecheck ImporterLifecycle.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-persistent-storage-bridge typecheck PersistentStorageBridge.tla
JVM_ARGS=-Xmx1536m apalache-mc --features=no-rows --out-dir=/tmp/apalache-query-semantics-bridge typecheck QuerySemanticsBridge.tla
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
cargo test --features google-books sources::google_books::importer::tests::shutdown_checkpoint_safety
cargo test --features google-books sources::google_books::state_machine::tests
cargo test util::cron::tests
cargo test --features loom-tests --test loom_formal_alignment
```

## TLA+ Models

### ShardWriteToken.tla

**DEPRECATED / RETIRED (lock-free overlay migration).** The WriteToken mechanism
this models was removed from `shard.rs` in favor of lock-free overlay writes
(`increment_cas`); the spec and `ShardWriteTokenProofs.tla` are retained for
historical reference but are no longer part of the formal gate. Its successor is
`AsyncShardSync.tla` (below), whose `AtMostOneSyncer` invariant subsumes the
single-writer guarantee. The original description follows.

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

Target: `src/sources/google_books/sharding/shard.rs` (the `ShardSyncCoordinator`
state machine) and `src/sources/google_books/sharding/coordinator/sync.rs`.

Models the lock-free overlay write path: dirty shard tracking, checkpoint target
capture, per-shard sync, failed-sync abort, and global checkpoint save. After the
overlay migration there is no defer-and-continue — workers issue `increment_cas`
writes that proceed concurrently with an in-flight sync. `WorkerProcessJob` fires
regardless of sync state, and its shard-state effect mirrors `mark_dirty` exactly:
Clean → Dirty, identity in Dirty/Syncing/SyncFailed.

Safety properties:

- `AtMostOneSyncer`
- `SyncerConsistency`
- `CheckpointAtomicity`
- `CleanMeansZeroDirty`
- `JobRepresented`, `JobSetsDisjoint`, `WorkerJobsDisjointFromSets`,
  `WorkerJobsUnique` (composed as `JobPartition`)
- `WorkerStateJobConsistency`
- `CheckpointReadyToSave`

Liveness properties require scheduler and retry fairness:

- queued jobs eventually complete
- the checkpoint eventually completes (or surfaces a `SyncFailed` shard)
- shard sync completion is strongly fair
- checkpoint start and per-shard sync are weakly fair

Because `mark_dirty` only transitions Clean → Dirty, a write that lands while a
shard is `Syncing` leaves the coordinator state unchanged and `complete_sync`
returns the shard to `Clean`. Whether that during-sync overlay write is in the
checkpoint's snapshot is deliberately out of scope here; its durability is
discharged by the imported libdictenstein contracts `LockFreeDurableCheckpoint.tla`
(committed-watermark capture loses no write) and `LockFreeCounterMergeAtomicity.tla`
(atomic, overflow-checked `increment_cas`). See
`dependencies/libdictenstein-contracts.md`.

`AsyncShardSyncProofs.tla` proves unbounded inductive safety with TLAPS
(249 obligations). The proof decomposes worker pick/process and the checkpoint
actions into helper theorems for function updates and per-action preservation
lemmas, replacing cardinality-heavy partition reasoning with small
set-disjointness and representation lemmas so both TLAPS and TLC stay tractable.

### CronStateMachine.tla

Target: `src/util/cron/mod.rs`

Models task scheduling, due-task execution, sleeping, termination requests,
handle drop, channel close, and panic accounting.

Safety properties:

- `ReadySignalSafety`
- `PanicIsolation`
- `TestCompletionRequiresExecution`
- `TestProgressRequiresExecution`
- `TerminationRequiresRequest`
- `NormalTaskWaitsForEarlierPanic`
- `TaskIdConsistency` in TLC configs

Liveness properties:

- `TestEventuallyCompletes`
- `CronEventuallyTerminates`
- `NormalTaskExecutesAfterPanic`
- `ReadySignalEventuallyReceived`

The liveness model requires the closed-empty channel path to terminate rather
than sleep forever. This matches the Rust `try_recv` disconnect handling.

The model abstracts the channel and task queue as sets to avoid state explosion;
FIFO order is not needed for the verified properties. Due-task execution is
still constrained to the earliest due task, matching the source min-heap and the
panicking-task-before-normal-task scenario. `CronStateMachineProofs.tla` proves
inductive safety with TLAPS; temporal and priority-order properties are checked
by TLC under fairness.

### WorkerShutdown.tla

Target: `src/sources/google_books/importer/worker_pool.rs` and related
state-machine shutdown code.

Models job ownership, stop requests, worker exit, safe checkpoint drain,
force-quit abort, and worker-exit tracking failure.

Safety properties:

- `NoJobLost`
- `JobUniqueOwnership`
- `ProcessingHasJob`
- `SendingHasJob`
- `IdleNoJob`
- `ResultsDisjoint`
- `CheckpointAfterDrain`
- `CheckpointRequiresShutdown`
- `AbortStatesDoNotCheckpoint`

Liveness is checked with `MC_WorkerShutdown_NoSymmetry.cfg` because symmetry can
hide some temporal-property issues in TLC. A graceful shutdown eventually either
reaches `Done` after every worker exits and every pending result is received, or
reaches an explicit no-checkpoint abort state for force quit / drain failure.

`WorkerShutdownProofs.tla` proves inductive safety with TLAPS. The job queue is
modeled as a set because FIFO order is irrelevant to the shutdown properties;
the proof tracks unique ownership across queued, in-worker, pending-result, and
received-result locations. Force-quit and drain-failure transitions are proved
as direct no-checkpoint actions, while the normal checkpoint transition remains
guarded by an empty pending-result set and all workers in `Exited`.

### ImporterLifecycle.tla

Target: `src/sources/google_books/importer/import_ops.rs` and
`src/sources/google_books/state_machine.rs`

Models the outer Google Books import lifecycle after worker processing reaches a
terminal condition: normal completion, graceful cancellation, force quit, and
missing worker results.

Safety properties:

- `CompletedRequiresAllWork`
- `PhaseOrder`
- `NoFinalizeBeforeCheckpoint`
- `EarlyTerminalSkipsFinalize`
- `ForceQuitSkipsCheckpoint`
- `FailedNeverCompletes`
- `TerminalErrorNeverCompletes`

Liveness property:

- `LifecycleEventuallyTerminates`

`ImporterLifecycleProofs.tla` proves inductive safety with TLAPS for the
completion and early-terminal invariants. `PhaseOrder` is kept as a TLC-checked
state-space invariant because it is primarily a traceability property over the
finite phase graph. The model uses compact numeric phase tags internally so the
proof obligations avoid brittle string-disequality reasoning.

The model exposed source requirements that are now enforced in
`import_http_reactive`: early terminal paths run cleanup before returning, force
quit cleans up without checkpointing, graceful cancellation checkpoints only
after workers have drained, and a missing-results abort returns immediately
without falling through to stats, merge, final checkpoint, or completion events.

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

### VarintTermIdView.v

Target: `src/ngram/vocabulary.rs` (`encode_varint`/`decode_varint`) and
`src/ngram/u64_view.rs` (`U64NgramView`).

Proves that the LEB128 varint term-id codec is a self-delimiting bijection, so
the `u64`-unit view the word-level corrector `T_gram` walks over the n-gram store
enumerates exactly the stored term-id sequences — no id dropped, duplicated, or
invented:

- `decode_aux_encode` — the generalized (accumulator/shift) round-trip invariant.
- `decode_encode` — decoding `encode_varint v ++ rest` recovers `(v, rest)`
  (injectivity of the codec together with its self-delimiting property).
- `encode_seq_decode_seq` — VIEW FAITHFULNESS: the concatenated-varint key of a
  term-id sequence decodes back to exactly that sequence (soundness AND
  completeness of the view's enumeration).
- `encode_seq_injective` — distinct term-id sequences produce distinct keys.

The codec is modelled over `nat` (`& 0x7F` as `mod 128`, `& 0x80 == 0` as
`< 128`, `<< shift` as `* 2 ^ shift`); the bijection holds for all naturals. The
store's reservation of term-id `0` (`FIRST_VALID_INDEX = 1`) is an orthogonal
namespace-disjointness invariant, not needed for the codec bijection.

### NoisyChannelDecode.v

Target: `src/integration/grammar_corrector.rs` (the beam decode).

Proves that the cascade's minimum-cost decode is the noisy-channel MAXIMUM A
POSTERIORI correction. With `cost(w) = c_channel(w) + λ·(−ln P(w))` and
`score(w) = exp(−c_channel(w))·P(w)^λ`:

- `neg_ln_reflect` — `−ln` is strictly antitone on the positives.
- `cost_is_neg_log_score` — `cost(w) = −ln(score(w))` (the log-semiring bridge).
- `min_cost_maximizes_score` — the least-cost hypothesis has the greatest score
  (so argmin cost = argmax score over any candidate set).
- `map_score_identity` / `min_cost_is_map` — with `c_channel = −ln P(x|w)` and
  `λ = 1` the score is the joint `P(x|w)·P(w)`, which — the evidence `P(x)` being
  constant in `w` — is proportional to the posterior `P(w|x)`; hence the min-cost
  decode is the Bayes-optimal (MAP) correction (Kernighan, Church & Gale 1990).

## Source Alignment

The verification pass led to implementation changes:

- `Shard::try_acquire_write` uses checked generation increment and rejects
  acquisition after `u64::MAX` instead of wrapping to zero.
- `FrequencyCounts::merge` and `AtomicFrequencyCounts::observe` use checked
  addition so Rocq's bounded-count model matches Rust behavior.
- `CronStateMachine::poll_check_events` now terminates after a previously
  disconnected channel drains its delayed task queue, matching
  `CronCheckChannelClosed`/`CronTerminateClosedFromSleep`.
- Graceful Google Books cancellation now waits for every worker-exit
  notification before saving a checkpoint. If force quit is observed or worker
  exit tracking fails during drain, it returns `ImportError::Interrupted`
  without checkpointing, matching `ForceQuitAborted` and `DrainAborted`.
- Reactive importer early-terminal paths now run the same cleanup guard as the
  normal path before returning. Missing worker results now save an emergency
  checkpoint when possible and return immediately without finalization,
  matching `ImporterLifecycle.tla`.
- `VocabularyIndexedDictionary::root()` now applies the same root-only metadata
  filtering as `MetadataFilteringZipper`, so liblevenshtein-style node
  traversals cannot yield internal `\x00` metadata keys or values. The
  query-facing tests also cover vocabulary-backed frequency lookup, read-only
  OOV behavior, and aggregated shard routing/value lookup.

The checkpoint, async shard sync, cron, worker shutdown, and importer lifecycle
source paths were reviewed against the final TLA+ abstractions. Focused Rust
tests exercise the worker no-checkpoint-before-exit path, and Loom tests
exercise the source-level concurrency assumptions for write-token exclusion,
single active syncer, worker result-before-exit, and cron ready-before-schedule.

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
| Cron priority-order refinement | TLC | Pass |
| Cron typecheck | Apalache | Pass |
| Worker shutdown safety/liveness | TLC | Pass |
| Worker shutdown typecheck | Apalache | Pass |
| Importer lifecycle safety | TLC | Pass |
| Importer lifecycle liveness | TLC | Pass |
| Importer lifecycle typecheck | Apalache | Pass |
| Persistent storage bridge safety | TLC | Pass |
| Persistent storage bridge typecheck | Apalache | Pass |
| Persistent storage bridge inductive safety | TLAPS | Pass |
| Query semantics bridge safety | TLC | Pass |
| Query semantics bridge typecheck | Apalache | Pass |
| Query semantics bridge inductive safety | TLAPS | Pass |
| Query wrapper correspondence tests | Cargo | Pass |
| Dependency-coupled full formal gate | Make | Pass |
| Strict dependency imported TLC/io_uring/Miri gate | Make/TLC/Cargo/Miri | Pass |
| Larger bounded stress models | TLC | Pass under capped `make complete` gate |
| Query bridge stress model | TLC | Pass |
| MKN arithmetic bounds | Rocq | Pass |
| MKN floating-point envelope | Rocq/Flocq/Interval | Pass |
| Frequency-count algebra and bounds | Rocq | Pass |
| Focused Rust alignment tests | Cargo | Pass |
| Cron execution-order correspondence test | Cargo | Pass |
| Shard write-token inductive safety | TLAPS | Pass |
| Checkpoint inductive safety | TLAPS | Pass |
| Async shard sync inductive safety | TLAPS | Pass |
| Cron inductive safety | TLAPS | Pass |
| Worker shutdown inductive safety | TLAPS | Pass |
| Importer lifecycle inductive safety | TLAPS | Pass |
| Formal-alignment schedule exploration | Loom/Cargo | Pass |

## Verification Scope

- TLC checks bounded finite models. The selected bounds cover contention,
  recovery, retry, shutdown, and failure interleavings. TLAPS supplies the
  unbounded inductive safety proofs for the safety invariants listed above.
- Stress configs deliberately add one orthogonal dimension at a time, or use
  symmetry where available, so the complete finite state spaces still finish
  under the default local resource caps.
- Imported dependency TLC checks use the same one-worker heap cap and a
  per-model timeout. The strict dependency gate keeps Miri and io_uring checks
  explicit because they depend on sibling-repository toolchain and kernel
  support.
- TLA+ models atomic operations as linearizable. Loom tests cover selected Rust
  memory-order interleavings for the modeled concurrency contracts.
- Rocq now proves both exact rational MKN bounds and a binary64 magnitude
  envelope. Bit-exact IEEE-754 rounding is outside the MKN property set; the
  verified requirement is finite magnitude plus clamped output ranges.
- Async, cron, and worker liveness assume fair scheduling and eventual retry or
  wake-up opportunities. TLC checks those temporal properties over the bounded
  state spaces.
- The async, cron, and worker models use set abstractions for queues where FIFO
  order is not part of the intended property set. The cron model still preserves
  the source min-heap obligation by constraining execution to the earliest due
  task.
- `ImporterLifecycle.tla` verifies ordering and terminal-control decisions, not
  filesystem durability, external API behavior, event-channel delivery, or the
  byte-level contents of checkpoint files.
