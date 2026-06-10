# Formal Verification Task Register

This register keeps verification scope, evidence, and boundary notes in-repo
next to the executable models and proof gates.

## Dependency Crash-Semantics Bridge

Status: local bridge implemented; dependency and full local gates passing

Objective: connect libgrammstein's importer/checkpoint models to the persistent
trie, WAL, checkpoint, vocabulary, and recovery guarantees already verified in
`libdictenstein`.

### Reason

The current libgrammstein TLA+ models verify importer scheduling, checkpoint
state transitions, async shard sync, worker shutdown, cron behavior, and outer
import lifecycle ordering. They intentionally abstract the lower storage layer:
`save_checkpoint`, shard sync/checkpoint, trie mutation, vocabulary checkpoint,
and reopen/recovery are modeled as atomic success or failure steps.

That abstraction is useful, but it leaves one important bridge unproved in this
repository: a completed importer prefix should be recoverable after a crash only
when the corresponding trie data, checkpoint metadata, and vocabulary index state
are covered by the dependency's durable-prefix and checkpoint guarantees.

### Existing Dependency Evidence To Reuse

Before adding new proof logic here, import and cite the relevant checked
libdictenstein and liblevenshtein contracts:

| Dependency artifact | Contract to reuse in libgrammstein |
| --- | --- |
| `formal-verification/tla+/StorageSyscallOutcome.tla` | Failed, short, interrupted, cancelled, or missing write/sync outcomes cannot advance the reported durable prefix. |
| `formal-verification/tla+/AsyncWalGroupCommit.tla` | Group commit publishes only FIFO durable LSN prefixes and returned LSNs are covered by the synced prefix. |
| `formal-verification/tla+/PublicDurabilityPolicy.tla` | Immediate/group-commit public acknowledgements imply synced WAL coverage; periodic/none acknowledgements do not overclaim durability. |
| `formal-verification/tla+/SharedPersistentConcurrency.tla` | Shared public writes, reads, sync, checkpoint, and recovery are mutually ordered and retain replay evidence for racing visible writes. |
| `formal-verification/tla+/ConcurrentCheckpointPublication.tla` | Mutation and checkpoint publication are mutually ordered; checkpoints do not lose visible mutations or truncate required WAL tails. |
| `formal-verification/tla+/PersistentEndToEndTrace.tla` | Public mutation, checkpoint publication, compaction rewrite, crash/reopen replay, and vocabulary bijection preservation compose into a recoverable trace. |
| `formal-verification/tla+/PersistentTransactionIncrementRecovery.tla` | Transaction increment aggregation and replay fail before publishing overflowed records, and recovery stops at invalid arithmetic prefixes. |
| `formal-verification/rocq/Spec/PersistentWalAtomicitySpec.v` | Persistent write-before-mutation and transaction commit laws for byte/char tries. |
| `formal-verification/rocq/Spec/PersistentVocabWalAtomicitySpec.v` | Vocabulary insert and batch insert WAL-before-visible-mutation plus stable index/bijection laws. |
| `formal-verification/rocq/Spec/PersistentVocabCheckpointSpec.v` | Vocabulary checkpoint/reopen bijection, sidecar publication, and post-checkpoint LSN continuation laws. |
| `formal-verification/rocq/Spec/PersistentRecoveryReplayCompletenessSpec.v` | Recovery replay covers all mutating WAL variants through the durable prefix and stops at corrupt or invalid suffixes. |
| `scripts/verify-formal-correspondence.sh` | Dependency-level Rust/Rocq/TLA+/unsafe-boundary correspondence harness to run before trusting imported contracts. |
| `../liblevenshtein-rust/docs/verification/FORMAL_VERIFICATION_MANIFEST.tsv` | Conservative manifest separating trusted, partial, legacy, and debug proof/model assets. |
| `../liblevenshtein-rust/scripts/verify-formal.sh trusted` | Trusted-contract audit for active proof gaps, assumptions, contracts, and evidence surfaces. |
| `../liblevenshtein-rust/docs/verification/core/theories/Core/MinLemmas.v` | Trusted minimum/substitution-cost helper lemmas supporting the edit-distance proof layer. |
| `../liblevenshtein-rust/docs/verification/core/theories/Triangle/SubstCostTriangle.v` | Trusted substitution-cost triangle helper used by metric reasoning. |
| `../liblevenshtein-rust/docs/verification/msm/theories/Indexing/IntervalCost.v` | Trusted interval-relaxed MSM per-element lower bounds. |
| `../liblevenshtein-rust/docs/verification/msm/theories/Indexing/QuantizationBounds.v` | Trusted executable quantization/bin-bound soundness for interval MSM indexing. |
| `../liblevenshtein-rust/docs/verification/msm/theories/Indexing/IntervalColumn.v` | Trusted interval-column admissibility and pruning soundness. |
| `../liblevenshtein-rust/docs/verification/tla/MsmTrieSearch.tla` | Trusted interval-pruned trie search model for MSM range search: no false positives, no missed matches, prune soundness, and termination under configured bounds. |
| `../liblevenshtein-rust/docs/verification/tla/ValueYieldingQuery.tla` | Supporting, non-trusted value-yielding dictionary query model; useful review evidence, but manifest status is partial. |
| `../liblevenshtein-rust/tests/persistent_artrie_integration.rs` | Persistent ARTrie dictionary/zipper/transducer integration coverage for liblevenshtein's libdictenstein re-export. |

### Libgrammstein Assumptions To Discharge

Track these assumptions explicitly in a bridge model:

1. `NgramStorage::SingleTrie` checkpoint metadata and n-gram data share the same
   persistent trie durability boundary.
2. `NgramStorage::Sharded` stores checkpoint metadata in the auxiliary
   `{output_path}.checkpoint.artrie` trie, while each shard has its own data
   trie and WAL/sync boundary.
3. `ShardHandle::commit_prefix` publishes a prefix only after the underlying
   trie transaction has either fully committed or failed without visible partial
   checkpoint state.
4. `ShardHandle::sync` and coordinator checkpoint paths do not mark a shard
   clean unless the dependency durable-prefix contract says the relevant WAL
   prefix is synced.
5. `GoogleBooksImporter::save_checkpoint` must not claim a prefix/order is
   recoverable unless the checkpoint metadata write and all required storage
   sync/checkpoint operations have succeeded.
6. Graceful cancellation may save a checkpoint only after worker shutdown has
   drained all in-flight writers and pending results.
7. Force quit and drain failure must not publish a new durable checkpoint claim.
8. Vocabulary-backed imports must preserve stable term-index bijection across
   checkpoint/reopen before n-gram keys are interpreted after recovery.

### Delivered Artifacts

1. Added dependency contract files under `formal/dependencies/` summarizing the
   exact libdictenstein and liblevenshtein artifact versions, commands, and
   contracts used by libgrammstein.
2. Added a compact TLA+ bridge model, `formal/tla/PersistentStorageBridge.tla`,
   with configs for single-trie and sharded storage.
3. Added TLC properties for:
   - completed prefix recovery requires durable data and durable checkpoint
     metadata,
   - failed storage sync cannot advance importer checkpoint claims,
   - graceful cancellation checkpoint recovery is sound after worker drain,
   - force quit publishes no new checkpoint claim,
   - vocabulary indices remain stable across checkpoint/reopen.
4. Added Apalache typechecking for the bridge model.
5. Added TLAPS inductive safety proofs for the bridge invariants.
6. Added Rust correspondence tests that exercise the bridge assumptions through
   libgrammstein public or crate-local APIs:
   - successful single-trie checkpoint metadata reopen,
   - successful sharded checkpoint metadata reopen,
   - completed prefix data plus metadata recovery,
   - graceful-cancel checkpoint recovery,
   - force-quit no-new-checkpoint behavior,
   - vocabulary checkpoint/reopen index stability.
7. Added the bridge checks to `formal/Makefile` under the regular `check` and
   `complete` gates once the model and correspondence tests are stable.

### Current Implementation State

- Dependency contract files added:
  `formal/dependencies/libdictenstein-contracts.md` and
  `formal/dependencies/liblevenshtein-contracts.md`.
- TLA+ bridge model added: `formal/tla/PersistentStorageBridge.tla`.
- TLC configs added for single-trie, sharded-with-vocabulary, and stress runs.
- TLAPS proof module added:
  `formal/tla/PersistentStorageBridgeProofs.tla`.
- Formal Makefile wiring added for TLC safety, Apalache, stress, TLAPS, and
  Rust correspondence tests filtered by `persistent_storage_bridge`.
- `formal/Makefile` exposes an explicit dependency-contract gate so the
  imported libdictenstein and liblevenshtein assumptions can be refreshed
  independently of the local libgrammstein checks.
- `../libdictenstein` unsafe-boundary ledgers were refreshed for the current
  `persistent_artrie_char` reclaim, eviction-registry, walk-guard, and disk-I/O
  unsafe surface. Its default formal correspondence harness now passes.
- `../liblevenshtein-rust/scripts/verify-formal.sh trusted` passed for the
  trusted manifest subset.

### Scope Boundaries

1. `../liblevenshtein-rust/docs/verification/tla/ValueYieldingQuery.tla`
   remains manifest-marked `partial`, so libgrammstein does not import it as a
   trusted dependency contract. `formal/tla/QuerySemanticsBridge.tla` covers
   libgrammstein's wrapper-level query obligations instead.
2. The local storage bridge verifies crash-recovery claims at the
   importer/checkpoint boundary. It imports, rather than reproves,
   libdictenstein WAL syscall, checkpoint publication, replay completeness, and
   vocabulary bijection internals.

### Acceptance Gates

The bridge task is complete only when all of the following pass on the current
tree:

```bash
cd ../libdictenstein
scripts/verify-formal-correspondence.sh

cd ../liblevenshtein-rust
scripts/verify-formal.sh trusted

cd ../libgrammstein
make -C formal dependency-contracts
make -C formal strict-dependency-contracts MIRI_TOOLCHAIN=nightly
make -C formal complete TLAPM=/home/dylon/.local/tlaps/bin/tlapm
cargo test --features google-books persistent_storage_bridge
```

Latest coupled evidence: `make -C formal complete-with-dependencies
TLAPM=/home/dylon/.local/tlaps/bin/tlapm` passed. That run refreshed the
libdictenstein formal-correspondence contracts, refreshed the liblevenshtein
trusted formal subset, and then ran the local capped `complete` gate.

Latest strict dependency evidence:

- `make -C formal dependency-imported-tlc` passed with one TLC worker,
  `-Xmx768m`, and a 120 second timeout per imported libdictenstein model.
- The largest imported dependency state space was
  `PersistentTransactionIncrementRecovery`: 18,056,329 generated states,
  2,891,300 distinct states, depth 20, completed in 57 seconds.
- Other imported libdictenstein models passed with smaller bounded graphs:
  `StorageSyscallOutcome`, `AsyncWalGroupCommit`, `PublicDurabilityPolicy`,
  `SharedPersistentConcurrency`, `ConcurrentCheckpointPublication`, and
  `PersistentEndToEndTrace`.
- `make -C formal dependency-io-uring` passed the io_uring completion-contract
  tests and the persistent ARTrie storage-correspondence test suite with the
  `persistent-artrie io-uring-backend` features.
- `make -C formal dependency-miri-unsafe MIRI_TOOLCHAIN=nightly` passed the
  strict-provenance Miri unsafe-boundary tests for raw child ownership,
  swizzled-pointer reclamation, vocabulary leaf eviction, heap node-map
  parent-chain tracking, and fixed-buffer registration.

## liblevenshtein Query-Semantics Bridge

Status: implemented; full local gate passing

Objective: use liblevenshtein's extensive trusted formal suite for dictionary,
metric, articulatory, and MSM-indexed search reasoning without importing
manifest-partial query models as trusted libgrammstein contracts.

### Reason

The persistent-storage bridge proves that libgrammstein does not claim a
recoverable importer prefix before the underlying data, metadata, and
vocabulary evidence is durable. It intentionally does not prove higher-level
query semantics over the recovered dictionaries.

liblevenshtein already has trusted Rocq proof islands for edit-distance
definitions, metric properties, weighted articulatory distance, WallBreaker
pigeonhole reasoning, and MSM interval-index pruning, plus a trusted
`MsmTrieSearch.tla` model. It also has useful partial TLA+ models for ordinary
value-yielding and priority queries. The next improvement is to connect only the
trusted dependency claims to libgrammstein's query-facing wrappers, and either
promote or locally bridge the partial value-yielding model before relying on it.

### Delivered Artifacts

1. Added a small libgrammstein query bridge model so libgrammstein has trusted
   wrapper-level value-yielding semantics without importing liblevenshtein's
   manifest-partial `ValueYieldingQuery.tla`.
2. Added Rust correspondence tests for libgrammstein wrappers that present
   liblevenshtein dictionary/query interfaces, including:
   `VocabularyDictionary`, `VocabularyIndexedDictionary`,
   `AggregatedLanguageModelDictionary`, and metadata-filtering zippers.
3. Updated `formal/dependencies/liblevenshtein-contracts.md` to keep
   liblevenshtein's manifest so trusted, partial, legacy, and debug assets remain
   clearly separated.
4. Wired the query bridge into the capped TLC, Apalache, stress, TLAPS, and
   Rust-alignment gates.

### Current Implementation State

- Local TLA+ bridge artifacts were added:
  `formal/tla/QuerySemanticsBridge.tla`,
  `formal/tla/QuerySemanticsBridge.cfg`,
  `formal/tla/QuerySemanticsBridge_NoMetadata.cfg`,
  `formal/tla/QuerySemanticsBridge_Stress.cfg`, and
  `formal/tla/QuerySemanticsBridgeProofs.tla`.
- The query bridge is wired into `formal/Makefile` for capped TLC safety,
  Apalache typechecking, stress, and TLAPS proof targets, plus Rust alignment
  filters for `vocabulary_query` and `metadata_filtering_zipper`.
- `VocabularyIndexedDictionary::root()` now applies the same root-only
  `\x00` metadata filtering as `MetadataFilteringZipper`, so
  liblevenshtein-style `DictionaryNode` traversals cannot emit checkpoint or
  vocabulary metadata as query results.
- `VocabularyIndexedDictionary::len()` and `is_empty()` use the backend O(1)
  length fast path when no root metadata edge exists, and fall back to a
  filtered visible-final count only when metadata is present.
- The `IterableDictionary` blanket implementation for
  `VocabularyIndexedDictionary<D>` now carries the explicit char-node bound
  required by the metadata-filtered query surface.
- Focused Rust correspondence tests were added for:
  `VocabularyDictionary` frequency lookup, `VocabularyIndexedDictionary`
  root traversal and value-yielding traversal, read-only OOV vocabulary
  behavior, and `AggregatedLanguageModelDictionary` shard routing/value lookup.
- `cargo test --features google-books vocabulary_query` passed locally.
- `QuerySemanticsBridge` TLC safety passed for metadata-present and
  no-metadata configurations under a 384 MB heap and 20 second timeout. Each
  run checked the complete state graph: 31 generated states, 6 distinct states,
  depth 3.
- `QuerySemanticsBridge_Stress.cfg` passed under the same cap.
- `QuerySemanticsBridge.tla` Apalache typechecking passed with a 768 MB heap
  and 30 second timeout.
- `QuerySemanticsBridgeProofs.tla` passed TLAPS with 28 obligations proved.
- After a local resource incident, `formal/Makefile` now caps TLC and Apalache
  defaults (`TLC_JAVA_OPTS="-XX:+UseParallelGC -Xmx768m"`, one TLC worker,
  60 second TLC timeouts, `APALACHE_JVM_ARGS=-Xmx1536m`) and bounds Apalache
  and TLAPS invocations with explicit timeouts.

### Scope Boundaries

1. If liblevenshtein later promotes `ValueYieldingQuery.tla` to trusted,
   libgrammstein can import that dependency contract directly after refreshing
   `formal/dependencies/liblevenshtein-contracts.md` and rerunning
   `make -C formal dependency-contracts`.
2. If libgrammstein adds direct MSM or articulatory query features, source-level
   tests should map those APIs to the trusted `IntervalCost.v`,
   `QuantizationBounds.v`, `IntervalColumn.v`, and `MsmTrieSearch.tla` claims.

### Acceptance Gates

```bash
cd ../liblevenshtein-rust
scripts/verify-formal.sh trusted

cd ../libgrammstein
make -C formal dependency-contracts
cargo test --features google-books vocabulary_query
cargo test metadata_filtering_zipper
```

Latest local evidence: the query bridge was included in the passing capped
`make -C formal complete TLAPM=/home/dylon/.local/tlaps/bin/tlapm` gate,
including TLC safety/stress, Apalache typechecking, TLAPS, and Rust alignment
tests.

## Cron Priority-Semantics Refinement

Status: implemented; cron-only model gates passing

Objective: close the gap between the cron TLA+ set abstraction and the Rust
`BinaryHeap` min-heap execution path without reintroducing FIFO/channel ordering
state explosion.

### Delivered Artifacts

1. Tightened `formal/tla/CronStateMachine.tla` so task execution chooses an
   `EarliestDueTask`, matching `CronStateMachine::execute_one_task` popping the
   earliest scheduled task from the min-heap.
2. Added `NormalTaskWaitsForEarlierPanic` to the cron TLC configs, proving the
   model cannot execute the delayed normal task before the earlier panicking
   task in the panic-safety scenario.
3. Added `test_execute_one_task_uses_earliest_due_task` to exercise the source
   execution path, not just raw heap ordering.
4. Added Makefile timeouts around Apalache and TLAPS invocations so future full
   formal gates remain resource-bounded.

### Evidence

- `CronStateMachine.cfg` TLC safety passed under `-Xmx384m` and `timeout 20s`:
  5,112 generated states, 3,139 distinct states, depth 217.
- `CronStateMachine_Liveness.cfg` TLC liveness passed under the same cap:
  2,752 generated states, 1,659 distinct states, depth 137.
- `CronStateMachine_Stress.cfg` TLC stress passed under the same cap:
  6,292 generated states, 3,879 distinct states, depth 257.
- `CronStateMachine.tla` Apalache typechecking passed under `-Xmx768m` and
  `timeout 30s`.
- `CronStateMachineProofs.tla` passed TLAPS under `timeout 60s` with 231
  obligations proved.
- `cargo test util::cron::tests::test_execute_one_task_uses_earliest_due_task`
  passed.

## Full Capped Formal Gate Evidence

Status: SUPERSEDED (pre-refactor run) — coupled re-run pending libdictenstein
stabilization. See "Phase C re-verification" below.

> ⚠ The lock-free overlay migration (Phase C) invalidated parts of this
> pre-refactor evidence: `ShardWriteToken` was retired from the gate, and
> `AsyncShardSync` was reworked to the lock-free model. The TLA-only portion for
> the reworked spec was re-verified this session (see "Phase C re-verification"
> below); the full coupled gate (`make complete-with-dependencies`, which runs
> the dependency-contract refresh and the cargo correspondence/loom suites) must
> be regenerated against the post-refactor libdictenstein once its working tree
> is clean.

Latest coupled command:

```bash
make -C formal complete-with-dependencies TLAPM=/home/dylon/.local/tlaps/bin/tlapm
```

The local portion of the gate passed with the Makefile defaults of one TLC
worker, `-Xmx768m` for TLC, 60 second TLC and Apalache timeouts, `-Xmx1536m`
for Apalache, and a 120 second TLAPS timeout.

Notable bounded stress results from the passing run:

- Checkpoint stress: 514,808 generated states, 69,151 distinct states, depth
  20.
- Async shard sync stress (pre-refactor; superseded by the Phase C re-run
  below): 234,087 generated states, 88,635 distinct states, depth 27.
- Cron stress: 6,292 generated states, 3,879 distinct states, depth 257.
- Worker shutdown stress: 273,749 generated states, 67,032 distinct states,
  depth 30, temporal properties checked.
- Persistent storage bridge stress: 280 generated states, 216 distinct states,
  depth 10.
- Query semantics bridge stress: 31 generated states, 6 distinct states, depth
  3.

The passing TLAPS portion (pre-refactor) proved the local proof modules,
including 193 checkpoint, 231 cron, 103 worker-shutdown, 157 importer lifecycle,
125 persistent-storage bridge, and 28 query bridge obligations. The retired
`ShardWriteTokenProofs` (214 obligations) is no longer in the gate; the reworked
`AsyncShardSyncProofs` is now 249 obligations (was 305), re-verified this session
(see below).

The coupled gate also ran the dependency-contract refresh first:
`../libdictenstein/scripts/verify-formal-correspondence.sh` and
`../liblevenshtein-rust/scripts/verify-formal.sh trusted` both completed before
the local Rocq, TLC, Apalache, Rust correspondence, Loom, stress, and TLAPS
checks. Dependency crates still emit warnings during Cargo builds, but the
formal gate exits successfully.

Additional strict dependency checks now have a reproducible Make target:

```bash
make -C formal strict-dependency-contracts MIRI_TOOLCHAIN=nightly
```

The strict dependency target composes the baseline dependency-contract refresh
with imported libdictenstein TLC checks, optional io_uring correspondence tests,
and Miri unsafe-boundary checks. The imported TLC run stays capped at one worker,
`-Xmx768m`, and 120 seconds per dependency model to avoid unconstrained state
exploration on developer machines.

### Phase C re-verification (lock-free overlay migration; TLA-only, this session)

The lock-free `AsyncShardSync` rework and the `ShardWriteToken` retirement were
re-verified with the libdictenstein-independent TLA tools (TLC, Apalache, TLAPS).
These runs do not require a compiling libdictenstein and were reproduced with the
Makefile defaults (one TLC worker, `-Xmx768m`, `-Xmx1536m` for Apalache):

- `AsyncShardSync.cfg` TLC safety: 1,375,508 generated, 358,932 distinct, depth
  31, no invariant violation (TypeOK, AtMostOneSyncer, CheckpointAtomicity,
  SyncerConsistency, CleanMeansZeroDirty, WorkerStateJobConsistency, JobPartition).
- `AsyncShardSync_Liveness.cfg` TLC liveness: 574 generated, 249 distinct, depth
  18 — `CheckpointEventuallyCompletes` and `AllJobsEventuallyComplete` hold.
- `AsyncShardSync_Stress.cfg` TLC stress: 205,152 generated, 77,799 distinct,
  depth 27, no violation.
- `AsyncShardSync.tla` Apalache typecheck: OK ("types are purrfect").
- `AsyncShardSyncProofs.tla` TLAPS: all 249 obligations proved.
- Full TLA gate (`make -C formal tla-safety tla-liveness apalache`): all 9 safety
  + 5 liveness TLC runs and 7 Apalache typechecks pass with `ShardWriteToken`
  removed — no collateral regression in the eight retained specs.

Reconciliation against libdictenstein `62ea161` (post overlay refactor +
production overlay-heap eviction + CX-to-traits checkpoint serializer) is
COMPLETE: the verified-revision re-pin landed in
`dependencies/libdictenstein-contracts.md` — `make -C formal
complete-with-dependencies` (which runs `verify-formal-correspondence.sh`) passed
on 2026-06-10, the tree clean within the contract surface bar a one-line io-uring
char-ctor warning-hygiene change (`entry_count` → `_entry_count`, out of contract
surface) plus untracked benchmark logs. The cargo `--all-features` lib suite is
green (933 passed / 0 failed); the full formal gate is green (dependency
contracts + rocq + 21 TLC runs no-error + 7 TLAPS modules [incl. AsyncShardSync's
249 obligations] + 7 Apalache typechecks + rust-alignment + `--all-features`
source-hygiene); the loom alignment tests pass (3/3). The optional `RUN_TLC=1`
dependency-imported-TLC sub-gate and Miri were not run.

## Source Alignment And Warning Hygiene

Status: local source gate passing; dependency warnings remain outside this repo

Latest local evidence:

```bash
make -C formal source-hygiene
```

Latest run: passed. In the sandboxed Codex environment this target needs
permission to bind loopback ports because the all-feature Google Books cache
tests start local `wiremock` servers.

This target runs:

```bash
rustfmt --edition 2021 --check <touched Rust files>
git diff --check
source marker hygiene scan over source, tests, manifest, README, and formal
assets for skipped-test attributes, common debt markers, implementation-hole
macros, and fixture wording that implies work remains
documentation scan for common high-signal debt markers
cargo check --all-targets --all-features
local warning audit over cargo JSON diagnostics
cargo test --lib --all-features -- --test-threads=1
```

Results:

- `libgrammstein` no longer emits local warnings in the all-targets,
  all-features check. The `local-warning-hygiene` target parses Cargo JSON
  diagnostics and fails on warnings whose package id is `libgrammstein`, while
  printing a non-fatal dependency-warning summary for the cross-repository
  cleanup plan.
- The all-features library suite passed single-threaded: 931 passed, 0 failed,
  0 ignored.
- `make -C formal check` now starts with `source-hygiene-fast`, so formatting,
  whitespace, and marker failures stop before expensive TLC, Apalache, and
  TLAPS work. `make -C formal complete` also includes `source-hygiene-full`, so
  the complete formal gate covers post-verification all-feature compilation and
  the full all-feature library suite as well as the Rocq, TLC, Apalache, Rust
  correspondence, stress, and TLAPS checks.
- The cleanup removed or discharged warning-backed implementation gaps by:
  exposing intended public helper APIs from private modules, using stored
  configuration and capacity fields, bounding the AST parse-tree cache,
  validating BERT embedding dimensions against detected model config,
  removing unused duplicate trie-mining wrappers, consolidating Wikipedia
  dump URL generation through the language module, replacing test mock grammar
  panics with a real test grammar, initializing grammar constraints at
  construction, enabling the grammar-corrector tests, adding cached Wikipedia
  corpus download helpers with sample/resume behavior, enabling
  Wikipedia/Gutenberg evaluation readers, and replacing external ModernBERT
  model-download tests with deterministic configuration/device/load-error
  tests.
- Remaining Cargo warnings are in sibling dependency crates:
  `../libdictenstein`, `../liblevenshtein-rust`, and `../PathMap`. They are not
  editable from the current libgrammstein workspace without explicit
  cross-repository write approval.
- `make -C formal dependency-warning-hygiene` is the explicit failing gate for
  this boundary. Current observed warning counts are:
  - `../libdictenstein`: 16 warnings
  - `../liblevenshtein-rust`: 1 warning in the all-feature local gate, with 5
    warnings under neural-rescore-only compilation because serialization-disabled
    helpers also trigger missing-docs diagnostics.
  - `../PathMap`: 5 warnings

Cross-repository warning cleanup plan:

- `../PathMap`: remove unused compatibility imports in `src/trie_map.rs` and
  `src/morphisms.rs`; either remove or deliberately expose the unused
  `gxhash128`, `HashMapExt`, and `HashSetExt` shim surface in `src/lib.rs`.
- `../liblevenshtein-rust`: remove the unused
  `OnlinePhoneticTransducerChar` import from `src/phonetic/grep_online.rs` and
  add the missing public docs for serialization-disabled helpers in
  `src/phonetic/llre/compiled.rs` when the `serialization` feature is off.
- `../libdictenstein`: discharge unused-result warnings in
  `PersistentARTrieChar` iterator construction, and either remove, use, or
  deliberately gate the currently dead helper surfaces reported in
  `persistent_artrie`, `persistent_artrie_char`,
  `persistent_artrie_core`, `persistent_vocab_artrie`, `scdawg`, and
  `suffix_automaton`.
