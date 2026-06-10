# libdictenstein Persistent Storage Contracts

This file records the libdictenstein dependency-level formal contracts that
libgrammstein uses when reasoning about Google Books importer checkpoint
recovery. liblevenshtein adapter and query contracts are recorded in
`formal/dependencies/liblevenshtein-contracts.md`.

Dependency repository: `../libdictenstein`
Verified revision: `a46d9c1aa3f1c921214ca68f41d05260741daeaf` (post lock-free
overlay refactor: overlay-default writes, lock collapse `Arc<RwLock<T>>` →
`Arc<T>`, overlay compaction, overlay-backed `DictionaryNode`, and the production
overlay-heap eviction in `checkpoint()` — task #39). Re-pinned from the
planning-era `02ec4d010109641247c1962465921d2560572f67` after the overlay
migration; `scripts/verify-formal-correspondence.sh` passed against this revision
on a clean tree.
Dependency tree status at latest verification: clean (`git status` empty at
`a46d9c1`); the unsafe-ledger files `formal-verification/UNSAFE_INVENTORY.tsv` and
`formal-verification/UNSAFE_CONTRACTS.tsv` are committed.

Current build target: libgrammstein now compiles + tests against the newer
libdictenstein `e2f7681` (which adds the CX-universal path-compressed checkpoint
serializer atop the same eviction surface). The verified-revision re-pin to a
clean `e2f7681`+ rev — and the `verify-formal-correspondence.sh` re-run — are
pending libdictenstein's in-flight CX-to-traits generalization landing (it leaves
the tree dirty and breaks `--all-features` until the `OverlayCompressedSerialize`
impls are wired for the char/vocab tries). The `a46d9c1` contracts above hold
unchanged across that serializer refactor.

## Verification Command

Run this command before treating the dependency contracts below as current:

```bash
cd ../libdictenstein
scripts/verify-formal-correspondence.sh
```

The dependency harness checks Rust correspondence tests, Rocq proofs, TLA+
syntax, and optional bounded TLC runs when `RUN_TLC=1` is set. The default
script includes the persistent ARTrie, WAL atomicity, public durability policy,
checkpoint publication, recovery replay, vocabulary checkpoint, and persistent
end-to-end trace correspondence tests used by this bridge.

Latest local result: `scripts/verify-formal-correspondence.sh` passed against
libdictenstein `a46d9c1` (clean tree) — Rust correspondence tests, Rocq proofs,
and TLA+ syntax all green. The default run skipped the optional bounded TLC
sub-gate (set `RUN_TLC=1` to enable) and Miri.

## Imported Contracts

| libdictenstein artifact | Contract imported by libgrammstein |
| --- | --- |
| `formal-verification/tla+/StorageSyscallOutcome.tla` | A failed, interrupted, cancelled, short, or missing write/sync outcome cannot advance the durable prefix reported to callers. |
| `formal-verification/tla+/AsyncWalGroupCommit.tla` | Group commit publishes FIFO durable WAL prefixes; an acknowledged LSN is covered by the synced prefix. |
| `formal-verification/tla+/PublicDurabilityPolicy.tla` | Immediate and group-commit acknowledgements imply synced WAL coverage; weaker policies do not overclaim durability. |
| `formal-verification/tla+/SharedPersistentConcurrency.tla` | Public mutation, read, sync, checkpoint, and recovery operations are mutually ordered and retain replay evidence for visible writes. |
| `formal-verification/tla+/ConcurrentCheckpointPublication.tla` | Checkpoint publication does not lose visible mutations and does not truncate WAL records needed for recovery. |
| `formal-verification/tla+/PersistentEndToEndTrace.tla` | Mutation, checkpoint publication, compaction rewrite, crash/reopen replay, and vocabulary bijection preservation compose into a recoverable trace. |
| `formal-verification/tla+/PersistentTransactionIncrementRecovery.tla` | Increment aggregation and replay fail before publishing overflowed records; recovery stops at invalid arithmetic prefixes. |
| `formal-verification/tla+/LockFreeCounterMergeAtomicity.tla` | Lock-free checked counter increments and the atomic overlay→persistent merge reject overflow without mutating the overlay, persistent map, or WAL; the u64 overlay counter and i64 WAL delta stay consistent. |
| `formal-verification/tla+/LockFreeDurableCheckpoint.tla` | A lock-free checkpoint that captures the committed watermark loses no write: every visible term is either within the snapshot (≤ watermark) or retained in the WAL and replayed (> watermark), even though writers commit out of LSN order with no lock excluding the checkpoint. |
| `formal-verification/tla+/LockFreeDurableCheckpointEviction.tla` | The eviction-ON checkpoint composes the watermark-bounded WAL reclaim with the eviction `DiskLocationRegistry` publication ordering: evicting a cold overlay node to its durable image loses no value — every evicted term faults back from the published image (or the retained WAL) on read. This is the lossless guarantee the importer's overlay-budget eviction relies on. |
| `formal-verification/rocq/Spec/PersistentWalAtomicitySpec.v` | Persistent mutation writes WAL records before making trie mutations visible, and committed transactions are atomic at the dependency boundary. |
| `formal-verification/rocq/Spec/PersistentVocabWalAtomicitySpec.v` | Vocabulary insert and batch insert write WAL records before visible mutation and preserve stable index mappings. |
| `formal-verification/rocq/Spec/PersistentVocabCheckpointSpec.v` | Vocabulary checkpoint/reopen preserves the term-index bijection, publishes sidecars consistently, and resumes WAL LSN allocation after the checkpoint. |
| `formal-verification/rocq/Spec/PersistentRecoveryReplayCompletenessSpec.v` | Recovery replays every mutating WAL variant through the durable prefix and stops at corrupt or invalid suffixes. |

## Libgrammstein Bridge Obligations

The bridge model in `formal/tla/PersistentStorageBridge.tla` discharges the
libgrammstein-side ordering obligations that remain after importing the
dependency contracts:

1. Completed importer prefixes are not claimed recoverable until their data trie
   updates are durable.
2. Checkpoint metadata is durable before libgrammstein publishes a new
   checkpoint claim.
3. A failed storage sync cannot advance a checkpoint claim unless a later sync
   establishes durable data coverage.
4. Graceful cancellation can publish a checkpoint only after worker draining has
   completed.
5. Force quit and drain failure paths do not publish a new checkpoint claim.
6. Vocabulary-backed imports require durable vocabulary checkpoint evidence
   before recovered n-gram keys are interpreted after reopen.
7. Single-trie storage treats data and metadata as one durability boundary after
   a checkpoint claim; sharded storage requires the auxiliary checkpoint
   metadata trie to be durable independently of shard data tries.

## Source Mapping

| Bridge concept | libgrammstein source |
| --- | --- |
| Single-trie data and checkpoint metadata | `src/sources/google_books/storage.rs` |
| Sharded data tries and auxiliary checkpoint trie | `src/sources/google_books/storage.rs` |
| Shard prefix commit and sync boundary | `src/sources/google_books/sharding/shard.rs` |
| Worker drain and graceful/force shutdown ordering | `src/sources/google_books/importer/import_ops.rs` |
| Import lifecycle checkpoint publication | `src/sources/google_books/state_machine.rs` |

The bridge intentionally stays at the importer/checkpoint composition layer.
The WAL syscall, checkpoint publication, vocabulary, and replay internals remain
owned by libdictenstein's formal verification suite.

## Async Shard Sync Coordinator Delegation

`formal/tla/AsyncShardSync.tla` models the `ShardSyncCoordinator` state machine
(`src/sources/google_books/sharding/shard.rs`) under the lock-free overlay write
path. After the overlay migration, worker writes (`increment_cas` + `mark_dirty`)
proceed concurrently with an in-flight checkpoint sync; `mark_dirty` only
transitions Clean → Dirty, so a write that lands while a shard is `Syncing`
leaves the coordinator state unchanged and `complete_sync` returns the shard to
`Clean`. AsyncShardSync deliberately does NOT model whether that during-sync
overlay write is captured by the checkpoint's snapshot — that durability question
is discharged by the two imported lock-free contracts above:

- `LockFreeCounterMergeAtomicity.tla` covers the `increment_cas` counter write
  abstracted by `WorkerProcessJob`: it is atomic and overflow-checked.
- `LockFreeDurableCheckpoint.tla` covers the during-sync window: a write committed
  after the captured watermark is retained in the WAL and replayed on reopen, so
  marking the shard `Clean` after `complete_sync` cannot silently drop it.

Thus AsyncShardSync's `AtMostOneSyncer` / `CheckpointAtomicity` (coordinator-level
safety, machine-checked in `AsyncShardSyncProofs.tla`) compose with the two
lock-free durability contracts to cover the full "writes during sync are safe and
never lost" argument. This replaces the retired `ShardWriteToken.tla`
defer-and-exclude model, whose write-token mechanism no longer exists in
production.

## Overlay-Heap Eviction Delegation

The importer arms libdictenstein's production overlay-heap eviction per shard
(`ShardHandle::arm_eviction` → `EvictableARTrie::enable_eviction` with a
`resident_budget_bytes`; see `docs/architecture/memory-optimization.md` #15). The
checkpoint tail then evicts the coldest resident overlay nodes down to the budget
to bound peak heap. libgrammstein does NOT re-verify that eviction is lossless —
that is exactly `LockFreeDurableCheckpointEviction.tla`'s guarantee (an evicted
node's value faults back from the published durable image or the retained WAL on
the next read). The libgrammstein-side obligation is only the *budget arithmetic*
(global budget ÷ resident-shard count) and the lifecycle (the eviction
coordinator's weak self-reference is torn down when the shard's last `Arc` drops),
which are covered by the importer's own eviction unit/concurrency tests in
`src/sources/google_books/sharding/shard.rs::tests`, not by a TLA model.
