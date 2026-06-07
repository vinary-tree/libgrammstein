# libdictenstein Persistent Storage Contracts

This file records the libdictenstein dependency-level formal contracts that
libgrammstein uses when reasoning about Google Books importer checkpoint
recovery. liblevenshtein adapter and query contracts are recorded in
`formal/dependencies/liblevenshtein-contracts.md`.

Dependency repository: `../libdictenstein`
Verified revision recorded during planning: `02ec4d010109641247c1962465921d2560572f67`
Dependency tree status at latest verification: dirty with reviewed unsafe-ledger
updates in `formal-verification/UNSAFE_INVENTORY.tsv` and
`formal-verification/UNSAFE_CONTRACTS.tsv`

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

Latest local result: the unsafe-boundary ledger was refreshed for the current
`persistent_artrie_char` reclaim, eviction-registry, walk-guard, and disk-I/O
unsafe surface, and `scripts/verify-formal-correspondence.sh` passed. Optional
Miri and TLC sub-gates were not enabled in that default run.

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
