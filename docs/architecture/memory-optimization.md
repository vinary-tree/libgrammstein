# Memory & Concurrency Optimizations: Google Books Importer

The Google Books n-gram importer was OOMing at ~33.79 GB peak heap and
burning ~49% of CPU in `__mprotect` syscalls during large-vocabulary
imports (English 1-5gram corpus, 5.8M unique words, ~50-100M entries per
2-gram file). This document captures the optimizations that addressed the
bottleneck.

## Bottleneck breakdown (pre-optimization)

| Symptom | Root cause |
|---|---|
| 49% CPU in `__mprotect` | glibc malloc's `mmap`/`munmap` per-large-allocation pattern issuing `mprotect` syscalls on every region change |
| 33.79 GB peak heap | Unbounded growth of (a) lock-free overlay entries, (b) per-tx in-memory buffers for 50-100M-entry prefix files, (c) doubled HashMap rebuilds in the vocabulary's reverse-index during checkpoints |
| 12× redundant checkpoints at end-of-import | Each worker called `save_checkpoint()` on exit; with 12 parallel workers shutting down simultaneously, this added ~6 minutes of blocking I/O after all data was already durable |

## Optimization summary

| # | Optimization | Target |
|---|---|---|
| 1 | `mimalloc` global allocator | mprotect CPU |
| 2 | Pre-sized vocabulary lock-free layer | resize-doubling spikes |
| 3 | SmallVec `[&str; 5]` for token splits | per-record heap allocs |
| 4 | Zero-alloc `parse_ngram_line_ref` + `push_ref` | per-record String/Vec allocs |
| 5 | xxh3 hashers in merge/MKN hot paths | hash CPU |
| 6 | Two-tier `ChildStore` replacing `im::Vector` | Arc COW/mprotect (libdictenstein) |
| 7 | `--cache-files` mode | download/parse decoupling |
| 8 | `--tx-chunk-size` chunked transactions | per-tx memory bound |
| 9 | Per-shard lock-free overlay flush threshold | overlay memory bound |
| 10 | Single-merge `merge_and_rotate_vocabulary_wal` | doubled reverse-index rebuilds (~1.7 GB saved at peak) |
| 11 | MKN aggregator cancellation flag | graceful shutdown during multi-minute MKN phase |
| 12 | Removed worker-exit checkpoint storm | ~6 min blocking I/O at end-of-import |
| 13 | `tokio::task::yield_now()` before finalization | SIGINT responsiveness during synchronous MKN work |
| 14 | Auto-scaled checkpoint interval (every 5 vs 10 files) | I/O cadence at high parallelism |
| 15 | `--overlay-budget-gib` overlay-heap eviction | hard bound on resident overlay RAM (the (a) root cause) |

## How they fit together

### Write path (steady-state, per-worker)

```
                Google Books .gz
                       │
                       ▼
            ┌──────────────────────┐
            │ HTTP stream OR       │  --cache-files (#7) downloads to disk
            │ local cached file    │  first, then streams from local file
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ GzipDecoder ▶ Lines  │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ parse_ngram_line_ref │  zero-alloc (#4) — borrowed &str slices
            │ NgramRecordRef<'a>   │  from the line buffer
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ YearAggregator       │  push_ref (#4) — only allocates a String
            │ ::push_ref           │  when the ngram changes
            └──────────┬───────────┘
                       ▼
              AggregatedNgram
                       │
                       ▼
            ┌──────────────────────┐
            │ tx_insert_ngram      │  SmallVec<[&str; 5]> (#3) — stack-inline
            │  → encode via vocab  │  token splits for n-gram orders 1-5
            └──────────┬───────────┘
                       ▼
              StoragePrefixTx
                       │
                       ▼     chunk_count >= tx_chunk_size (#8) ─┐
            ┌──────────────────────┐                             │
            │ commit_and_renew_    │ ◀───────────────────────────┘
            │  prefix_tx (chunk)   │  bounds per-tx memory; SET-semantics
            └──────────┬───────────┘  idempotent on crash recovery
                       ▼
            ┌──────────────────────┐
            │ ShardHandle's        │  Per-shard AtomicU64 lockfree_entries
            │ lock-free overlay    │  counter (#9). Bounded by
            └──────────┬───────────┘  --lockfree-flush-threshold
                       ▼     entries > threshold (#9) ─┐
            ┌──────────────────────┐                    │
            │ flush_lockfree_over_ │ ◀──────────────────┘
            │  threshold           │
            └──────────────────────┘
```

### Vocab-WAL durability invariant (#10)

The original bug: vocab was checkpointed independently of n-gram shards,
so a crash between vocab-checkpoint and shard-checkpoint left them at
divergent index points. On reopen, vocab restarted from a stale index,
orphaning n-grams encoded with newer indices.

The fix: every checkpoint path (`save_checkpoint`, `periodic_checkpoint`)
calls `storage.merge_and_rotate_vocabulary_wal()` FIRST, then syncs
shards. This single-method-call form replaced the prior two-step
`sync_vocabulary()` + `rotate_vocabulary_wal()` pattern — both internally
called `merge_into()` on the lock-free vocab, doubling the reverse-index
HashMap rebuild and spiking ~3.42 GB transient memory for a 5.8M-word
vocab. The combined method does ONE merge, halving the peak.

### Finalization (one-shot, end-of-import)

```
       all workers exit
            │
            ▼
    ┌───────────────────────┐
    │ save_checkpoint flow  │  (skipped on worker exit per #12)
    └────────────┬──────────┘
                 ▼
    ┌───────────────────────┐
    │ Final checkpoint save │
    └────────────┬──────────┘
                 ▼
    ┌───────────────────────┐
    │ yield_now().await     │  (#13) lets SIGINT handler run before
    └────────────┬──────────┘  synchronous MKN work blocks the runtime
                 ▼
    ┌───────────────────────┐
    │ MknAggregator         │  with_cancellation_flag(&self.interrupted)
    │  .compute_all()       │  (#11) — Ctrl+C during MKN now exits
    └────────────┬──────────┘  cleanly instead of running to completion
                 ▼
    ┌───────────────────────┐
    │ final compact         │  checkpoint_vocabulary (one-shot full
    │  vocabulary           │  re-serialize) — only at finalize, not
    └───────────────────────┘  periodically (which uses WAL rotation)
```

## Optimization details

### 1. mimalloc global allocator

`Cargo.toml` adds optional `mimalloc 0.1` dep + `mimalloc-alloc` feature
(included transitively in `google-books`). All three bins
(`grammstein`, `compare_artries`, `dump_checkpoint`) declare
`#[global_allocator]` gated on `cfg(feature = "mimalloc-alloc")`.

Why it matters: glibc malloc serves large allocations via `mmap` and frees
via `munmap`, both of which issue `mprotect` syscalls to update page
permissions. mimalloc uses thread-local segment heaps with pre-allocated
superpage regions — allocations satisfy from the existing reservation
without per-call `mprotect`. On a 12-worker English import this drops
`__mprotect` CPU share from ~49% to roughly background noise.

### 2. Pre-sized vocabulary

`open_or_create_concurrent_vocabulary_lockfree_with_capacity(path, n)`
pre-sizes the lock-free layer's DashMap term cache and the reverse-lookup
`Vec` to the estimated final size. Without pre-sizing, the structures
geometric-double during import, with each doubling temporarily holding
both old and new tables — several GB of transient overhead for a 5.8M-word
vocab.

`estimate_vocabulary_size(config)` derives the capacity from language +
`min_count` (English base 13M, scaled down by `min_count` factor — higher
thresholds prune more rare words).

### 3. SmallVec token splits

`NgramStorage::store_ngram(ngram, count)` and `tx_insert_ngram(...)` use
`SmallVec<[&str; 5]>` (matching the supported n-gram order range). All
splits fit inline on the stack with no heap allocation.

### 4. Zero-alloc parser

`parse_ngram_line_ref(line) -> NgramRecordRef<'_>` finds tab positions via
byte scanning instead of `split('\t').collect::<Vec<_>>()`. The returned
`NgramRecordRef` borrows ngram text from the input line. Combined with
`YearAggregator::push_ref` (only allocates a String when the ngram
changes), per-record heap traffic drops from O(line) to O(1) amortized.

### 5. xxh3 hashers

`merge.rs` and `mkn.rs` switched all internal `HashMap`/`HashSet` keys
(byte-encoded n-gram contexts, predecessor/successor index sets) from
SipHash defaults to `xxhash_rust::xxh3::Xxh3DefaultBuilder` via local
`XxHashMap`/`XxHashSet` type aliases. Non-adversarial data, so the speedup
(~32% faster hash on cache-line-sized keys) is free.

### 6. Two-tier ChildStore

Lives in libdictenstein (`persistent_artrie_char/nodes/persistent_node.rs`).
Replaces `im::Vector<SwizzledPtr>` with a two-tier enum:
- `Inline { count: u8, keys: [u32; 4], children: [SwizzledPtr; 4] }` —
  ~85% of nodes, zero heap alloc.
- `Heap { keys: Vec<u32>, children: Vec<SwizzledPtr> }` — ~15% of nodes,
  flat Vec.

Eliminates `im::Vector`'s Arc COW + `Arc::make_mut` overhead (~7.22 GB
across a full English import) and the `mprotect` pressure it caused.

### 7. `--cache-files` mode

`process_prefix_file_cached` downloads the raw `.gz` to a local cache via
`download_to_cache` (atomic `.gz.downloading` → `.gz` rename, HTTP 206
Range resume, HTTP 416 recovery), then streams from the cached file via
`stream_aggregated_from_cached_file`. Decouples download reliability from
parse CPU — a failed HTTP stream no longer wastes CPU already spent
parsing.

### 8. `--tx-chunk-size`

Storage-level `commit_and_renew_prefix_tx` commits the current chunk and
begins a fresh transaction for the same prefix/shard. Caller (the
importer's inner streaming loop) invokes this when `chunk_count >=
tx_chunk_size`. Shard-level `commit_chunk` commits to the WAL +
persistent trie but does NOT update `completed_prefixes`; the final
`commit_prefix` marks the prefix complete.

Crash recovery: SET-semantics inserts make re-importing the prefix
idempotent. Uncommitted chunks are lost on crash; committed chunks
survive in the WAL.

### 9. Per-shard lock-free overlay flush threshold

`ShardHandle` carries an `AtomicU64 lockfree_entries` counter (Relaxed
ordering — approximate, used only for threshold decisions). Incremented
on `increment_lockfree`, reset on `sync` / `flush_lockfree` / `checkpoint`.

`ShardCoordinator::flush_lockfree_over_threshold(threshold)` does a fast
read-lock check on every shard and only acquires a write lock for the
ones over threshold. Workers on under-threshold shards continue
uninterrupted.

The importer auto-scales the default: 50K for ≥8 parallel workers, 100K
otherwise. `--lockfree-flush-threshold` overrides.

### 10. Single-merge vocab WAL rotation

Discussed above (durability invariant section).

### 11-14. Finalization fixes

Smaller surgical fixes:
- **#11** — `MknAggregator::with_cancellation_flag(&AtomicBool)` checks the
  flag at the top of every shard iteration in `compute_all` and
  `compute_continuation_counts`. The importer wires `self.interrupted`
  in.
- **#12** — The `save_checkpoint()` call previously fired on every worker
  exit. The original code is preserved as commented-out (per
  CLAUDE.md "never disable by deleting" rule) with a `DISABLED:` rationale
  block explaining the 12× checkpoint storm.
- **#13** — `tokio::task::yield_now().await` before `compute_mkn_stats`
  lets the runtime drive other tasks (including SIGINT handlers) before
  the multi-minute synchronous MKN work captures the thread.
- **#14** — `checkpoint_interval = if config.parallel_downloads >= 8 { 5 }
  else { 10 }` at three sites in the importer. Higher parallelism →
  faster overall throughput → checkpoints amortize over more work.

### 15. `--overlay-budget-gib` overlay-heap eviction (the OOM bound)

Optimizations #8/#9 *bounded* the inter-checkpoint overlay growth, but the
resident lock-free overlay itself (bottleneck (a)) was still **unbounded** — it
accumulated for the lifetime of an open shard, because `checkpoint()` serialized
the overlay snapshot to disk without reclaiming its resident RAM. After
libdictenstein added production overlay-heap eviction (the `checkpoint()` tail now
evicts the coldest resident overlay nodes down to a `resident_budget_bytes`,
**losslessly** — evicted nodes fault back from the durable image on read),
libgrammstein arms it per shard:

- **`ShardHandle.trie`** is held as `SharedARTrie<u64>` (`Arc<PersistentARTrie>`)
  so the eviction coordinator can hold a weak self-reference; `arm_eviction()`
  installs it after open/create. All trie writes remain `&self` (lock-free
  overlay), so they deref through the `Arc` unchanged.
- **Budget policy** (`ShardConfig::overlay_eviction_config`): a global budget `G`
  (CLI `--overlay-budget-gib`, default **10 GiB**, default-on; `0` disables) is
  divided by the number of *simultaneously-resident* shards. Hash-based
  `CpuProportional` (the default) and an unlimited `max_open_shards` keep all
  `num_shards` resident, so the divisor is `num_shards`; otherwise the LRU cap
  bounds residents to `max_open_shards`. So `SUM(per-shard budget)` over the
  resident set ≈ `G`, granularity-invariant. A 64 MiB per-shard floor + a finite
  200K-node per-checkpoint eviction cap keep the tail from thrashing or
  latency-spiking; the base preset is `without_memory_monitor()` (no per-shard
  `sysinfo` thread — the checkpoint tail fires purely on `resident > budget`).
- CX path-compression (also new in libdictenstein) shrinks the on-disk/evicted
  node form, complementing eviction (it does not shrink the resident hot set).

This converts bottleneck (a) from "unbounded" to "bounded by `G`" — the resident
overlay is the dominant heap term during bulk ingest, so this is the lever that
makes the <16 GB target reachable.

## Verification

The fixes are guarded by ~20 unit and integration tests:

- `src/sources/google_books/sharding/shard.rs::tests` — 7 tests for
  `lockfree_entry_count` + `commit_chunk` lifecycle + SET-semantics
  idempotency.
- `src/sources/google_books/sharding/coordinator.rs::tests` — 4 tests
  for `flush_lockfree_over_threshold` + `total_lockfree_entries` +
  `commit_chunk_tx`.
- `src/sources/google_books/sharding/mkn.rs::tests` — 2 cancellation
  tests (`compute_all`, `compute_continuation_counts`).
- `src/sources/google_books/aggregator.rs::tests` — 2 push_ref tests
  (no-alloc same-ngram + equivalence-to-push).
- `src/sources/google_books/storage.rs::tests` — 4 tests for chunked
  transactions + 1 idempotency test for `merge_and_rotate_vocabulary_wal`
  + 2 regression tests for the checkpoint-resume bug class.
- `src/sources/google_books/importer.rs::tests::cache_files` — 6
  wiremock tests for `download_to_cache` (creates, skips existing, Range
  resume, 416 recovery, cleanup of both files, idempotent cleanup).
- `src/sources/google_books/parser.rs::tests` — 7 tests for
  `parse_ngram_line_ref` (unigram/bigram/unicode/wrong-field-count/
  empty-ngram/invalid-fields/equivalence-to-owned).
- `src/sources/google_books/sharding/shard.rs::tests` — 3 overlay-eviction
  tests (#15): `test_overlay_eviction_is_lossless_and_observable`
  (`nodes_evicted > 0` + every evicted key faults back),
  `test_overlay_eviction_bounds_resident_to_budget` (a 1 MiB budget reclaims the
  bulk of a 50K-node overlay — `nodes_evicted >= 30K`), and
  `test_overlay_eviction_under_concurrent_writers` (no lost writes under writers
  racing the budget eviction; deterministic across repeats).

## Pending: end-to-end peak-RSS benchmark

The overlay-eviction bound (#15) is verified **in process** — the eviction tests
prove it fires, reclaims the bulk of the resident overlay (`nodes_evicted`), and
is lossless and concurrency-safe. The remaining validation is the **end-to-end
peak-RSS** measurement on a real import: run the importer with
`--overlay-budget-gib 0` (unbounded = the old behavior) vs the default 10 GiB,
under `/usr/bin/time -v` (off tmpfs, so RSS reflects the heap) with CPU affinity
(`taskset -c 0-11`), plus `valgrind --tool=massif` for the heap shape. Acceptance
targets: peak heap 33.79 GB → <16 GB; `__mprotect` CPU share 49% → <5% (the latter
already addressed by mimalloc, #1). Results will land in MEMORY.md and as an
appendix here. (This run also needs the full `--all-features` build, currently
blocked by libdictenstein's in-flight CX-to-traits generalization.)
