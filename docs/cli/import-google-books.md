# `grammstein train import-google-books` — Memory & Reliability Flags

This guide covers three CLI flags that tune the Google Books n-gram importer
for different machine classes and network conditions. They are independent —
you can combine any of them.

## TL;DR — what to set on a 32GB machine

```bash
grammstein train import-google-books \
  --language en \
  --orders 1..=5 \
  --output english.artrie \
  --parallel 8 \
  --tx-chunk-size 250000 \
  --lockfree-flush-threshold 25000 \
  --cache-files
```

## `--cache-files`

**Default: off.** Downloads each `.gz` n-gram file to a local cache directory
before importing from disk, instead of streaming directly from HTTP into the
parser.

| When to enable | When to skip |
|---|---|
| Unstable upstream connection — failed downloads can be retried without re-parsing | Stable, fast network and limited local disk |
| Long-running imports where a network blip mid-stream wastes hours of CPU | Quick single-prefix runs that finish faster than they'd cache |
| Debugging parser/encoder issues against a fixed input | RAM-constrained systems where the extra disk write is unwelcome |

**Mechanics:**
- Cached files live in `{output_path_parent}/grammstein-cache/`.
- Downloads use a `.gz.downloading` suffix and rename atomically to `.gz`
  after the full payload arrives. Interrupted downloads can resume via HTTP
  Range; servers returning 416 Range-Not-Satisfiable trigger an automatic
  full re-download.
- Cached files are deleted on successful import and on final failure (after
  retries are exhausted). They are NOT auto-deleted on deferred/retryable
  errors — a subsequent retry can reuse the partial download.

## `--tx-chunk-size <entries>`

**Default: 500,000.** Maximum n-grams buffered in a single prefix transaction
before a chunked commit. Critical for large prefix files (2-gram files have
50-100M entries × ~4 GB per worker if not chunked).

| Value | Effect |
|---|---|
| `0` | Disable chunking. Buffer the entire prefix file in one transaction. Lowest WAL write frequency; highest peak memory. |
| 100,000–250,000 | Memory-constrained (16–32 GB RAM with parallel=8). Per-tx buffer ≈ 5–25 MB. |
| 500,000 (default) | Balanced for 64+ GB systems. Per-tx buffer ≈ 50 MB. |
| 1,000,000+ | High-memory systems (256+ GB) prioritizing throughput. |

Chunked commits use **SET semantics** so re-importing the same prefix is
idempotent — if the process crashes between chunk commits, the prefix is
re-imported from scratch on resume and the previously-committed chunks are
overwritten with identical values.

## `--lockfree-flush-threshold <entries>`

**Default: auto-scaled** (50,000 for ≥8 parallel workers, 100,000 otherwise).

Maximum lock-free overlay entries per shard before the importer forces a
flush. The overlay is the high-concurrency write buffer in front of each
shard's persistent trie — keeping it bounded prevents unbounded growth
during checkpoint-free stretches.

| Value | Effect |
|---|---|
| 10,000–25,000 | Very memory-constrained. Frequent flushes; lower peak heap but more I/O. |
| 50,000 (default for ≥8 workers) | Standard for high-parallelism imports. |
| 100,000 (default for <8 workers) | Lower parallelism amortizes the flush cost. |
| 200,000+ | Large-memory systems with fast SSD. Fewer flushes; higher peak. |

Setting this explicitly overrides the auto-scaled default. Setting to a
very low value (e.g., 1,000) effectively turns every write into a flush —
useful for debugging but kills throughput.

## Interaction notes

- `--cache-files` is orthogonal to `--tx-chunk-size` and
  `--lockfree-flush-threshold`. The cache layer affects the *download* path;
  the others affect the *write* path.
- On a freshly-resumed import, `--tx-chunk-size` and
  `--lockfree-flush-threshold` re-apply per the new run's CLI args; they
  are not stored in the checkpoint state.
- The `mimalloc-alloc` feature (enabled by default under `google-books`)
  eliminates per-allocation `mprotect` syscall pressure independent of any
  of these flags. There is no CLI knob — it's a compile-time choice.

## See also

- `docs/training/large-corpora.md` — broader corpus-scale strategies.
- `docs/architecture/memory-optimization.md` — the full design story of the
  importer's memory + concurrency optimizations.
- `docs/debugging/checkpoint-resume-bug.md` — the durability invariant the
  chunked-tx + vocab-merge pipeline guards.
