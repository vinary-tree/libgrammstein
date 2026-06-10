# libgrammstein → libdictenstein single-lock-free vocab migration

libdictenstein deleted `ConcurrentVocabARTrie` / `LockFreeVocab` (the wrappers) — `PersistentVocabARTrie`
IS the single lock-free impl now: `insert`/`insert_batch`/`insert_with_index` + `checkpoint`/`sync` are
all `&self`, so many threads insert through a shared `Arc<PersistentVocabARTrie>` with NO external lock.
This migration replaces the deleted wrappers across 4 files.

## Semantic + perf note (BENCHMARK)
The old `ConcurrentVocabARTrie::insert_cas` was IN-MEMORY (durability via checkpoint only). The new
`PersistentVocabARTrie::insert` is DURABLE Order-A (per-insert WAL append → overlay CAS → CommitRank).
IMPORTANT: this durable insert REQUIRES `durability_policy` ∈ {Immediate, GroupCommit} and REJECTS
`None`/`Periodic` (an acknowledged write must be durable BEFORE it becomes visible, so a buffered "cheap"
policy is rejected at insert time). The factories therefore use the DEFAULT policy — the `None` override
an earlier draft of this doc suggested would error. For the bulk n-gram import hot path, set
`durability_policy(GroupCommit)`: batched fsync keeps the WAL append cheap while honoring the
ACK-durability contract. Benchmark import throughput (old in-memory `insert_cas` vs the new durable insert
under Immediate vs GroupCommit) before/after.

## File 1 — src/ngram/vocabulary.rs (the bulk)
- **import (65):** drop `ConcurrentMode, ConcurrentVocabARTrie, ConcurrentVocabStats, LockFreeVocab,
  LockFreeVocabStats`. Keep `PersistentVocabARTrie, SharedVocabARTrie, VocabSyncHandle`.
- **`SharedConcurrentVocab` alias (323):** `Arc<ConcurrentVocabARTrie>` → `Arc<PersistentVocabARTrie>`.
- **`open_or_create_concurrent_vocabulary_lockfree(path)` (255-266):** return `Arc<PersistentVocabARTrie>`;
  body = open/create the trie (DEFAULT durability_policy — the durable insert REJECTS `None`), then
  `Ok(Arc::new(trie))`. (Drop the `ConcurrentVocabARTrie::new_lockfree(trie)` wrap.)
- **`..._with_capacity` (279-294):** same (drop the capacity arg or document it as ignored — the overlay
  DashMap auto-sizes; or expose a capacity ctor in libdictenstein later).
- **`..._with_bloom` (304-320):** `create_with_start_index_and_bloom` is DELETED → use
  `create_with_start_index`; return `Arc::new(trie)`.
- **`create_concurrent_vocabulary_lockfree(vocab: SharedVocabARTrie)` (237-241):** DELETE (test-only; you
  no longer wrap a SharedVocabARTrie — use the path-based factory which returns `Arc<PersistentVocabARTrie>`).
- **`encode_ngram_key_lockfree(words, vocab: &PersistentVocabARTrie) -> String` (527):**
  `vocab.insert_batch(words).expect("vocab insert")` (was infallible `insert_batch_concurrent`).
- **`try_encode_ngram_key_lockfree` (546):** propagate — `vocab.insert_batch(words).map_err(|e|
  VocabularyError::Trie(e.to_string()))?` then encode.
- **`encode_ngram_key_with_lockfree_vocab(words, vocab: &LockFreeVocab)` (566):** retype to
  `&PersistentVocabARTrie`, body identical to `encode_ngram_key_lockfree` (or delete + redirect callers).
- **`with_encoded_ngram_key_lockfree` (615) + `encode_ngram_key_lockfree_bytes` (636):** `vocab:
  &PersistentVocabARTrie`; `vocab.insert_batch(words).expect(...)`.
- **line ~800:** `vocab: &ConcurrentVocabARTrie` → `&PersistentVocabARTrie`.
- **tests (1363-1425):**
  - `concurrent.insert_cas(t)` → `concurrent.insert(t).expect("insert")`.
  - DELETE the `concurrent1.inner().write().checkpoint()` block — ONE `concurrent.checkpoint()` now
    persists everything (the overlay IS the persistent layer).
  - `test_lockfree_vocab_stats`: there is no `lockfree_stats()`/`stats()`/`ConcurrentMode` — assert via
    `concurrent.len() == 3` + `concurrent.next_index() == 4`. Drop the `general_stats.mode ==
    ConcurrentMode::LockFree` assert.

## File 2 — src/dictionary/vocabulary_backed.rs (tests only)
- The 3 test calls `create_concurrent_vocabulary_lockfree(vocabulary.clone())` (129/180/220): replace with
  `open_or_create_concurrent_vocabulary_lockfree(&vocab_path)` (returns `Arc<PersistentVocabARTrie>`); drop
  the now-unused `open_or_create_vocabulary` SharedVocabARTrie there. Production code (uses SharedVocabARTrie
  + `.read()`) is UNCHANGED.

## File 3 — src/sources/google_books/storage.rs
- line 157 `.enable_lockfree()`: DELETE the call (the vocab is always lock-free; `enable_lockfree` is now
  `pub(crate)`). Verify the encode-fn imports/signatures still resolve (they take `&PersistentVocabARTrie`).

## File 4 — src/sources/google_books/sharding/shard.rs
- lines 568, 601 `.enable_lockfree()`: DELETE both calls.

## Gate
`cargo build` + `cargo test` in libgrammstein; spot-check the google_books import path compiles + a small
end-to-end ngram-train run; benchmark bulk-import throughput (the durable-insert perf note above).
