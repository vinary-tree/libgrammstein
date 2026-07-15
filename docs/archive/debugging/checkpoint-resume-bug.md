> ⚠ **Archived — historical record, not current design.**
> A point-in-time report for a specific interrupted-import defect and its fix. The
> vocabulary-checkpoint path it describes (`checkpoint_vocabulary()` / `sync_vocabulary()`) has
> since been replaced by `merge_and_rotate_vocabulary_wal()`. For the current importer design see
> [`../../architecture/memory-optimization.md`](../../architecture/memory-optimization.md) and
> [`../../architecture/google-books-importer.md`](../../architecture/google-books-importer.md).

# Checkpoint Resume Bug: Count Mismatch After Interrupted Import

## Problem Summary

When a Google Books n-gram import is interrupted (Ctrl+C) and then resumed:
- The resumed model has **fewer unique entries** (3,462,740 vs 5,800,990)
- Many entries have **doubled counts** (e.g., "A'" has 149380 vs 74690 - exactly 2x)
- Some entries are **missing entirely** (2,338,250 entries lost)

## Root Cause

The `save_checkpoint()` method was only checkpointing the **n-gram trie and checkpoint metadata**, but NOT the **vocabulary trie**.

During import:
1. New vocabulary entries are written to the vocabulary's WAL (Write-Ahead Log)
2. N-grams using those vocabulary indices are written to shards
3. Checkpoint is called, which flushes the checkpoint trie but NOT the vocabulary
4. On interruption, the vocabulary WAL contains 140+ MB of unwritten data

On resume:
1. The vocabulary artrie is opened with only the checkpointed data (tiny, 262KB)
2. The 140MB vocabulary WAL is **not replayed** because the artrie was opened fresh
3. New vocabulary indices are assigned starting from the last checkpointed index
4. The same words get **different indices** than in the interrupted run
5. N-grams written with old indices are now orphaned
6. N-grams written with new indices are duplicates (same words, different key encoding)

## Evidence

From `dump_checkpoint` tool output:

```
=== Interrupted Backup ===
english.vocab.wal - 142.4 MB (LARGE - NOT CHECKPOINTED!)
english.vocab.artrie - 262 KB (tiny)
Vocabulary entries: 3,462,742

=== Completed Backup ===
english.vocab.wal - 64 B (empty - properly checkpointed)
english.vocab.artrie - 1 GB (full size)
Vocabulary entries: 5,800,992
```

The 142.4 MB WAL file proves the vocabulary was never checkpointed during the interrupted run.

## Fix Applied

Modified `save_checkpoint()` in `src/sources/google_books/importer.rs` to checkpoint the vocabulary BEFORE saving the checkpoint metadata:

```rust
pub fn save_checkpoint(&mut self) -> Result<(), ImportError> {
    // ... sync atomic counters ...

    // CRITICAL: Checkpoint vocabulary FIRST to ensure vocabulary indices are
    // persisted before the checkpoint marks prefixes as completed.
    self.storage.sync_vocabulary().map_err(|e| {
        ImportError::Trie(format!("Failed to sync vocabulary: {}", e))
    })?;
    self.storage.checkpoint_vocabulary().map_err(|e| {
        ImportError::Trie(format!("Failed to checkpoint vocabulary: {}", e))
    })?;

    // ... rest of checkpoint save ...
}
```

## Verification Plan

To verify the fix works:

1. **Fresh import to completion** - establish baseline
2. **Import, interrupt at ~50%** of files
3. **Resume import** to completion
4. **Compare** resumed model with baseline using `compare_artries`
5. **Verify** counts match exactly

## Related Files

| File | Description |
|------|-------------|
| `src/sources/google_books/importer.rs` | Main importer with `save_checkpoint()` fix |
| `src/sources/google_books/storage.rs` | `sync_vocabulary()` and `checkpoint_vocabulary()` methods |
| `src/ngram/vocabulary.rs` | `SharedVocabulary` with WAL-backed storage |
| `src/bin/dump_checkpoint.rs` | Diagnostic tool for inspecting checkpoint state |

## Diagnostic Tool

A new diagnostic binary `dump_checkpoint` was created to inspect checkpoint state:

```bash
cargo run --release --bin dump_checkpoint --features cli,google-books -- \
    --dir bak-sharded-interrupted/ \
    --dir bak-sharded-completed/ \
    --dir .
```

This shows:
- WAL file sizes and whether they're checkpointed
- Checkpoint trie contents (prefix states, n-gram counts)
- Vocabulary entry counts
- Comparison across multiple directories

## Lessons Learned

1. **All persistent state must be checkpointed together** - the vocabulary, shards, and checkpoint metadata must be in a consistent state
2. **Large WAL files are a red flag** - a WAL > 1MB suggests data hasn't been checkpointed
3. **Add diagnostic tooling early** - the `dump_checkpoint` tool immediately revealed the issue
4. **Vocabulary indices are structural** - losing vocabulary mappings corrupts all n-gram data
