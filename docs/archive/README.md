# Documentation Archive

This directory holds **historical documentation** — point-in-time reports, completed migration
work-orders, dated evaluations, and superseded designs. These files are kept for provenance and
context, but they **do not describe the current system**. Each carries an *Archived* banner at the
top pointing to its live successor (where one exists).

> Looking for current documentation? Start at [`../README.md`](../README.md).

## Why these were archived

Archived documents fall into two kinds:

- **Point-in-time records** — a bug report, an experiment ledger, a sanitizer run, a migration
  checklist. Each captured the state of the project on a specific date. They were never meant to be
  living design docs, and they go stale by nature (line numbers drift, counts change, fixes land).
- **Superseded designs** — documentation of an approach or API that no longer exists in the code.
  Keeping it in the live tree would actively mislead a reader.

Neither kind is deleted: the material remains valuable for understanding *how* and *why* the system
reached its current shape, and it stays reachable from git history and from here.

## Contents

| Archived document | Kind | Why archived | Current successor |
|---|---|---|---|
| [`debugging/checkpoint-resume-bug.md`](debugging/checkpoint-resume-bug.md) | bug report | A specific interrupted-import defect and its fix. Describes the retired `checkpoint_vocabulary()` / `sync_vocabulary()` path (now `merge_and_rotate_vocabulary_wal()`). | [`../architecture/memory-optimization.md`](../architecture/memory-optimization.md), [`../architecture/google-books-importer.md`](../architecture/google-books-importer.md) |
| [`sanitizer-evaluation-results.md`](sanitizer-evaluation-results.md) | dated evaluation | A 2026-01-19 ASan/TSan/Miri run across four repositories; results and test counts are a stale snapshot. | — (provenance only) |
| [`vocab-single-lockfree-migration.md`](vocab-single-lockfree-migration.md) | migration work-order | A completed, line-numbered checklist for collapsing the vocabulary onto one lock-free implementation (`PersistentVocabARTrie`). | [`../architecture/data-flow.md`](../architecture/data-flow.md) |
| [`experiments/formal-proof-optimizations.md`](experiments/formal-proof-optimizations.md) | experiment ledger | A scientific ledger of finished A/B optimizations (ACCEPTED/REJECTED + commit hashes). | [`../../formal/README.md`](../../formal/README.md) |
| [`integration/dictionary-wfst.md`](integration/dictionary-wfst.md) | superseded design | Documents a `grammstein dictionary export-fst` / OpenFST-text workflow that does not exist in `src/`. | [`../integration/lling-llang/overview.md`](../integration/lling-llang/overview.md), [`../integration/liblevenshtein/overview.md`](../integration/liblevenshtein/overview.md) |

## Policy

- **Do not update archived files** to reflect current behavior — that is what the successor docs are
  for. Archived files should change only to correct a factual statement *about the historical event*
  they record, or to fix their *Archived* banner.
- **Do not link to archived files from live documentation** except through this index or an explicit
  "history" reference. Live docs should link to the successor, not the archive.
- When a live doc is retired in the future, move it here (preserving its subpath), prepend the
  *Archived* banner with a successor link, and add a row to the table above.
