//! RAII ownership for *ephemeral* vocabulary directories.
//!
//! Loading a portable model (`NgramModel::from_portable` / `load_portable`,
//! `HybridLanguageModel::load_portable`) and training without an explicit
//! vocabulary (`VocabularyMode::Ephemeral`) both materialize a
//! [`SharedVocabARTrie`](super::vocabulary::SharedVocabARTrie) in a fresh
//! directory under [`std::env::temp_dir`]. That directory backs the vocabulary's
//! memory map and is written to again by the trie's own `Drop` (a final
//! checkpoint), so it must live *exactly* as long as the vocabulary — which lives
//! as long as the owning model's [`TermIdStore`](super::store::TermIdStore).
//!
//! Nothing in libdictenstein removes that directory (the vocabulary trie is an
//! `Arc<PersistentVocabARTrie>` whose fields are private to that crate), so the
//! guard has to be owned on the libgrammstein side. [`VocabTempDir`] is that
//! owner: a [`TermIdStore`](super::store::TermIdStore) that was built for an
//! ephemeral vocabulary holds one (wrapped in an `Arc` so store clones share a
//! single owner), and the directory is unlinked when the last clone drops.
//!
//! Persistent, caller-named vocabularies (`create_vocabulary`,
//! `open_or_create_vocabulary`, and the caller-path loaders `from_portable_at` /
//! `load_portable_at`) are **not** ephemeral: they carry no guard, so their
//! directories are never deleted.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Owns an ephemeral vocabulary directory and removes it (recursively) when
/// dropped.
///
/// # Drop ordering
///
/// The backing directory is written to by
/// `PersistentVocabARTrie`'s `Drop` (a final checkpoint) *and* mapped for the
/// vocabulary's whole lifetime, so the vocabulary must be dropped before this
/// guard. [`TermIdStore`](super::store::TermIdStore) guarantees that by declaring
/// its `vocab_tempdir` field after `vocab`; Rust drops struct fields in
/// declaration order, so the vocabulary trie (and its checkpoint-on-drop) runs
/// first, then this guard removes the now-idle directory.
///
/// Removal is best-effort: a failure to unlink (e.g. the directory was already
/// removed) is ignored rather than panicking in `Drop`.
#[derive(Debug)]
pub(crate) struct VocabTempDir {
    path: PathBuf,
}

impl Drop for VocabTempDir {
    fn drop(&mut self) {
        // Best-effort: the vocabulary (and its checkpoint-on-drop) has already run
        // by construction (see the type-level "Drop ordering" note), so the
        // directory is idle. Ignore errors — there is no recovery in `Drop`, and a
        // missing directory is the success condition anyway.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Create a fresh, uniquely-named ephemeral vocabulary directory under
/// [`std::env::temp_dir`] and return it together with the [`VocabTempDir`] guard
/// that will remove it.
///
/// `infix` distinguishes the two ephemeral call sites in the directory name:
/// `"vocab"` for the portable-load path (`libgrammstein-vocab-{pid}-{seq}`) and
/// `"train-vocab"` for the ephemeral-training path
/// (`libgrammstein-train-vocab-{pid}-{seq}`). The process id plus a
/// process-monotonic sequence keep concurrent directories from colliding.
///
/// The returned `PathBuf` is the directory itself; callers materialize the
/// vocabulary at `dir.join("vocabulary")`.
pub(crate) fn create_guarded_temp_vocab_dir(
    infix: &str,
) -> std::io::Result<(PathBuf, VocabTempDir)> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "libgrammstein-{infix}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&dir)?;
    let guard = VocabTempDir { path: dir.clone() };
    Ok((dir, guard))
}
