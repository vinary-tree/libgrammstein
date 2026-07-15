//! Regression + behavioral test for ephemeral-vocabulary directory cleanup.
//!
//! Loading a portable model and training with the default `Ephemeral` vocabulary
//! both materialize a `SharedVocabARTrie` in a fresh directory under
//! `std::env::temp_dir()`. Historically neither directory was ever removed, so a
//! long-running process (e.g. a service that loads a model per query)
//! accumulated `libgrammstein-vocab-*` / `libgrammstein-train-vocab-*`
//! directories without bound. These tests pin the RAII fix: the directory now
//! lives exactly as long as the owning model and is unlinked when the model
//! drops. They also cover the caller-path loaders (`from_portable_at` /
//! `load_portable_at`), whose vocabularies are persistent and caller-owned — no
//! temporary directory is created and nothing is auto-removed.
//!
//! Every temp-vocabulary directory created here is isolated into a private
//! `$TMPDIR`, so the directory-count assertions are immune to other test
//! binaries running concurrently in the shared system temp directory.

#![cfg(feature = "serde-extras")]

use std::path::{Path, PathBuf};

use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel};
use libgrammstein::ngram::store::TermIdStore;
use libgrammstein::ngram::{InMemoryTermIdModel, NgramEntry, TrainerBuilder};
use tempfile::TempDir;

/// In-memory byte-native store used throughout these tests.
type TestStore = TermIdStore<DynamicDawg<NgramEntry>>;

const CORPUS: &str = "the quick brown fox the quick brown dog the lazy fox \
                      the quick brown fox the quick brown dog the lazy fox \
                      the quick brown fox the quick brown dog the lazy fox";

/// Count directories under `root` whose name marks an ephemeral libgrammstein
/// vocabulary — both the portable-load prefix (`libgrammstein-vocab-`) and the
/// ephemeral-training prefix (`libgrammstein-train-vocab-`).
///
/// Both prefixes are matched explicitly: a naïve `libgrammstein-vocab*` glob
/// would miss the training directories, whose `-train-` infix sits between the
/// stem and `vocab`.
fn count_ephemeral_vocab_dirs(root: &Path) -> usize {
    std::fs::read_dir(root)
        .expect("read isolated temp root")
        .filter_map(Result::ok)
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("libgrammstein-vocab-")
                || name.starts_with("libgrammstein-train-vocab-")
        })
        .count()
}

fn write_corpus(dir: &Path) -> PathBuf {
    let path = dir.join("corpus.txt");
    std::fs::write(&path, CORPUS).expect("write corpus");
    path
}

/// Train a model with the default `Ephemeral` vocabulary (leak site #2).
fn train_ephemeral(corpus: &Path) -> InMemoryTermIdModel {
    let reader = PlaintextReader::from_file(corpus).expect("open corpus");
    TrainerBuilder::new(DynamicDawg::<NgramEntry>::new())
        .order(3)
        .train(reader)
        .expect("train ephemeral term-id model")
}

/// Build a hybrid model whose n-gram half was trained with an ephemeral
/// vocabulary; used to exercise the hybrid portable loaders.
fn build_hybrid(corpus: &Path) -> HybridLanguageModel<TestStore> {
    let ngram = train_ephemeral(corpus);
    let reader = PlaintextReader::from_file(corpus).expect("open corpus");
    let embedding = EmbeddingTrainerBuilder::new()
        .dim(10)
        .window_size(2)
        .min_count(1)
        .epochs(2)
        .train(reader)
        .expect("train embedding");
    HybridLanguageModel::new(ngram, embedding, HybridConfig::default())
}

/// The whole invariant lives in one `#[test]` so this binary has no in-process
/// concurrency: it sets `$TMPDIR` process-wide for isolation, which is sound only
/// when a single thread writes it once, before any reader.
#[test]
fn ephemeral_vocab_directories_are_cleaned_up() {
    // Isolate this test's temp-vocabulary directories into a private root so the
    // count assertions cannot be perturbed by other test binaries. `temp_dir()`
    // reads `$TMPDIR`; set it once, before any model (or Rayon worker) reads it,
    // and never mutate it again — concurrent reads of an unchanging env var are
    // sound. This is the sole `#[test]` in its binary, so there is no in-process
    // writer race either.
    let iso = TempDir::new().expect("create isolated temp root");
    std::env::set_var("TMPDIR", iso.path());
    assert_eq!(
        std::env::temp_dir(),
        iso.path(),
        "TMPDIR isolation must redirect temp_dir()"
    );
    let root = iso.path();
    let corpus = write_corpus(root);

    // ---- Leak site #2: ephemeral training ---------------------------------
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "no ephemeral vocab dirs exist before any model is built"
    );
    let trained = train_ephemeral(&corpus);
    assert!(
        count_ephemeral_vocab_dirs(root) >= 1,
        "a live ephemerally-trained model must own its backing temp dir"
    );
    assert!(
        trained.log_prob("fox", &["quick", "brown"]).is_finite(),
        "trained model must score"
    );
    drop(trained);
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "dropping the trained model must remove its ephemeral vocab dir"
    );

    // ---- Leak site #1: portable load --------------------------------------
    let portable_path = root.join("model.bin");
    train_ephemeral(&corpus)
        .save_portable(&portable_path)
        .expect("save portable model");
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "the throwaway model built only to save must not leak its vocab dir"
    );

    let loaded = InMemoryTermIdModel::load_portable(&portable_path, DynamicDawg::<NgramEntry>::new)
        .expect("load portable model");
    assert!(
        count_ephemeral_vocab_dirs(root) >= 1,
        "a live portable-loaded model must own its rebuilt temp vocab dir"
    );
    assert!(
        loaded.log_prob("fox", &["quick", "brown"]).is_finite(),
        "loaded model must score"
    );
    drop(loaded);
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "dropping the loaded model must remove its rebuilt temp vocab dir"
    );

    // ---- Bounded accumulation over many load/train cycles -----------------
    const CYCLES: usize = 8;
    for _ in 0..CYCLES {
        drop(train_ephemeral(&corpus));
        drop(
            InMemoryTermIdModel::load_portable(&portable_path, DynamicDawg::<NgramEntry>::new)
                .expect("load portable model"),
        );
    }
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "repeated load/train cycles must leave no residual temp vocab dirs"
    );

    // ---- Option 2 (n-gram): from_portable_at into a caller-owned dir -------
    let portable = train_ephemeral(&corpus).to_portable();
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "building the source model for to_portable() must not leak"
    );

    let caller = TempDir::new().expect("caller-owned vocab root");
    let caller_vocab = caller.path().join("vocabulary");
    let before = count_ephemeral_vocab_dirs(root);
    let at_model = InMemoryTermIdModel::from_portable_at(
        portable,
        DynamicDawg::<NgramEntry>::new,
        &caller_vocab,
    )
    .expect("load model at caller path");
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        before,
        "from_portable_at must not create any ephemeral temp vocab dir"
    );
    assert!(
        caller_vocab.exists(),
        "from_portable_at must materialize the vocabulary at the caller path"
    );
    assert!(
        at_model.log_prob("fox", &["quick", "brown"]).is_finite(),
        "caller-path model must score"
    );
    drop(at_model);
    assert!(
        caller_vocab.exists(),
        "a caller-owned vocabulary must NOT be removed when the model drops"
    );

    // ---- Option 2 (hybrid): load_portable vs load_portable_at -------------
    let hybrid_path = root.join("hybrid.bin");
    build_hybrid(&corpus)
        .save_portable(&hybrid_path)
        .expect("save hybrid portable");
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "building and saving the hybrid must not leak a temp vocab dir"
    );

    // Temp-path hybrid load creates and then cleans a temp vocab dir.
    let hybrid_tmp = HybridLanguageModel::<TestStore>::load_portable(
        &hybrid_path,
        DynamicDawg::<NgramEntry>::new,
    )
    .expect("hybrid load_portable");
    assert!(
        count_ephemeral_vocab_dirs(root) >= 1,
        "a live hybrid loaded via load_portable must own its temp vocab dir"
    );
    assert!(hybrid_tmp.score("fox", &["the", "quick"]).is_finite());
    drop(hybrid_tmp);
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        0,
        "dropping a hybrid loaded via load_portable must remove its temp vocab dir"
    );

    // Caller-path hybrid load creates no temp dir and leaves the vocab in place.
    let hybrid_caller = TempDir::new().expect("caller-owned hybrid vocab root");
    let hybrid_vocab = hybrid_caller.path().join("vocabulary");
    let before = count_ephemeral_vocab_dirs(root);
    let hybrid_at = HybridLanguageModel::<TestStore>::load_portable_at(
        &hybrid_path,
        DynamicDawg::<NgramEntry>::new,
        &hybrid_vocab,
    )
    .expect("hybrid load_portable_at");
    assert_eq!(
        count_ephemeral_vocab_dirs(root),
        before,
        "hybrid load_portable_at must not create any temp vocab dir"
    );
    assert!(
        hybrid_vocab.exists(),
        "hybrid load_portable_at must materialize the caller vocabulary"
    );
    assert!(hybrid_at.score("fox", &["the", "quick"]).is_finite());
    drop(hybrid_at);
    assert!(
        hybrid_vocab.exists(),
        "a hybrid caller-owned vocabulary must survive the model drop"
    );
}
