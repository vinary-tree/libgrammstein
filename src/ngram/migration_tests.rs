//! Acceptance tests for the MKN byte-native (term-id / LEB128) migration.
//!
//! These live inside the crate (not `tests/`) so they run under the required
//! `cargo test --lib` gates. Tests that need the portable serialization format
//! are individually gated on `serde-extras`.
//!
//! There is exactly one live n-gram store — [`TermIdStore`]; no `'|'`-delimited
//! model is ever instantiated. The legacy `'|'` wire format appears only as a
//! test *fixture* fed to the one-shot legacy reader (`from_portable`).

use super::store::{MutableNgramStore, TermIdStore};
use super::vocabulary::create_vocabulary;
use super::{NgramEntry, NgramModel, TrainerBuilder};
use crate::corpus::PlaintextReader;
use libdictenstein::dynamic_dawg::DynamicDawg;
use std::io::Write;
use tempfile::TempDir;

/// In-memory byte-native (term-id) model — the sole live model type.
type TermIdModel = NgramModel<TermIdStore<DynamicDawg<NgramEntry>>>;

/// A corpus rich enough that the MKN discount statistics `n1..n4` are all
/// populated and multiple continuation contexts exist.
const CORPUS: &str = "the quick brown fox jumps over the lazy dog\n\
                      the quick brown fox runs past the lazy cat\n\
                      the slow green turtle walks over the lazy dog\n\
                      a quick brown fox jumps over a lazy dog\n\
                      the quick brown fox jumps over the lazy dog\n\
                      the lazy dog sleeps while the quick fox runs\n\
                      the quick brown fox jumps over the lazy dog\n";

/// The (word, context) probes exercised: seen high-order, seen bigram, unigram,
/// seen-context-unseen-word (backoff), OOV word, unseen context, and the
/// empty-continuation edge case that previously triggered `-inf`.
const PROBES: &[(&str, &[&str])] = &[
    ("fox", &["quick", "brown"]),
    ("brown", &["quick"]),
    ("the", &[]),
    ("dog", &["lazy"]),
    ("cat", &["lazy"]),
    ("turtle", &["green"]),
    ("zzz_oov", &["the"]),
    ("fox", &["nonexistent", "context"]),
    ("", &["the"]),
];

fn write_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
    let path = dir.join("corpus.txt");
    let mut file = std::fs::File::create(&path).expect("create corpus");
    file.write_all(content.as_bytes()).expect("write corpus");
    path
}

fn train_termid(corpus: &str, order: usize) -> (TempDir, TermIdModel) {
    let dir = TempDir::new().expect("temp dir");
    let path = write_corpus(dir.path(), corpus);
    let reader = PlaintextReader::from_file(&path).expect("open corpus");
    let model = TrainerBuilder::new(DynamicDawg::<NgramEntry>::new())
        .order(order)
        .train(reader)
        .expect("train term-id model");
    (dir, model)
}

/// Assert two log-probabilities agree within `1e-12`, treating matching infinities
/// as equal.
fn assert_logprob_eq(a: f64, b: f64, ctx: &str) {
    let ok = if a.is_infinite() && b.is_infinite() {
        a.signum() == b.signum()
    } else {
        (a - b).abs() < 1e-12
    };
    assert!(ok, "log_prob mismatch for {ctx}: {a} vs {b}");
}

/// (7b) Score parity / encoding-invariance: two independently-trained models over
/// the same corpus assign **different** term-ids (independent ephemeral
/// vocabularies) yet score every probe identically — the "byte-identical MKN
/// scores modulo encoding" invariant, with no delimited representation anywhere.
#[test]
fn score_invariant_to_termid_encoding() {
    let (_d1, a) = train_termid(CORPUS, 3);
    let (_d2, b) = train_termid(CORPUS, 3);

    for (word, context) in PROBES {
        assert_logprob_eq(
            a.log_prob(word, context),
            b.log_prob(word, context),
            &format!("({word:?}, {context:?})"),
        );
    }

    let sentence = [
        "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    ];
    assert_logprob_eq(
        a.sentence_log_prob(&sentence),
        b.sentence_log_prob(&sentence),
        "sentence_log_prob",
    );
}

/// (7a, in-memory half) A token literally containing `'|'` trains and scores
/// through the term-id path — impossible under the old `'|'`-joined key scheme.
#[test]
fn delimiter_collision_trains_and_scores() {
    let (_dir, model) = train_termid("foo|bar baz qux foo|bar baz qux foo|bar baz qux\n", 2);

    // The two-token n-gram is stored intact (not corrupted into three tokens).
    assert!(
        model.count(&["foo|bar", "baz"]) >= 3,
        "count for ['foo|bar','baz'] should be >= 3, got {}",
        model.count(&["foo|bar", "baz"])
    );
    assert!(model.log_prob("baz", &["foo|bar"]).is_finite());

    // It iterates back as exactly the two-token sequence.
    let found = model
        .iter_ngrams()
        .any(|(w, _)| w == vec!["foo|bar".to_string(), "baz".to_string()]);
    assert!(
        found,
        "['foo|bar','baz'] must be recoverable via iter_ngrams"
    );
}

/// (7c) `U64NgramView` composes directly over a disk-backed
/// `TermIdStore<Arc<PersistentARTrie<NgramEntry>>>`: the model's byte trie is a
/// `Unit = u64` term-id dictionary whose fuzzy queries recover the stored
/// [`NgramEntry`] values.
#[test]
fn u64_view_over_persistent_termid_store() {
    use super::U64NgramView;
    use libdictenstein::persistent_artrie::PersistentARTrie;
    use libdictenstein::{Dictionary, DictionaryNode, MappedDictionaryNode};
    use liblevenshtein::prelude::{Algorithm, Transducer};
    use std::sync::Arc;

    let dir = TempDir::new().expect("temp dir");
    let vocab = create_vocabulary(&dir.path().join("vocab")).expect("create vocab");
    let backend: Arc<PersistentARTrie<NgramEntry>> = Arc::new(
        PersistentARTrie::<NgramEntry>::create(dir.path().join("ngrams.artrie"))
            .expect("create trie"),
    );
    let store = TermIdStore::new(backend, vocab, 3);

    store.bump(&["the", "quick", "fox"], 5);
    store.bump(&["the", "quick", "dog"], 2);
    store.bump(&["the", "slow", "fox"], 1);

    let id = |word: &str| -> u64 {
        store
            .vocabulary()
            .as_ref()
            .get_index(word)
            .expect("word in vocabulary")
    };
    let seq = |words: &[&str]| -> Vec<u64> { words.iter().map(|w| id(w)).collect() };

    let view = U64NgramView::new(store.backend().clone());

    // Forward walk recovers the stored NgramEntry value.
    let target = seq(&["the", "quick", "fox"]);
    let mut node = view.root();
    for term_id in &target {
        node = node
            .transition(*term_id)
            .expect("term-id transition exists in the view");
    }
    assert!(node.is_final(), "the [the quick fox] path must be final");
    assert_eq!(
        node.value().map(|e| e.count()),
        Some(5),
        "the view recovers the stored entry count"
    );

    // A fuzzy (term-id-level) transducer over the view recovers the exact
    // sequence at distance 0, and the substitution neighbour at distance 1.
    let transducer = Transducer::new(view, Algorithm::Standard);
    let exact: Vec<Vec<u64>> = transducer.query_units(&target, 0).collect();
    assert_eq!(
        exact,
        vec![target.clone()],
        "distance-0 recovers exactly the sequence"
    );

    let neighbours: Vec<Vec<u64>> = transducer.query_units(&target, 1).collect();
    assert!(
        neighbours.contains(&seq(&["the", "quick", "dog"])),
        "distance-1 recovers the [the quick dog] substitution neighbour, got {neighbours:?}"
    );
}

// ── Portable serialization (serde-extras) ─────────────────────────────────────

/// Build a historical `'|'`-joined [`KeyEncoding::LegacyPipe`] portable model
/// from a term-id model's decoded n-grams — a **legacy wire-format fixture** used
/// solely to exercise the one-shot legacy reader. The `'|'` here forms test
/// input; it is never a live representation.
#[cfg(feature = "serde-extras")]
fn legacy_pipe_fixture(model: &TermIdModel) -> super::PortableNgramModel {
    use super::{KeyEncoding, NgramEntrySnapshot, PortableNgramModel};

    let entries: Vec<(String, NgramEntrySnapshot)> = model
        .iter_ngrams()
        .map(|(words, entry)| (words.join("|"), NgramEntrySnapshot::from(&entry)))
        .collect();

    PortableNgramModel {
        entries,
        entries_bytes: Vec::new(),
        max_order: model.order(),
        vocab_size: model.vocab_size(),
        total_count: model.total_count(),
        smoothing: model.smoothing().clone(),
        vocabulary: None,
        key_encoding: KeyEncoding::LegacyPipe,
        vocab_fingerprint: None,
    }
}

/// (7d) TermIdBytes save/reload parity: the portable file is tagged
/// [`KeyEncoding::TermIdBytes`], carries byte-keyed entries + a fingerprinted
/// vocabulary, and reloads to byte-identical scores.
#[cfg(feature = "serde-extras")]
#[test]
fn termid_portable_roundtrip_parity() {
    use super::{KeyEncoding, PortableNgramModel};

    let (_dir, model) = train_termid(CORPUS, 3);
    let file = tempfile::NamedTempFile::new().expect("temp file");
    model.save_portable(file.path()).expect("save portable");

    // Inspect the on-disk shape.
    let portable: PortableNgramModel = {
        let reader = std::io::BufReader::new(std::fs::File::open(file.path()).expect("open"));
        bincode::deserialize_from(reader).expect("deserialize portable")
    };
    assert_eq!(portable.key_encoding, KeyEncoding::TermIdBytes);
    assert!(!portable.entries_bytes.is_empty(), "byte entries present");
    assert!(portable.entries.is_empty(), "no legacy string entries");
    assert!(portable.vocabulary.is_some(), "vocabulary embedded");
    assert!(portable.vocab_fingerprint.is_some(), "fingerprint stamped");

    // Reload and compare scores.
    let loaded = TermIdModel::load_portable(file.path(), DynamicDawg::<NgramEntry>::new)
        .expect("load portable");
    assert_eq!(loaded.ngram_count(), model.ngram_count());
    for (word, context) in PROBES {
        assert_logprob_eq(
            model.log_prob(word, context),
            loaded.log_prob(word, context),
            &format!("reload ({word:?}, {context:?})"),
        );
    }
}

/// (7a, full) Delimiter-collision round-trips through save/load/iterate.
#[cfg(feature = "serde-extras")]
#[test]
fn delimiter_collision_roundtrips_through_portable() {
    let (_dir, model) = train_termid("foo|bar baz foo|bar baz foo|bar baz\n", 2);
    let file = tempfile::NamedTempFile::new().expect("temp file");
    model.save_portable(file.path()).expect("save");

    let loaded =
        TermIdModel::load_portable(file.path(), DynamicDawg::<NgramEntry>::new).expect("load");
    assert_eq!(
        loaded.count(&["foo|bar", "baz"]),
        model.count(&["foo|bar", "baz"])
    );
    assert!(loaded
        .iter_ngrams()
        .any(|(w, _)| w == vec!["foo|bar".to_string(), "baz".to_string()]));
}

/// (7d, legacy path) A `LegacyPipe` portable fixture is read exactly once by the
/// one-shot legacy reader into a term-id model whose scores match the source —
/// the realistic "old `'|'` data still loads" guarantee.
#[cfg(feature = "serde-extras")]
#[test]
fn legacy_pipe_reader_roundtrips() {
    let (_dir, model) = train_termid(CORPUS, 3);
    let fixture = legacy_pipe_fixture(&model);

    let loaded = TermIdModel::from_portable(fixture, DynamicDawg::<NgramEntry>::new)
        .expect("legacy reader load");
    assert_eq!(loaded.ngram_count(), model.ngram_count());
    for (word, context) in PROBES {
        assert_logprob_eq(
            model.log_prob(word, context),
            loaded.log_prob(word, context),
            &format!("legacy reader ({word:?}, {context:?})"),
        );
    }
}

/// (7d, forward-compat) A self-describing (JSON) portable value that omits the
/// migration's new fields deserializes with the correct `#[serde(default)]`
/// values. (bincode is not self-describing, so this documents the default
/// semantics that self-describing formats rely on.)
#[cfg(feature = "serde-extras")]
#[test]
fn portable_serde_defaults_for_missing_new_fields() {
    use super::{KeyEncoding, PortableNgramModel};

    let json = r#"{
        "entries": [["the", {"count": 3, "continuation_count": 0, "unique_continuations": 1}]],
        "max_order": 3,
        "vocab_size": 1,
        "total_count": 3,
        "smoothing": {"d1": 0.75, "d2": 0.85, "d3_plus": 0.95}
    }"#;

    let portable: PortableNgramModel = serde_json::from_str(json).expect("json deserialize");
    assert_eq!(portable.key_encoding, KeyEncoding::LegacyPipe);
    assert!(portable.entries_bytes.is_empty());
    assert!(portable.vocab_fingerprint.is_none());
    assert!(portable.vocabulary.is_none());
    assert_eq!(portable.entries.len(), 1);
}

/// The vocabulary fingerprint guard rejects a `(model, vocabulary)` pair whose
/// vocabularies disagree, rather than silently mis-decoding term-id keys.
#[cfg(feature = "serde-extras")]
#[test]
fn fingerprint_mismatch_is_rejected() {
    let (_dir, model) = train_termid(CORPUS, 3);
    let file = tempfile::NamedTempFile::new().expect("temp file");
    model.save_portable(file.path()).expect("save");

    let portable: super::PortableNgramModel = {
        let reader = std::io::BufReader::new(std::fs::File::open(file.path()).expect("open"));
        bincode::deserialize_from(reader).expect("deserialize")
    };

    // A different vocabulary (different words) must fail the fingerprint check.
    let other_dir = TempDir::new().expect("temp dir");
    let wrong_vocab = create_vocabulary(&other_dir.path().join("vocab")).expect("vocab");
    wrong_vocab.as_ref().insert("totally").expect("insert");
    wrong_vocab.as_ref().insert("different").expect("insert");

    let result = TermIdModel::from_portable_with_vocabulary(
        portable,
        DynamicDawg::<NgramEntry>::new(),
        wrong_vocab,
    );
    assert!(
        matches!(result, Err(crate::Error::Model(_))),
        "a mismatched vocabulary must be rejected with a Model error"
    );
}
