//! Property-based tests for the `T_lex ∘ T_gram` grammar corrector.
//!
//! These complement the example-based unit tests in
//! `src/integration/grammar_corrector.rs` and the `u64`-view faithfulness
//! properties in `src/ngram/u64_view.rs`. The headline property is **`T_gram`
//! soundness + completeness**: `grammar_neighbors` must return *exactly* the stored
//! term-id sequences within word-edit distance `k` of a query — verified
//! differentially against a brute-force edit-distance reference over the stored set.
//!
//! Run with: `cargo test --features lling-llang-integration --test grammar_corrector_proptest`.

#![cfg(feature = "lling-llang-integration")]

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::integration::{GrammarCorrector, GrammarCorrectorConfig};
use libgrammstein::ngram::vocabulary::{create_vocabulary, encode_varint, SharedVocabARTrie};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use tempfile::TempDir;

/// Standard (Levenshtein) word-level edit distance between two term-id sequences —
/// insertion / deletion / substitution at unit cost. Matches [`Algorithm::Standard`].
fn word_edit_distance(a: &[u64], b: &[u64]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, &ai) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &bj) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ai != bj);
            curr[j + 1] = substitution.min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Build a byte-carrier count store keyed by concatenated LEB128 varints.
fn store_from_sequences(sequences: &[Vec<u64>]) -> DynamicDawg<u64> {
    let store = DynamicDawg::<u64>::new();
    for sequence in sequences {
        let mut key = Vec::new();
        for &id in sequence {
            encode_varint(id, &mut key);
        }
        store.insert_bytes_with_value(&key, 1);
    }
    store
}

/// A shared corpus fixture, built once (persistent vocab creation is not free).
struct Fixture {
    _dir: TempDir,
    vocab: SharedVocabARTrie,
    store: DynamicDawg<u64>,
    order: usize,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let dir = TempDir::new().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");
        let order = 3usize;
        let sentences = [
            "the quick brown fox",
            "the quick brown fox",
            "the quick brown dog",
            "the lazy brown fox",
            "a slow green turtle swims",
        ];
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        for sentence in sentences {
            let mut words: Vec<&str> = vec!["<s>"; order - 1];
            words.extend(sentence.split_whitespace());
            words.push("</s>");
            let ids: Vec<u64> = words
                .iter()
                .map(|w| vocab.as_ref().insert(w).expect("insert"))
                .collect();
            for width in 1..=order {
                for window in ids.windows(width) {
                    *counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
        }
        let store = DynamicDawg::<u64>::new();
        for (sequence, count) in &counts {
            let mut key = Vec::new();
            for &id in sequence {
                encode_varint(id, &mut key);
            }
            store.insert_bytes_with_value(&key, *count);
        }
        Fixture {
            _dir: dir,
            vocab,
            store,
            order,
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// **`T_gram` soundness + completeness.** Over an arbitrary set of stored term-id
    /// sequences, an arbitrary query, and an arbitrary radius `k`, `grammar_neighbors`
    /// returns exactly the stored sequences within word-edit distance `k` — the set a
    /// brute-force Levenshtein scan produces. Soundness: nothing farther than `k`.
    /// Completeness: nothing within `k` is missed.
    #[test]
    fn grammar_neighbors_are_sound_and_complete(
        raw in prop::collection::vec(prop::collection::vec(1u64..=16u64, 1..5usize), 1..14usize),
        query in prop::collection::vec(1u64..=16u64, 1..5usize),
        k in 0usize..=2usize,
    ) {
        let stored: BTreeSet<Vec<u64>> = raw.iter().cloned().collect();
        let store = store_from_sequences(&raw);

        // grammar_neighbors does not consult the vocabulary; reuse the shared one.
        let config = GrammarCorrectorConfig {
            algorithm: Algorithm::Standard,
            ..GrammarCorrectorConfig::default()
        };
        let corrector =
            GrammarCorrector::from_store(fixture().vocab.clone(), store, 5, config);

        let got: BTreeSet<Vec<u64>> = corrector
            .grammar_neighbors(&query, k)
            .into_iter()
            .map(|neighbor| neighbor.ids)
            .collect();

        let expected: BTreeSet<Vec<u64>> = stored
            .iter()
            .filter(|sequence| word_edit_distance(sequence, &query) <= k)
            .cloned()
            .collect();

        prop_assert_eq!(got, expected);
    }

    /// **`T_gram` frequency fidelity.** Every neighbor's reported frequency equals the
    /// count actually stored for that sequence (here every inserted sequence has count
    /// `1`), and its reported distance never exceeds the query radius.
    #[test]
    fn grammar_neighbor_frequencies_and_distances_are_faithful(
        raw in prop::collection::vec(prop::collection::vec(1u64..=12u64, 1..4usize), 1..10usize),
        query in prop::collection::vec(1u64..=12u64, 1..4usize),
        k in 0usize..=2usize,
    ) {
        let store = store_from_sequences(&raw);
        let config = GrammarCorrectorConfig {
            algorithm: Algorithm::Standard,
            ..GrammarCorrectorConfig::default()
        };
        let corrector =
            GrammarCorrector::from_store(fixture().vocab.clone(), store, 5, config);

        for neighbor in corrector.grammar_neighbors(&query, k) {
            prop_assert_eq!(neighbor.frequency, 1, "stored count is 1 for {:?}", neighbor.ids);
            prop_assert!(neighbor.distance <= k, "distance within radius for {:?}", neighbor.ids);
            prop_assert_eq!(
                word_edit_distance(&neighbor.ids, &query),
                neighbor.distance,
                "reported distance matches Levenshtein for {:?}",
                neighbor.ids
            );
        }
    }

    /// **`T_lex` value consistency.** Every lexical candidate carries the vocabulary's
    /// own index for its term (the OOV sentinel `0` excepted), so the term-id alphabet
    /// is coherent between `T_lex` and the n-gram store.
    #[test]
    fn lex_values_are_vocabulary_indices(token in "[a-z]{1,8}") {
        let fx = fixture();
        let corrector = GrammarCorrector::from_store(
            fx.vocab.clone(),
            fx.store.clone(),
            fx.order,
            GrammarCorrectorConfig::default(),
        );
        for candidate in corrector.lex_candidates(&token) {
            if candidate.id != 0 {
                prop_assert_eq!(Some(candidate.id), fx.vocab.get_index(&candidate.term));
            }
        }
    }

    /// **Cascade cost invariants.** Every returned hypothesis has a finite, non-negative
    /// cost (all three components — `` $`c_\text{lex} \ge 0`$ ``, edit penalties `` $`\ge 0`$ ``,
    /// and `` $`-\log S \ge 0`$ `` since `` $`S \in (0,1]`$ ``), and the decode is
    /// deterministic (equal input ⇒ equal output). The best hypothesis is ranked first.
    #[test]
    fn correction_costs_are_nonnegative_and_deterministic(
        words in prop::collection::vec("[a-z]{1,6}", 1..6usize),
    ) {
        let fx = fixture();
        let corrector = GrammarCorrector::from_store(
            fx.vocab.clone(),
            fx.store.clone(),
            fx.order,
            GrammarCorrectorConfig::default(),
        );
        let text = words.join(" ");

        let first = corrector.correct(&text);
        let second = corrector.correct(&text);
        prop_assert_eq!(&first, &second, "decode is deterministic");

        prop_assert!(!first.is_empty(), "non-empty input yields a hypothesis");
        for correction in &first {
            prop_assert!(correction.cost.is_finite(), "cost is finite: {correction:?}");
            prop_assert!(correction.cost >= 0.0, "cost is non-negative: {correction:?}");
        }
        for window in first.windows(2) {
            prop_assert!(window[0].cost <= window[1].cost, "hypotheses ranked by cost");
        }
    }
}
