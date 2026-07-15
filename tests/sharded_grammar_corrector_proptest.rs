//! Property-based + differential tests for the **sharded** grammar corrector.
//!
//! These are the sharded analogue of `tests/grammar_corrector_proptest.rs`. They pin
//! down the guarantees the multi-shard refactor must uphold:
//!
//! 1. **Fanout soundness + completeness** — `grammar_neighbors_fanout` returns
//!    *exactly* the stored term-id sequences within word-edit distance `k`, across all
//!    shards (including first-token-edit neighbors), matching a brute-force reference.
//! 2. **Anchored narrowing** — the default `grammar_neighbors` returns exactly the
//!    same-shard subset of that set (the honest, decoder-relied-upon contract).
//! 3. **Decoder parity** — `correct()` over a sharded store is identical to `correct()`
//!    over a single store built from the same n-grams (the seam is behavior-preserving).
//! 4. **Read-only query mode** — a batch of queries (including one whose first-token
//!    routes to a never-populated shard) creates/modifies no shard files on disk; a
//!    reopened (open-from-disk) corpus is likewise served without writing.
//! 5. **Empty-history successor parity** — the sharded all-shards root fan-out returns
//!    the same first-token (unigram) set as the single store's whole view, keeping
//!    first-position insertion at parity (no boundary-less sentence-start gap).
//! 6. **Adaptive-granularity routing** — under a length-sensitive granularity, the
//!    produced-length (`|h|+1`) successor routing and each backoff level's own-length
//!    routing are pinned (invisible under the length-invariant `TwoChar`).
//!
//! Run with:
//! `cargo test --features "lling-llang-integration google-books" --test sharded_grammar_corrector_proptest`.

#![cfg(all(feature = "lling-llang-integration", feature = "google-books"))]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libgrammstein::integration::{
    GrammarCorrector, GrammarCorrectorConfig, NgramViewSource, ShardedGrammarCorrector,
    ShardedView, SingleView,
};
use libgrammstein::ngram::vocabulary::{create_vocabulary, encode_varint, SharedVocabARTrie};
use libgrammstein::sources::google_books::sharding::{
    compute_shard_key_from_token, ShardConfig, ShardCoordinator, ShardGranularity,
};
use liblevenshtein::transducer::Algorithm;
use proptest::prelude::*;
use tempfile::TempDir;

/// Standard (Levenshtein) word-level edit distance between two term-id sequences —
/// the brute-force reference (identical to the single-store proptest's).
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

/// A distinct 2-letter word for alphabet slot `k` (`k < 16`), so that under `TwoChar`
/// each slot routes to its OWN shard (the words share no 2-char prefix). This gives a
/// genuinely multi-shard corpus with deterministic routing.
fn alphabet_word(k: usize) -> String {
    format!(
        "{}{}",
        (b'a' + (k / 4) as u8) as char,
        (b'a' + (k % 4) as u8) as char
    )
}

fn config(algorithm: Algorithm) -> GrammarCorrectorConfig {
    GrammarCorrectorConfig {
        algorithm,
        ..GrammarCorrectorConfig::default()
    }
}

/// A random multi-shard corpus over a fixed alphabet of distinct-prefix words. The
/// generator produces sequences of alphabet *slots*; each slot maps to its assigned
/// vocabulary id (so `get_term(id)` recovers the routing string).
struct RandomCorpus {
    _dir: TempDir,
    coordinator: Arc<ShardCoordinator>,
    vocab: SharedVocabARTrie,
    /// The deduped stored id-sequences (the ground-truth set).
    stored: BTreeSet<Vec<u64>>,
    /// `alphabet[k]` = the vocabulary id assigned to `alphabet_word(k)`.
    alphabet: Vec<u64>,
    granularity: ShardGranularity,
}

impl RandomCorpus {
    /// Build from generated slot-sequences. `max_open_shards = 0` (all resident) is the
    /// required query-mode configuration.
    fn build(
        slot_sequences: &[Vec<usize>],
        alphabet_size: usize,
        granularity: ShardGranularity,
    ) -> Self {
        let dir = TempDir::new().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");
        let alphabet: Vec<u64> = (0..alphabet_size)
            .map(|k| {
                vocab
                    .as_ref()
                    .insert(&alphabet_word(k))
                    .expect("vocab insert")
            })
            .collect();

        let cfg = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(granularity.clone())
            .with_max_open_shards(0);
        let coordinator = Arc::new(ShardCoordinator::new(cfg).expect("coordinator"));

        // Map slots → ids, dedup to counts (count is irrelevant to neighbor queries).
        let mut counts: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
        for slots in slot_sequences {
            let ids: Vec<u64> = slots.iter().map(|&s| alphabet[s]).collect();
            *counts.entry(ids).or_insert(0) += 1;
        }
        for (ids, count) in &counts {
            store_ngram(&coordinator, &vocab, &granularity, ids, *count);
        }

        Self {
            _dir: dir,
            coordinator,
            vocab,
            stored: counts.keys().cloned().collect(),
            alphabet,
            granularity,
        }
    }

    fn corrector(&self) -> ShardedGrammarCorrector {
        ShardedGrammarCorrector::new(
            Arc::clone(&self.coordinator),
            self.vocab.clone(),
            5,
            1, // total_count is unused by neighbor queries
            config(Algorithm::Standard),
        )
    }

    /// The shard key a stored/queried sequence routes to — mirrors the importer.
    fn shard_of(&self, ids: &[u64]) -> String {
        let first = self.vocab.get_term(ids[0]).expect("first token");
        compute_shard_key_from_token(&first, ids.len() as u8, &self.granularity).as_file_stem()
    }
}

/// Store one n-gram exactly as the importer would: route by (first-token, length),
/// key by concatenated LEB128 varints.
fn store_ngram(
    coordinator: &ShardCoordinator,
    vocab: &SharedVocabARTrie,
    granularity: &ShardGranularity,
    ids: &[u64],
    count: u64,
) {
    let first = vocab.get_term(ids[0]).expect("first token");
    let shard_key = compute_shard_key_from_token(&first, ids.len() as u8, granularity);
    let mut key = Vec::new();
    for &id in ids {
        encode_varint(id, &mut key);
    }
    coordinator
        .store_in_shard(&shard_key, &key, count)
        .expect("store_in_shard");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// **Fanout soundness + completeness.** Over a random multi-shard corpus, an
    /// arbitrary query, and a radius `k`, `grammar_neighbors_fanout` returns exactly the
    /// stored sequences within word-edit distance `k` — including neighbors whose first
    /// token differs from the query's (which live in other shards). This is the sharded
    /// analogue of the single-store soundness+completeness property.
    #[test]
    fn fanout_neighbors_are_sound_and_complete(
        slot_seqs in prop::collection::vec(prop::collection::vec(0usize..8, 1..5usize), 1..14usize),
        query_slots in prop::collection::vec(0usize..8, 1..5usize),
        k in 0usize..=2usize,
    ) {
        let corpus = RandomCorpus::build(&slot_seqs, 8, ShardGranularity::TwoChar);
        let query: Vec<u64> = query_slots.iter().map(|&s| corpus.alphabet[s]).collect();
        let corrector = corpus.corrector();

        let got: BTreeSet<Vec<u64>> = corrector
            .grammar_neighbors_fanout(&query, k)
            .into_iter()
            .map(|neighbor| neighbor.ids)
            .collect();

        let expected: BTreeSet<Vec<u64>> = corpus
            .stored
            .iter()
            .filter(|sequence| word_edit_distance(sequence, &query) <= k)
            .cloned()
            .collect();

        prop_assert_eq!(got, expected);
    }

    /// **Anchored narrowing.** The default `grammar_neighbors` returns exactly the
    /// same-shard subset of the full within-`k` set — i.e. the stored sequences that are
    /// within `k` AND co-located with the query's own `(first-token, length)` shard.
    /// This documents-and-verifies the narrowed contract the decoder relies on.
    #[test]
    fn anchored_neighbors_are_the_same_shard_subset(
        slot_seqs in prop::collection::vec(prop::collection::vec(0usize..8, 1..5usize), 1..14usize),
        query_slots in prop::collection::vec(0usize..8, 1..5usize),
        k in 0usize..=2usize,
    ) {
        let corpus = RandomCorpus::build(&slot_seqs, 8, ShardGranularity::TwoChar);
        let query: Vec<u64> = query_slots.iter().map(|&s| corpus.alphabet[s]).collect();
        let corrector = corpus.corrector();
        let query_shard = corpus.shard_of(&query);

        let got: BTreeSet<Vec<u64>> = corrector
            .grammar_neighbors(&query, k)
            .into_iter()
            .map(|neighbor| neighbor.ids)
            .collect();

        let expected: BTreeSet<Vec<u64>> = corpus
            .stored
            .iter()
            .filter(|sequence| {
                word_edit_distance(sequence, &query) <= k && corpus.shard_of(sequence) == query_shard
            })
            .cloned()
            .collect();

        prop_assert_eq!(got, expected);
    }
}

/// A sentence corpus stored BOTH as a single `DynamicDawg` and as a sharded coordinator,
/// for differential decoder-parity + read-only tests.
struct SentenceCorpus {
    _dir: TempDir,
    coordinator: Arc<ShardCoordinator>,
    single: DynamicDawg<u64>,
    vocab: SharedVocabARTrie,
    total_count: u64,
    order: usize,
    shard_dir: PathBuf,
    granularity: ShardGranularity,
}

impl SentenceCorpus {
    fn build(sentences: &[&str], order: usize, granularity: ShardGranularity) -> Self {
        Self::build_with_boundaries(sentences, order, granularity, true)
    }

    /// Build with or without `<s>`/`</s>` boundary padding. Boundary-LESS corpora
    /// (`boundaries = false`, the Google-Books reality) leave `bos_id = None`, so the
    /// decoder reaches the EMPTY-history successor oracle at the sentence start — the
    /// path the sharded `all_shards_root_successors` fan-out must serve at parity.
    fn build_with_boundaries(
        sentences: &[&str],
        order: usize,
        granularity: ShardGranularity,
        boundaries: bool,
    ) -> Self {
        let dir = TempDir::new().expect("tempdir");
        let vocab = create_vocabulary(&dir.path().join("vocab")).expect("vocabulary");

        // Count all 1..=order n-grams, optionally with boundary padding.
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        for sentence in sentences {
            let mut words: Vec<&str> = if boundaries {
                vec!["<s>"; order - 1]
            } else {
                Vec::new()
            };
            words.extend(sentence.split_whitespace());
            if boundaries {
                words.push("</s>");
            }
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

        // Single store.
        let single = DynamicDawg::<u64>::new();
        for (sequence, count) in &counts {
            let mut key = Vec::new();
            for &id in sequence {
                encode_varint(id, &mut key);
            }
            single.insert_bytes_with_value(&key, *count);
        }

        // Sharded store (same n-grams, same counts, importer routing).
        let shard_dir = dir.path().join("shards");
        let cfg = ShardConfig::new(shard_dir.clone())
            .with_granularity(granularity.clone())
            .with_max_open_shards(0);
        let coordinator = Arc::new(ShardCoordinator::new(cfg).expect("coordinator"));
        for (sequence, count) in &counts {
            store_ngram(&coordinator, &vocab, &granularity, sequence, *count);
        }

        // Corpus total N = Σ unigram counts (matches `sum_unigram_counts`).
        let total_count: u64 = counts
            .iter()
            .filter(|(sequence, _)| sequence.len() == 1)
            .map(|(_, count)| *count)
            .sum();

        Self {
            _dir: dir,
            coordinator,
            single,
            vocab,
            total_count,
            order,
            shard_dir,
            granularity,
        }
    }

    fn single_corrector(&self) -> GrammarCorrector<DynamicDawg<u64>> {
        GrammarCorrector::new(
            self.vocab.clone(),
            self.single.clone(),
            self.order,
            self.total_count,
            GrammarCorrectorConfig::default(),
        )
    }

    fn sharded_corrector(&self) -> ShardedGrammarCorrector {
        ShardedGrammarCorrector::new(
            Arc::clone(&self.coordinator),
            self.vocab.clone(),
            self.order,
            self.total_count,
            GrammarCorrectorConfig::default(),
        )
    }

    /// Sorted `(file_name, len, mtime)` snapshot of the shard directory.
    fn shard_dir_snapshot(&self) -> Vec<(String, u64, SystemTime)> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.shard_dir).expect("read shard dir") {
            let entry = entry.expect("dir entry");
            let meta = entry.metadata().expect("metadata");
            out.push((
                entry.file_name().to_string_lossy().into_owned(),
                meta.len(),
                meta.modified().expect("mtime"),
            ));
        }
        out.sort();
        out
    }
}

const SENTENCES: &[&str] = &[
    "the quick brown fox",
    "the quick brown fox",
    "the quick brown fox",
    "the quick brown dog",
    "the lazy brown fox",
    "the quick brown fox jumps over",
];

/// **Decoder parity.** `correct()` over the sharded store is identical to `correct()`
/// over a single store built from the same n-grams — proving the `NgramViewSource` seam
/// is behavior-preserving (the whole point of `GrammarCore<P>`).
#[test]
fn sharded_correct_matches_single_store() {
    let corpus = SentenceCorpus::build(SENTENCES, 3, ShardGranularity::TwoChar);
    let single = corpus.single_corrector();
    let sharded = corpus.sharded_corrector();

    for input in [
        "teh quick brown fox",     // substitution
        "the quikc brown fox",     // transposition
        "the the quick brown fox", // extraneous word
        "the quick fox",           // missing word (insertion)
        "the quick brown fox",     // clean
        "the lazy brown dog",      // mixed
    ] {
        let single_out = single.correct(input);
        let sharded_out = sharded.correct(input);
        assert_eq!(
            single_out, sharded_out,
            "sharded and single-store decodes must match for {input:?}"
        );
    }
}

/// **Read-only query mode.** A batch of queries — including one whose first-token routes
/// to a never-populated shard — creates or modifies NO shard files on disk. This is the
/// concrete guarantee behind `get_shard_readonly` (open-only, no create, no eviction
/// checkpoint) under `max_open_shards = 0`.
#[test]
fn query_mode_creates_no_shard_files() {
    let corpus = SentenceCorpus::build(SENTENCES, 3, ShardGranularity::TwoChar);
    let sharded = corpus.sharded_corrector();

    // An extra vocab word that never begins any stored n-gram → routes to an absent shard.
    let absent_id = corpus.vocab.as_ref().insert("zz").expect("insert absent");
    let absent_prefix = format!(
        "shard_{}",
        compute_shard_key_from_token("zz", 1, &corpus.granularity).as_file_stem()
    );

    let before = corpus.shard_dir_snapshot();
    assert!(
        !before
            .iter()
            .any(|(name, _, _)| name.starts_with(&absent_prefix)),
        "precondition: no shard file exists for the absent first-token 'zz'"
    );

    // A batch of read-only queries, including the absent-first-token one.
    let _ = sharded.correct("the quick brown fox");
    let _ = sharded.correct("teh quick fox");
    let _ = sharded.grammar_neighbors(&[absent_id], 1);
    let _ = sharded.grammar_neighbors(&[absent_id, absent_id], 2);
    let _ = sharded.grammar_neighbors_fanout(&[absent_id], 1);
    let _ = sharded.correct("zz zz zz");

    let after = corpus.shard_dir_snapshot();
    assert_eq!(
        before, after,
        "query mode must not create or modify any shard file (absent first-token must \
         not materialize a shard)"
    );
}

/// **Fanout recovers what anchored cannot.** A first-token substitution neighbor lives in
/// a different shard than the query, so the anchored `grammar_neighbors` misses it while
/// `grammar_neighbors_fanout` recovers it — the concrete M4 demonstration.
#[test]
fn fanout_recovers_first_token_edit_neighbor() {
    // Two trigrams differing only in the FIRST token (→ different shards under TwoChar).
    let slot_seqs = vec![vec![0usize, 1, 2], vec![3usize, 1, 2]];
    let corpus = RandomCorpus::build(&slot_seqs, 8, ShardGranularity::TwoChar);
    let corrector = corpus.corrector();

    let query: Vec<u64> = vec![corpus.alphabet[0], corpus.alphabet[1], corpus.alphabet[2]];
    let first_token_neighbor: Vec<u64> =
        vec![corpus.alphabet[3], corpus.alphabet[1], corpus.alphabet[2]];

    // The two sequences are in different shards.
    assert_ne!(
        corpus.shard_of(&query),
        corpus.shard_of(&first_token_neighbor),
        "the first-token-edit neighbor must live in a different shard"
    );

    let anchored: BTreeSet<Vec<u64>> = corrector
        .grammar_neighbors(&query, 1)
        .into_iter()
        .map(|n| n.ids)
        .collect();
    let fanout: BTreeSet<Vec<u64>> = corrector
        .grammar_neighbors_fanout(&query, 1)
        .into_iter()
        .map(|n| n.ids)
        .collect();

    assert!(
        anchored.contains(&query),
        "anchored finds the exact (same-shard) sequence"
    );
    assert!(
        !anchored.contains(&first_token_neighbor),
        "anchored MISSES the first-token-edit neighbor (different shard): {anchored:?}"
    );
    assert!(
        fanout.contains(&query) && fanout.contains(&first_token_neighbor),
        "fanout recovers BOTH the exact sequence and the first-token-edit neighbor: {fanout:?}"
    );
}

/// **Reopen from disk.** After the corpus is finalized (`checkpoint_all`), a FRESH
/// coordinator opened over the same shard directory answers `grammar_neighbors_fanout`
/// and `correct()` identically to the just-populated (resident) corrector — i.e. the
/// sharded corrector works in the real deployment scenario, reading pre-built shards
/// from disk rather than from a resident overlay.
#[test]
fn reopen_from_disk_matches_resident_queries() {
    let corpus = SentenceCorpus::build(SENTENCES, 3, ShardGranularity::TwoChar);

    // Baseline from the resident (just-populated) corrector.
    let resident = corpus.sharded_corrector();
    let query: Vec<u64> = ["the", "quick", "brown"]
        .iter()
        .map(|w| corpus.vocab.get_index(w).expect("query word in vocab"))
        .collect();
    let baseline_neighbors: BTreeSet<Vec<u64>> = resident
        .grammar_neighbors_fanout(&query, 1)
        .into_iter()
        .map(|neighbor| neighbor.ids)
        .collect();
    let baseline_correct = resident.correct("teh quick brown fox");
    assert!(
        !baseline_neighbors.is_empty(),
        "the corpus is non-trivial (sanity)"
    );

    // Finalize to disk, then open a FRESH coordinator over the same shard directory —
    // forcing open-from-disk (on-demand `ShardHandle::open` + WAL replay).
    corpus.coordinator.checkpoint_all().expect("checkpoint_all");
    let cfg = ShardConfig::new(corpus.shard_dir.clone())
        .with_granularity(ShardGranularity::TwoChar)
        .with_max_open_shards(0);
    let reopened_coordinator = Arc::new(ShardCoordinator::new(cfg).expect("reopen coordinator"));
    let reopened = ShardedGrammarCorrector::new(
        reopened_coordinator,
        corpus.vocab.clone(),
        corpus.order,
        corpus.total_count,
        GrammarCorrectorConfig::default(),
    );

    // The finalized corpus is served strictly read-only: querying the reopened
    // (open-from-disk) coordinator must not create or modify any shard file — WAL replay
    // on a cleanly-checkpointed corpus is a no-op.
    let files_before = corpus.shard_dir_snapshot();

    let reopened_neighbors: BTreeSet<Vec<u64>> = reopened
        .grammar_neighbors_fanout(&query, 1)
        .into_iter()
        .map(|neighbor| neighbor.ids)
        .collect();
    let reopened_correct = reopened.correct("teh quick brown fox");

    let files_after = corpus.shard_dir_snapshot();
    assert_eq!(
        files_before, files_after,
        "querying a reopened (open-from-disk) corpus must not create or modify shard files"
    );

    assert_eq!(
        baseline_neighbors, reopened_neighbors,
        "fanout neighbors must survive a checkpoint + reopen from disk"
    );
    assert_eq!(
        baseline_correct, reopened_correct,
        "correct() must survive a checkpoint + reopen from disk"
    );
}

/// **Adaptive-granularity routing — the `|h|+1` produced-length invariant.** Under
/// `Adaptive`, order-1 n-grams route to a 1-char-prefix shard and order ≥ 2 to a
/// 2-char-prefix shard, so a history and the `(history+1)`-gram it produces can live in
/// DIFFERENT shards. This pins the successor oracle's produced-length routing and each
/// stupid-backoff level's own-length routing — the subtlest correctness property, and one
/// the length-invariant `TwoChar` granularity (every other test) cannot exercise. A
/// regression routing by `|h|` instead of `|h|+1` would open the wrong shard and diverge
/// from the single store here.
#[test]
fn sharded_correct_matches_single_store_adaptive() {
    for order in [2usize, 3] {
        let corpus = SentenceCorpus::build(SENTENCES, order, ShardGranularity::Adaptive);
        let single = corpus.single_corrector();
        let sharded = corpus.sharded_corrector();
        for input in [
            "teh quick brown fox",     // substitution
            "the quikc brown fox",     // transposition
            "the the quick brown fox", // extraneous word
            "the quick fox",           // missing word (insertion)
            "the quick brown fox",     // clean
            "the lazy brown dog",      // mixed
        ] {
            assert_eq!(
                single.correct(input),
                sharded.correct(input),
                "adaptive order {order}: sharded and single decodes must match for {input:?}"
            );
        }
    }
}

/// **Empty-history successor parity — the all-shards fan-out fix.** Over a BOUNDARY-LESS
/// multi-shard corpus (`bos_id = None`, so the decoder reaches the empty-history oracle at
/// the sentence start), the sharded `successors(&[])` — the all-shards root fan-out —
/// returns exactly the set of first-tokens (unigrams) that the single store's
/// `successors(&[])` (its `whole_view().root().edges()`) returns. This is the completeness
/// that keeps sharded first-position insertion at parity with the single-store decoder:
/// BEFORE the fix, `ShardedView::whole_view()` was `None` so the sharded `successors(&[])`
/// was empty, and the sharded decoder could not insert a word before the first token.
/// Non-empty histories are checked too, confirming the anchored routing agrees.
#[test]
fn sharded_successors_match_single_store_empty_and_nonempty_history() {
    // Run under BOTH a length-invariant (`TwoChar`) and a length-SENSITIVE (`Adaptive`)
    // granularity. Under Adaptive a 1-token history's produced (2-gram) successors live in
    // the 2-char-prefix shard, while the 1-char-prefix shard holds only unigrams — so the
    // NON-EMPTY case here pins the produced-length (`|h|+1`) successor routing that the
    // differential decode cannot (insertions are cost-neutral, so a mis-routed successor
    // set does not change `correct()`'s output).
    for granularity in [ShardGranularity::TwoChar, ShardGranularity::Adaptive] {
        let corpus =
            SentenceCorpus::build_with_boundaries(SENTENCES, 3, granularity.clone(), false);
        let single_view = SingleView::new(corpus.single.clone());
        let sharded_view = ShardedView::new(Arc::clone(&corpus.coordinator), corpus.vocab.clone());

        // Empty history → every stored first-token (all unigrams), identical across backends.
        let single_roots: BTreeSet<u64> = single_view.successors(&[]).into_iter().collect();
        let sharded_roots: BTreeSet<u64> = sharded_view.successors(&[]).into_iter().collect();
        assert!(
            !single_roots.is_empty(),
            "{granularity:?}: the corpus has unigrams (sanity)"
        );
        assert_eq!(
            single_roots, sharded_roots,
            "{granularity:?}: empty-history successors (all unigrams) must match — the \
             all-shards fan-out equals the single store's whole-view root edges"
        );

        // Non-empty history → same set (co-located, produced-length routing).
        let the = corpus.vocab.get_index("the").expect("'the' in vocab");
        let quick = corpus.vocab.get_index("quick").expect("'quick' in vocab");
        let single_succ: BTreeSet<u64> = single_view.successors(&[the]).into_iter().collect();
        let sharded_succ: BTreeSet<u64> = sharded_view.successors(&[the]).into_iter().collect();
        assert_eq!(
            single_succ, sharded_succ,
            "{granularity:?}: non-empty-history successors must match (pins produced-length routing)"
        );
        assert!(
            single_succ.contains(&quick),
            "{granularity:?}: 'the' → 'quick' is a known continuation (sanity)"
        );
    }
}
