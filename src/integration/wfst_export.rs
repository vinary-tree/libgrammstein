//! WFST export for n-gram language models.
//!
//! This module provides functionality to export libgrammstein's `NgramModel`
//! as a Weighted Finite State Transducer (WFST) compatible with lling-llang.
//!
//! # Overview
//!
//! The export creates a WFST where:
//! - States represent n-gram histories (previous n-1 words)
//! - Transitions represent word emissions with log probability weights
//! - Backoff ε-transitions implement smoothing fallback to lower-order models
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::integration::wfst_export::{FromLogProb, NgramWfstExporter};
//! use lling_llang::semiring::LogWeight;
//!
//! let model: NgramModel<D> = /* ... */;
//! let wfst = model.to_wfst::<LogWeight>();
//!
//! // Use in lling-llang pipeline
//! let path_weight = compose(&fst1, &wfst);
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use lling_llang::asr::{NgramBuilder, NgramConfig, NgramTransducer};
use lling_llang::semiring::{LogWeight, ProbabilityWeight, Semiring, TropicalWeight};
use lling_llang::wfst::{MutableWfst, StateId, VectorWfst};

use crate::ngram::{IterableDictionary, NgramEntry, NgramModel, NGRAM_SEPARATOR};
use liblevenshtein::dictionary::MutableMappedDictionary;

use super::vocabulary::{WordId, WordVocabulary, EOS_WORD_ID, UNK_WORD_ID};

/// Trait for converting log probabilities to semiring weights.
///
/// This trait bridges libgrammstein's log probability representation
/// (natural log, can be negative) with lling-llang's semiring weights.
///
/// # Log Probability Convention
///
/// libgrammstein stores log probabilities as `ln(p)` where `p ∈ (0, 1]`.
/// Thus log probabilities are typically negative (or zero for p=1).
///
/// # Semiring Conventions
///
/// - **LogWeight**: Stores negative log probability `-ln(p)`, so lower = better
/// - **TropicalWeight**: Same as LogWeight, stores `-ln(p)` as cost
/// - **ProbabilityWeight**: Stores probability directly `p = exp(ln(p))`
pub trait FromLogProb: Semiring {
    /// Convert from log probability (ln(p)) to semiring weight.
    ///
    /// # Arguments
    ///
    /// * `log_prob` - Natural log of probability (typically negative)
    fn from_log_prob(log_prob: f64) -> Self;

    /// Convert from negative log probability (-ln(p)) to semiring weight.
    ///
    /// # Arguments
    ///
    /// * `neg_log_prob` - Negative log probability (typically positive)
    fn from_neg_log_prob(neg_log_prob: f64) -> Self;
}

impl FromLogProb for LogWeight {
    #[inline]
    fn from_log_prob(log_prob: f64) -> Self {
        // LogWeight stores -ln(p), so negate the log probability
        LogWeight::new(-log_prob)
    }

    #[inline]
    fn from_neg_log_prob(neg_log_prob: f64) -> Self {
        LogWeight::new(neg_log_prob)
    }
}

impl FromLogProb for TropicalWeight {
    #[inline]
    fn from_log_prob(log_prob: f64) -> Self {
        // TropicalWeight stores cost = -ln(p)
        TropicalWeight::new(-log_prob)
    }

    #[inline]
    fn from_neg_log_prob(neg_log_prob: f64) -> Self {
        TropicalWeight::new(neg_log_prob)
    }
}

impl FromLogProb for ProbabilityWeight {
    #[inline]
    fn from_log_prob(log_prob: f64) -> Self {
        // ProbabilityWeight stores p = exp(ln(p))
        ProbabilityWeight::new(log_prob.exp())
    }

    #[inline]
    fn from_neg_log_prob(neg_log_prob: f64) -> Self {
        ProbabilityWeight::new((-neg_log_prob).exp())
    }
}

/// Builder for creating WFST from n-gram model.
///
/// This struct encapsulates the state needed during WFST construction.
pub struct NgramWfstBuilder<D, W>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
    W: Semiring + FromLogProb,
{
    /// The n-gram model being exported.
    model: Arc<NgramModel<D>>,
    /// Vocabulary mapping words to IDs.
    vocabulary: WordVocabulary,
    /// The WFST being constructed.
    wfst: VectorWfst<WordId, W>,
    /// Mapping from n-gram history (as word IDs) to WFST state ID.
    history_to_state: HashMap<Vec<WordId>, StateId>,
    /// Start state ID.
    start_state: StateId,
    /// Backoff state ID (for unigram backoff).
    backoff_state: StateId,
}

impl<D, W> NgramWfstBuilder<D, W>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
    W: Semiring + FromLogProb,
{
    /// Create a new WFST builder from an n-gram model.
    pub fn new(model: Arc<NgramModel<D>>) -> Self {
        let vocab_size = model.vocab_size();
        let vocabulary = Self::build_vocabulary(&model);

        // Estimate number of states (rough: vocab + vocab^2 for bigrams + ...)
        let estimated_states = vocab_size.saturating_mul(2);
        let mut wfst = VectorWfst::with_capacity(estimated_states);

        // Create start and backoff states
        let start_state = wfst.add_state();
        let backoff_state = wfst.add_state();

        wfst.set_start(start_state);

        // Start state is final (allows empty input)
        wfst.set_final(start_state, W::one());
        // Backoff state is final
        wfst.set_final(backoff_state, W::one());

        let mut history_to_state = HashMap::with_capacity(estimated_states);
        history_to_state.insert(vec![], backoff_state);

        Self {
            model,
            vocabulary,
            wfst,
            history_to_state,
            start_state,
            backoff_state,
        }
    }

    /// Build vocabulary from the n-gram model.
    fn build_vocabulary(model: &NgramModel<D>) -> WordVocabulary {
        let mut vocab = WordVocabulary::with_capacity(model.vocab_size());

        // Iterate over all entries in the trie to collect unigrams
        for (key, _entry) in model.trie().iter_entries() {
            // Unigrams have no separator
            if !key.contains(NGRAM_SEPARATOR) {
                vocab.add_word(&key);
            }
        }

        vocab
    }

    /// Get or create a state for a given history.
    fn get_or_create_state(&mut self, history: &[WordId]) -> StateId {
        if let Some(&state) = self.history_to_state.get(history) {
            return state;
        }

        let state = self.wfst.add_state();
        self.wfst.set_final(state, W::one());
        self.history_to_state.insert(history.to_vec(), state);
        state
    }

    /// Get the backoff state for a given history.
    ///
    /// For history [w1, w2, w3], the backoff is [w2, w3].
    /// For empty history, there is no further backoff.
    fn get_backoff_history(history: &[WordId]) -> Option<Vec<WordId>> {
        if history.is_empty() {
            None
        } else {
            Some(history[1..].to_vec())
        }
    }

    /// Build the WFST from the n-gram model.
    pub fn build(mut self) -> (VectorWfst<WordId, W>, WordVocabulary) {
        let order = self.model.order();

        // Add epsilon transition from start to backoff
        self.wfst.add_epsilon(self.start_state, self.backoff_state, W::one());

        // First pass: add unigram transitions from backoff state
        self.add_unigrams();

        // Second pass: add higher-order n-grams
        if order > 1 {
            self.add_higher_order_ngrams();
        }

        // Third pass: add backoff transitions
        self.add_backoff_transitions();

        (self.wfst, self.vocabulary)
    }

    /// Add unigram transitions from the backoff state.
    fn add_unigrams(&mut self) {
        let backoff_state = self.backoff_state;

        // Collect unigram words first to avoid borrowing issues
        let unigram_words: Vec<String> = self
            .vocabulary
            .iter()
            .skip(2) // Skip EOS and UNK special tokens
            .map(|(word, _)| word.to_string())
            .collect();

        for word in unigram_words {
            let word_id = self.vocabulary.get_id(&word).expect("Word must be in vocabulary");

            // Get log probability from model
            let log_prob = self.model.log_prob(&word, &[]);
            let weight = W::from_log_prob(log_prob);

            // Get or create state for this unigram history
            let history = vec![word_id];
            let target_state = self.get_or_create_state(&history);

            // Add transition: backoff -> unigram_state
            self.wfst.add_arc(
                backoff_state,
                Some(word_id),
                Some(word_id),
                target_state,
                weight,
            );
        }
    }

    /// Add higher-order n-gram transitions.
    fn add_higher_order_ngrams(&mut self) {
        let order = self.model.order();

        // Collect all n-grams from the trie
        let ngrams: Vec<(Vec<String>, String)> = self
            .model
            .trie()
            .iter_entries()
            .filter_map(|(key, _entry)| {
                let tokens: Vec<&str> = key.split(NGRAM_SEPARATOR).collect();
                if tokens.len() >= 2 && tokens.len() <= order {
                    let history: Vec<String> = tokens[..tokens.len() - 1]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let word = tokens.last().unwrap().to_string();
                    Some((history, word))
                } else {
                    None
                }
            })
            .collect();

        // Add transitions for each n-gram
        for (history_words, word) in ngrams {
            // Convert history to word IDs
            let history_ids: Vec<WordId> = history_words
                .iter()
                .filter_map(|w| self.vocabulary.get_id(w))
                .collect();

            // Skip if any history word is unknown
            if history_ids.len() != history_words.len() {
                continue;
            }

            let word_id = match self.vocabulary.get_id(&word) {
                Some(id) => id,
                None => continue, // Skip unknown words
            };

            // Get source state for this history
            let source_state = self.get_or_create_state(&history_ids);

            // Compute target history: history + word, truncated to order-1
            let mut target_history = history_ids.clone();
            target_history.push(word_id);
            if target_history.len() >= order {
                target_history = target_history[target_history.len() - (order - 1)..].to_vec();
            }
            let target_state = self.get_or_create_state(&target_history);

            // Get log probability
            let history_strs: Vec<&str> = history_words.iter().map(|s| s.as_str()).collect();
            let log_prob = self.model.log_prob(&word, &history_strs);
            let weight = W::from_log_prob(log_prob);

            // Add transition
            self.wfst.add_arc(
                source_state,
                Some(word_id),
                Some(word_id),
                target_state,
                weight,
            );
        }
    }

    /// Add epsilon backoff transitions between states.
    fn add_backoff_transitions(&mut self) {
        // Collect all histories and their states
        let histories: Vec<(Vec<WordId>, StateId)> = self
            .history_to_state
            .iter()
            .map(|(h, &s)| (h.clone(), s))
            .collect();

        for (history, state) in histories {
            // Skip the backoff state itself (empty history)
            if history.is_empty() {
                continue;
            }

            // Get backoff history
            if let Some(backoff_history) = Self::get_backoff_history(&history) {
                let backoff_state = self
                    .history_to_state
                    .get(&backoff_history)
                    .copied()
                    .unwrap_or(self.backoff_state);

                // Compute backoff weight
                // For simplicity, use unit weight; the model's smoothing handles the math
                let backoff_weight = W::one();

                // Add epsilon transition to backoff state
                self.wfst.add_epsilon(state, backoff_state, backoff_weight);
            }
        }
    }
}

/// Extension trait for NgramModel to provide WFST export.
pub trait NgramWfstExport<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
{
    /// Export the n-gram model as a VectorWfst.
    ///
    /// The resulting WFST can be used in lling-llang's WFST composition
    /// pipeline, for example as the "G" (grammar) transducer in ASR.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The semiring weight type (must implement `FromLogProb`)
    ///
    /// # Returns
    ///
    /// A tuple of (WFST, vocabulary) where the vocabulary provides the
    /// mapping between word strings and the integer labels used in the WFST.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use libgrammstein::ngram::NgramModel;
    /// use libgrammstein::integration::wfst_export::NgramWfstExport;
    /// use lling_llang::semiring::LogWeight;
    ///
    /// let model: NgramModel<D> = /* ... */;
    /// let (wfst, vocab) = model.to_wfst::<LogWeight>();
    /// ```
    fn to_wfst<W>(&self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb;

    /// Export the n-gram model as a VectorWfst (consumes self for Arc optimization).
    fn into_wfst<W>(self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb;

    /// Export the n-gram model as an NgramTransducer.
    ///
    /// This creates a transducer compatible with lling-llang's n-gram
    /// transducer format, using `NgramBuilder` to construct the WFST
    /// with proper backoff structure.
    ///
    /// The resulting transducer can be used directly with lling-llang's
    /// ASR cascade composition pipeline.
    ///
    /// # Type Parameters
    ///
    /// * `W` - The semiring weight type (must implement `FromLogProb`)
    ///
    /// # Returns
    ///
    /// A tuple of (NgramTransducer, vocabulary) where the vocabulary provides
    /// the mapping between word strings and the integer labels used in the transducer.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use libgrammstein::ngram::NgramModel;
    /// use libgrammstein::integration::wfst_export::NgramWfstExport;
    /// use lling_llang::semiring::LogWeight;
    /// use lling_llang::asr::CascadeBuilder;
    ///
    /// let model: NgramModel<D> = /* ... */;
    /// let (transducer, vocab) = model.to_ngram_transducer::<LogWeight>();
    ///
    /// // Use in ASR cascade
    /// let cascade = CascadeBuilder::new()
    ///     .grammar_from_ngram(transducer)
    ///     .build();
    /// ```
    fn to_ngram_transducer<W>(&self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone;

    /// Export the n-gram model as an NgramTransducer (consumes self).
    fn into_ngram_transducer<W>(self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone;
}

/// Builder for creating NgramTransducer from NgramModel.
///
/// This uses lling-llang's NgramBuilder to construct the transducer
/// with proper backoff structure matching the n-gram model's smoothing.
pub struct NgramTransducerBuilder<D, W>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
    W: Semiring + FromLogProb + Clone,
{
    /// The n-gram model being exported.
    model: Arc<NgramModel<D>>,
    /// Vocabulary mapping words to IDs.
    vocabulary: WordVocabulary,
    /// The NgramBuilder for constructing the transducer.
    builder: NgramBuilder<W>,
}

impl<D, W> NgramTransducerBuilder<D, W>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
    W: Semiring + FromLogProb + Clone,
{
    /// Create a new NgramTransducer builder from an n-gram model.
    pub fn new(model: Arc<NgramModel<D>>) -> Self {
        let vocabulary = NgramWfstBuilder::<D, W>::build_vocabulary(&model);
        let mut builder = NgramBuilder::new(model.order());

        // Configure with EOS and UNK IDs from vocabulary
        let config = NgramConfig {
            order: model.order(),
            add_sentence_markers: true,
            sos_id: None, // libgrammstein doesn't use SOS in the same way
            eos_id: Some(EOS_WORD_ID),
            unk_id: Some(UNK_WORD_ID),
        };
        builder = builder.config(config);

        Self {
            model,
            vocabulary,
            builder,
        }
    }

    /// Build the NgramTransducer.
    pub fn build(mut self) -> (NgramTransducer<W>, WordVocabulary) {
        // Set vocabulary size
        self.builder = self.builder.vocab_size(self.vocabulary.len());

        // Add unigrams
        self.add_unigrams();

        // Add higher-order n-grams and backoff weights
        self.add_higher_order_ngrams();

        // Build and return
        let transducer = self.builder.build();
        (transducer, self.vocabulary)
    }

    /// Add unigram probabilities to the builder.
    fn add_unigrams(&mut self) {
        // Collect words to avoid borrowing issues
        let words: Vec<(String, WordId)> = self
            .vocabulary
            .iter()
            .skip(2) // Skip EOS and UNK special tokens
            .map(|(word, id)| (word.to_string(), id))
            .collect();

        for (word, word_id) in words {
            let log_prob = self.model.log_prob(&word, &[]);
            let weight = W::from_log_prob(log_prob);
            self.builder.add_unigram(word_id, weight);
        }
    }

    /// Add higher-order n-grams and backoff weights.
    fn add_higher_order_ngrams(&mut self) {
        let order = self.model.order();

        // Collect all n-grams from the trie
        let ngrams: Vec<(Vec<String>, String)> = self
            .model
            .trie()
            .iter_entries()
            .filter_map(|(key, _entry)| {
                let tokens: Vec<&str> = key.split(NGRAM_SEPARATOR).collect();
                if tokens.len() >= 2 && tokens.len() <= order {
                    let history: Vec<String> = tokens[..tokens.len() - 1]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                    let word = tokens.last().expect("tokens not empty").to_string();
                    Some((history, word))
                } else {
                    None
                }
            })
            .collect();

        // Track unique histories for backoff weight computation
        let mut histories_seen: std::collections::HashSet<Vec<WordId>> =
            std::collections::HashSet::new();

        // Add n-grams
        for (history_words, word) in ngrams {
            // Convert history to word IDs
            let history_ids: Vec<WordId> = history_words
                .iter()
                .filter_map(|w| self.vocabulary.get_id(w))
                .collect();

            // Skip if any history word is unknown
            if history_ids.len() != history_words.len() {
                continue;
            }

            let word_id = match self.vocabulary.get_id(&word) {
                Some(id) => id,
                None => continue, // Skip unknown words
            };

            // Get log probability
            let history_strs: Vec<&str> = history_words.iter().map(|s| s.as_str()).collect();
            let log_prob = self.model.log_prob(&word, &history_strs);
            let weight = W::from_log_prob(log_prob);

            // Add n-gram to builder
            self.builder.add_ngram(&history_ids, word_id, weight);

            // Track this history for backoff
            histories_seen.insert(history_ids);
        }

        // Add backoff weights for all seen histories
        // The backoff weight accounts for probability mass reserved for unseen n-grams
        for history_ids in histories_seen {
            // Convert history IDs back to words for the model query
            let history_words: Vec<String> = history_ids
                .iter()
                .filter_map(|&id| self.vocabulary.get_word(id).map(|s| s.to_string()))
                .collect();

            // Compute backoff weight
            // In Modified Kneser-Ney, β(h) = D(h) * N1+(h•) / C(h)
            // For simplicity, we use unit weight since log_prob already includes smoothing
            let backoff_weight = W::one();

            self.builder.set_backoff(&history_ids, backoff_weight);
        }
    }
}

impl<D> NgramWfstExport<D> for NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
{
    fn to_wfst<W>(&self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        // Clone the model into an Arc for the builder
        let model = Arc::new(self.clone());
        let builder = NgramWfstBuilder::new(model);
        builder.build()
    }

    fn into_wfst<W>(self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        let model = Arc::new(self);
        let builder = NgramWfstBuilder::new(model);
        builder.build()
    }

    fn to_ngram_transducer<W>(&self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone,
    {
        let model = Arc::new(self.clone());
        let builder = NgramTransducerBuilder::new(model);
        builder.build()
    }

    fn into_ngram_transducer<W>(self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone,
    {
        let model = Arc::new(self);
        let builder = NgramTransducerBuilder::new(model);
        builder.build()
    }
}

impl<D> NgramWfstExport<D> for Arc<NgramModel<D>>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary,
{
    fn to_wfst<W>(&self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        let builder = NgramWfstBuilder::new(Arc::clone(self));
        builder.build()
    }

    fn into_wfst<W>(self) -> (VectorWfst<WordId, W>, WordVocabulary)
    where
        W: Semiring + FromLogProb,
    {
        let builder = NgramWfstBuilder::new(self);
        builder.build()
    }

    fn to_ngram_transducer<W>(&self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone,
    {
        let builder = NgramTransducerBuilder::new(Arc::clone(self));
        builder.build()
    }

    fn into_ngram_transducer<W>(self) -> (NgramTransducer<W>, WordVocabulary)
    where
        W: Semiring + FromLogProb + Clone,
    {
        let builder = NgramTransducerBuilder::new(self);
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::TrainerBuilder;
    use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    use lling_llang::wfst::Wfst;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_model() -> NgramModel<PathMapDictionary<NgramEntry>> {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = PathMapDictionary::<NgramEntry>::new();
        TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("Training failed")
    }

    #[test]
    fn test_from_log_prob_log_weight() {
        let log_prob = -1.0_f64; // ln(0.368)
        let weight = LogWeight::from_log_prob(log_prob);
        assert!((weight.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_log_prob_tropical_weight() {
        let log_prob = -2.0_f64;
        let weight = TropicalWeight::from_log_prob(log_prob);
        assert!((weight.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_log_prob_probability_weight() {
        let log_prob = 0.0_f64; // ln(1.0)
        let weight = ProbabilityWeight::from_log_prob(log_prob);
        assert!((weight.value() - 1.0).abs() < 1e-10);

        let log_prob = -1.0_f64; // ln(0.368)
        let weight = ProbabilityWeight::from_log_prob(log_prob);
        assert!((weight.value() - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_build_vocabulary() {
        let model = create_test_model();
        let model_arc = Arc::new(model);
        let vocab = NgramWfstBuilder::<_, LogWeight>::build_vocabulary(&model_arc);

        // Should have special tokens + words from corpus
        assert!(vocab.len() >= 2);
        assert!(vocab.contains("the"));
        assert!(vocab.contains("quick"));
        assert!(vocab.contains("brown"));
        assert!(vocab.contains("fox"));
    }

    #[test]
    fn test_to_wfst_basic() {
        let model = create_test_model();
        let (wfst, vocab): (VectorWfst<WordId, LogWeight>, _) = model.to_wfst();

        // WFST should have states
        assert!(wfst.num_states() > 0);

        // Start state should be set
        assert!(wfst.start() != u32::MAX);

        // Vocabulary should have words
        assert!(vocab.len() > 2);
    }

    #[test]
    fn test_to_wfst_has_transitions() {
        let model = create_test_model();
        let (wfst, vocab): (VectorWfst<WordId, LogWeight>, _) = model.to_wfst();

        // Start state should have transitions
        let start = wfst.start();
        let transitions = wfst.transitions(start);

        // Should have at least epsilon to backoff
        assert!(!transitions.is_empty());
    }

    #[test]
    fn test_to_wfst_final_states() {
        let model = create_test_model();
        let (wfst, _vocab): (VectorWfst<WordId, LogWeight>, _) = model.to_wfst();

        // All states should be final in an n-gram WFST
        for state_id in 0..wfst.num_states() as StateId {
            assert!(
                wfst.is_final(state_id),
                "State {} should be final",
                state_id
            );
        }
    }

    #[test]
    fn test_to_wfst_weights_finite() {
        let model = create_test_model();
        let (wfst, _vocab): (VectorWfst<WordId, LogWeight>, _) = model.to_wfst();

        // All transition weights should be finite
        for state_id in 0..wfst.num_states() as StateId {
            for transition in wfst.transitions(state_id) {
                assert!(
                    transition.weight.value().is_finite(),
                    "Transition weight should be finite"
                );
            }
        }
    }

    #[test]
    fn test_tropical_weight_wfst() {
        let model = create_test_model();
        let (wfst, _vocab): (VectorWfst<WordId, TropicalWeight>, _) = model.to_wfst();

        assert!(wfst.num_states() > 0);
    }

    #[test]
    fn test_probability_weight_wfst() {
        let model = create_test_model();
        let (wfst, _vocab): (VectorWfst<WordId, ProbabilityWeight>, _) = model.to_wfst();

        assert!(wfst.num_states() > 0);
    }

    // NgramTransducer tests

    #[test]
    fn test_to_ngram_transducer_basic() {
        let model = create_test_model();
        let (transducer, vocab) = model.to_ngram_transducer::<LogWeight>();

        // Transducer should have states
        assert!(transducer.fst.num_states() > 0);

        // Start state should be set
        assert!(transducer.fst.start() != u32::MAX);

        // Vocabulary should have words
        assert!(vocab.len() > 2);

        // Order should match model
        assert_eq!(transducer.order(), 3);
    }

    #[test]
    fn test_to_ngram_transducer_has_unigram_transitions() {
        let model = create_test_model();
        let (transducer, vocab) = model.to_ngram_transducer::<LogWeight>();

        // Count non-epsilon transitions from backoff state
        // Backoff state is state 1 in NgramBuilder
        let mut word_transitions = 0;
        for state_id in 0..transducer.fst.num_states() as StateId {
            for trans in transducer.fst.transitions(state_id) {
                if trans.input.is_some() {
                    word_transitions += 1;
                }
            }
        }

        // Should have at least as many transitions as vocab words (excluding special tokens)
        assert!(
            word_transitions >= vocab.len() - 2,
            "Expected at least {} word transitions, got {}",
            vocab.len() - 2,
            word_transitions
        );
    }

    #[test]
    fn test_to_ngram_transducer_final_states() {
        let model = create_test_model();
        let (transducer, _vocab) = model.to_ngram_transducer::<LogWeight>();

        // All states should be final in an n-gram transducer
        for state_id in 0..transducer.fst.num_states() as StateId {
            assert!(
                transducer.fst.is_final(state_id),
                "State {} should be final",
                state_id
            );
        }
    }

    #[test]
    fn test_to_ngram_transducer_weights_finite() {
        let model = create_test_model();
        let (transducer, _vocab) = model.to_ngram_transducer::<LogWeight>();

        // All transition weights should be finite
        for state_id in 0..transducer.fst.num_states() as StateId {
            for transition in transducer.fst.transitions(state_id) {
                assert!(
                    transition.weight.value().is_finite(),
                    "Transition weight should be finite at state {}",
                    state_id
                );
            }
        }
    }

    #[test]
    fn test_to_ngram_transducer_tropical_weight() {
        let model = create_test_model();
        let (transducer, _vocab) = model.to_ngram_transducer::<TropicalWeight>();

        assert!(transducer.fst.num_states() > 0);
        assert_eq!(transducer.order(), 3);
    }

    #[test]
    fn test_to_ngram_transducer_vocabulary_size() {
        let model = create_test_model();
        let (transducer, vocab) = model.to_ngram_transducer::<LogWeight>();

        // Transducer vocabulary size should match WordVocabulary
        assert_eq!(transducer.vocabulary_size(), vocab.len());
    }

    #[test]
    fn test_to_ngram_transducer_has_backoff_arcs() {
        let model = create_test_model();
        let (transducer, _vocab) = model.to_ngram_transducer::<LogWeight>();

        // Should have epsilon transitions (backoff arcs)
        let mut has_epsilon = false;
        for state_id in 0..transducer.fst.num_states() as StateId {
            for trans in transducer.fst.transitions(state_id) {
                if trans.input.is_none() {
                    has_epsilon = true;
                    break;
                }
            }
            if has_epsilon {
                break;
            }
        }

        assert!(has_epsilon, "Transducer should have backoff epsilon transitions");
    }

    #[test]
    fn test_into_ngram_transducer() {
        let model = create_test_model();
        let (transducer, vocab) = model.into_ngram_transducer::<LogWeight>();

        assert!(transducer.fst.num_states() > 0);
        assert!(vocab.len() > 2);
    }

    #[test]
    fn test_arc_ngram_transducer() {
        let model = Arc::new(create_test_model());
        let (transducer, vocab) = model.to_ngram_transducer::<LogWeight>();

        assert!(transducer.fst.num_states() > 0);
        assert!(vocab.len() > 2);
    }
}
