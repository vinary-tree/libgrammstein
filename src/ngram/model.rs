//! N-gram language model with probability queries.
//!
//! This module provides the main `NgramModel` struct that combines the n-gram
//! trie with smoothing algorithms for probability estimation.

use super::entry::NgramEntry;
use super::smoothing::KneserNeySmoothing;
use super::trie::NgramTrie;
use libdictenstein::MappedDictionary;
// Only the `serde-extras` portable-load impl below needs the mutable trait; gating the
// import to match keeps the default-feature build warning-free.
#[cfg(feature = "serde-extras")]
use libdictenstein::MutableMappedDictionary;

#[cfg(feature = "serde-extras")]
use std::path::Path;

/// N-gram language model with Modified Kneser-Ney smoothing.
///
/// This is the main interface for training and querying n-gram language models.
///
/// # Type Parameters
///
/// * `D` - Dictionary backend type (e.g., `DynamicDawgChar<NgramEntry>`)
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::NgramModel;
/// use libgrammstein::corpus::PlaintextReader;
///
/// // Train a trigram model
/// let reader = PlaintextReader::from_file("corpus.txt")?;
/// let model = NgramModel::train(reader, 3)?;
///
/// // Query probabilities
/// let log_prob = model.log_prob("fox", &["quick", "brown"]);
/// println!("log P(fox | quick brown) = {}", log_prob);
///
/// // Score a sentence
/// let sentence_log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
/// ```
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "D: serde::Serialize + serde::de::DeserializeOwned")]
pub struct NgramModel<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    /// N-gram trie storage.
    trie: NgramTrie<D>,

    /// Smoothing algorithm.
    smoothing: KneserNeySmoothing,

    /// Vocabulary size (number of unique unigrams).
    vocab_size: usize,

    /// Total unigram count (corpus size in tokens).
    total_count: u64,
}

impl<D> NgramModel<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    /// Create a new n-gram model from a trained trie.
    ///
    /// This is typically called after the training process completes.
    pub fn new(
        trie: NgramTrie<D>,
        smoothing: KneserNeySmoothing,
        vocab_size: usize,
        total_count: u64,
    ) -> Self {
        Self {
            trie,
            smoothing,
            vocab_size,
            total_count,
        }
    }

    /// Get the n-gram order (maximum context length + 1).
    #[inline]
    pub fn order(&self) -> usize {
        self.trie.max_order()
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the total unigram count.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Get a reference to the underlying trie.
    #[inline]
    pub fn trie(&self) -> &NgramTrie<D> {
        &self.trie
    }

    /// Get the raw count for an n-gram.
    #[inline]
    pub fn count(&self, tokens: &[&str]) -> u64 {
        self.trie.count(tokens)
    }

    /// Compute log probability of a word given context.
    ///
    /// Uses Modified Kneser-Ney smoothing with backoff to lower-order models.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to compute probability for
    /// * `context` - The preceding context words (may be empty for unigram)
    ///
    /// # Returns
    ///
    /// Log probability (base e) of the word given the context.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // P(fox | quick brown)
    /// let log_prob = model.log_prob("fox", &["quick", "brown"]);
    ///
    /// // P(the) - unigram
    /// let log_prob_unigram = model.log_prob("the", &[]);
    /// ```
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        self.smoothing
            .log_prob(word, context, &self.trie, self.vocab_size, self.total_count)
    }

    /// The Modified Kneser-Ney smoothing parameters (discounts and the
    /// `N₁₊(•,•)` continuation denominator) computed for this model.
    pub fn smoothing(&self) -> &KneserNeySmoothing {
        &self.smoothing
    }

    /// Compute log probability of a complete sentence.
    ///
    /// Sums log probabilities of each word given its context, using the
    /// appropriate context length based on position in the sentence.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The sentence tokens
    ///
    /// # Returns
    ///
    /// Log probability (base e) of the entire sentence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
    /// ```
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let order = self.order();
        let mut total_log_prob = 0.0;

        for i in 0..tokens.len() {
            let word = tokens[i];
            let context_start = i.saturating_sub(order - 1);
            let context = &tokens[context_start..i];
            total_log_prob += self.log_prob(word, context);
        }

        total_log_prob
    }

    /// Check if a word is in the vocabulary (has been seen during training).
    #[inline]
    pub fn in_vocabulary(&self, word: &str) -> bool {
        self.trie.contains(&[word])
    }

    /// Get the number of n-grams stored in the model.
    #[inline]
    pub fn ngram_count(&self) -> usize {
        self.trie.len()
    }

    /// Get the log probability assigned to out-of-vocabulary words.
    ///
    /// This is the uniform distribution over the vocabulary: log(1/V).
    #[inline]
    pub fn oov_log_prob(&self) -> f64 {
        if self.vocab_size == 0 {
            f64::NEG_INFINITY
        } else {
            -(self.vocab_size as f64).ln()
        }
    }
}

impl<D> Clone for NgramModel<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    fn clone(&self) -> Self {
        Self {
            trie: self.trie.clone(),
            smoothing: self.smoothing.clone(),
            vocab_size: self.vocab_size,
            total_count: self.total_count,
        }
    }
}

// Serialization support (requires bincode via serde-extras feature)
#[cfg(feature = "serde-extras")]
impl<D> NgramModel<D>
where
    D: MappedDictionary<Value = NgramEntry> + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Save the model to a binary file.
    ///
    /// Uses bincode for efficient binary serialization.
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.save("model.bin")?;
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load a model from a binary file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("model.bin")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        Ok(model)
    }
}

// Portable serialization that doesn't require D: Serialize
// This exports the model as a list of (key, entry) pairs

/// Portable vocabulary format for serialization.
///
/// This allows vocabulary-indexed models to be self-contained, including
/// the mapping from PUA characters to words.
#[cfg(feature = "serde-extras")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct PortableVocabulary {
    /// Words indexed by their PUA character offset from PUA_START.
    ///
    /// Index 0 corresponds to PUA_START (U+F0000), index 1 to U+F0001, etc.
    pub words: Vec<String>,
}

/// Portable N-gram model format for serialization.
///
/// This format doesn't require the dictionary to implement serde traits,
/// making it compatible with all dictionary backends.
#[cfg(feature = "serde-extras")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PortableNgramModel {
    /// N-gram entries as (key, snapshot) pairs.
    pub entries: Vec<(String, crate::ngram::NgramEntrySnapshot)>,
    /// Maximum n-gram order.
    pub max_order: usize,
    /// Vocabulary size (unique unigrams).
    pub vocab_size: usize,
    /// Total token count.
    pub total_count: u64,
    /// Smoothing parameters.
    pub smoothing: KneserNeySmoothing,
    /// Optional vocabulary for vocabulary-indexed models.
    ///
    /// When present, the model uses PUA character encoding. When absent,
    /// the model uses legacy pipe-separated encoding.
    #[serde(default)]
    pub vocabulary: Option<PortableVocabulary>,
}

#[cfg(feature = "serde-extras")]
impl<D> NgramModel<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    /// Export to portable format for serialization (without vocabulary).
    ///
    /// This method iterates over all dictionary entries and exports them
    /// as (key, snapshot) pairs, allowing serialization without requiring
    /// the dictionary type to implement serde traits.
    ///
    /// For vocabulary-indexed models, use `to_portable_with_vocabulary` instead
    /// to include the vocabulary mapping.
    pub fn to_portable(&self) -> PortableNgramModel
    where
        D: crate::ngram::trie::IterableDictionary,
    {
        self.to_portable_with_vocabulary(None)
    }

    /// Export to portable format with optional vocabulary.
    ///
    /// When a vocabulary is provided, the resulting portable model is self-contained
    /// and can be decoded back to human-readable words.
    ///
    /// # Arguments
    ///
    /// * `vocabulary` - Optional shared vocabulary for vocabulary-indexed models
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For vocabulary-indexed models
    /// let portable = model.to_portable_with_vocabulary(Some(&vocab));
    ///
    /// // For legacy models or when vocabulary is not needed
    /// let portable = model.to_portable_with_vocabulary(None);
    /// ```
    pub fn to_portable_with_vocabulary(
        &self,
        vocabulary: Option<&crate::ngram::SharedVocabARTrie>,
    ) -> PortableNgramModel
    where
        D: crate::ngram::trie::IterableDictionary,
    {
        let entries: Vec<(String, crate::ngram::NgramEntrySnapshot)> = self
            .trie
            .iter_entries()
            .map(|(key, entry)| (key, crate::ngram::NgramEntrySnapshot::from(&entry)))
            .collect();

        // Convert vocabulary to portable format if provided
        // Use O(1) get_term() lookups, iterating in index order
        let portable_vocab = vocabulary.map(|vocab| {
            let guard = vocab.read();
            let len = guard.len();
            let mut words = Vec::with_capacity(len);

            // Indices are 1-based (FIRST_VALID_INDEX = 1), iterate in order
            for i in 1..=(len as u64) {
                if let Some(term) = guard.get_term(i) {
                    words.push(term);
                }
            }

            PortableVocabulary { words }
        });

        PortableNgramModel {
            entries,
            max_order: self.trie.max_order(),
            vocab_size: self.vocab_size,
            total_count: self.total_count,
            smoothing: self.smoothing.clone(),
            vocabulary: portable_vocab,
        }
    }

    /// Save model to a portable binary file.
    ///
    /// This format can be loaded into any dictionary backend.
    /// For vocabulary-indexed models, use `save_portable_with_vocabulary` to
    /// include the vocabulary mapping in the output.
    pub fn save_portable<P: AsRef<Path>>(&self, path: P) -> crate::Result<()>
    where
        D: crate::ngram::trie::IterableDictionary,
    {
        let portable = self.to_portable();
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &portable)?;
        Ok(())
    }

    /// Save model to a portable binary file with vocabulary included.
    ///
    /// The vocabulary mapping is included in the output, making the model
    /// file self-contained and allowing decoding of PUA keys to words.
    pub fn save_portable_with_vocabulary<P: AsRef<Path>>(
        &self,
        path: P,
        vocabulary: &crate::ngram::SharedVocabARTrie,
    ) -> crate::Result<()>
    where
        D: crate::ngram::trie::IterableDictionary,
    {
        let portable = self.to_portable_with_vocabulary(Some(vocabulary));
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &portable)?;
        Ok(())
    }
}

// Loading from a portable snapshot fills an empty backend via `insert_with_value`, so it
// requires a writable (mutable) dictionary. Kept separate from the read-only query/export
// surface above.
#[cfg(feature = "serde-extras")]
impl<D> NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// Load model from a portable binary file.
    ///
    /// Reconstructs the model using the provided dictionary factory.
    pub fn load_portable<P, F>(path: P, dictionary_factory: F) -> crate::Result<Self>
    where
        P: AsRef<Path>,
        F: FnOnce() -> D,
    {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let portable: PortableNgramModel = bincode::deserialize_from(reader)?;

        Self::load_portable_from_portable(portable, dictionary_factory)
    }

    /// Reconstruct model from a portable format struct.
    ///
    /// This is used internally by HybridLanguageModel to load embedded N-gram models.
    pub fn load_portable_from_portable<F>(
        portable: PortableNgramModel,
        dictionary_factory: F,
    ) -> crate::Result<Self>
    where
        F: FnOnce() -> D,
    {
        let dictionary = dictionary_factory();
        for (key, snapshot) in portable.entries {
            dictionary.insert_with_value(&key, NgramEntry::from(snapshot));
        }

        let trie = NgramTrie::new(dictionary, portable.max_order);

        Ok(Self {
            trie,
            smoothing: portable.smoothing,
            vocab_size: portable.vocab_size,
            total_count: portable.total_count,
        })
    }
}

/// Static (read-only) `DoubleArrayTrieChar`-backed construction.
///
/// A Double-Array Trie is bulk-built and immutable — it answers queries faster than the
/// dynamic backends but cannot be trained or updated. These constructors let `--to-static`
/// (and downstream inference) load a portable model into the fast static backend.
#[cfg(feature = "serde-extras")]
impl NgramModel<libdictenstein::double_array_trie::char::DoubleArrayTrieChar<NgramEntry>> {
    /// Build a static `DoubleArrayTrieChar`-backed model from a portable snapshot.
    ///
    /// `from_terms_with_values` sorts the entries with the trie builder's own comparator,
    /// so no manual key ordering is required. The result answers `count`/`log_prob`
    /// identically to the source model.
    pub fn from_portable_static(portable: PortableNgramModel) -> Self {
        use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

        let entries = portable
            .entries
            .into_iter()
            .map(|(key, snapshot)| (key, NgramEntry::from(snapshot)));
        let dictionary = DoubleArrayTrieChar::from_terms_with_values(entries);
        let trie = NgramTrie::new(dictionary, portable.max_order);

        Self {
            trie,
            smoothing: portable.smoothing,
            vocab_size: portable.vocab_size,
            total_count: portable.total_count,
        }
    }

    /// Load a static `DoubleArrayTrieChar`-backed model from a portable binary file.
    ///
    /// The fast read-only counterpart to [`NgramModel::load_portable`].
    pub fn load_static_portable<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let portable: PortableNgramModel = bincode::deserialize_from(reader)?;
        Ok(Self::from_portable_static(portable))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::TrainerBuilder;
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
    use libdictenstein::pathmap::PathMapDictionary;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    fn create_test_ngram_model() -> NgramModel<DynamicDawgChar<NgramEntry>> {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let dictionary = DynamicDawgChar::<NgramEntry>::new();
        TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("N-gram training failed")
    }

    #[test]
    fn test_model_properties() {
        let model = create_test_ngram_model();
        assert_eq!(model.order(), 3);
        assert!(model.vocab_size() > 0);
        assert!(model.total_count() > 0);
    }

    #[test]
    fn test_log_prob() {
        let model = create_test_ngram_model();

        // Test known n-gram
        let log_prob = model.log_prob("fox", &["brown"]);
        assert!(log_prob.is_finite());
        assert!(log_prob <= 0.0);

        // Test unigram
        let unigram_prob = model.log_prob("the", &[]);
        assert!(unigram_prob.is_finite());
    }

    #[test]
    fn test_log_prob_finite_for_unseen_ngrams() {
        // Regression for the Modified Kneser-Ney backoff bug: an unseen n-gram whose
        // context IS seen must still yield a finite (backed-off) log probability.
        // Previously discount(0) == 0 zeroed both the discounted term and the
        // interpolation weight, giving prob = 0 and log_prob = -inf.
        let model = create_test_ngram_model();

        // "lazy" is a seen context, but "lazy dog" never occurs in the corpus.
        let unseen = model.log_prob("dog", &["lazy"]);
        assert!(
            unseen.is_finite(),
            "unseen bigram must be finite, got {unseen}"
        );
        assert!(unseen <= 0.0);

        // A higher-order unseen n-gram backs off through multiple levels.
        let unseen3 = model.log_prob("dog", &["the", "lazy"]);
        assert!(
            unseen3.is_finite(),
            "unseen trigram must be finite, got {unseen3}"
        );

        // An out-of-vocabulary word and an empty-token continuation (the case that
        // triggered the WFST weight panics) must be finite too.
        assert!(model.log_prob("zzz_oov", &["the"]).is_finite());
        assert!(model.log_prob("", &["the"]).is_finite());
    }

    #[cfg(feature = "serde-extras")]
    #[test]
    fn static_dat_model_matches_source() {
        use libdictenstein::double_array_trie::char::DoubleArrayTrieChar;

        let model = create_test_ngram_model();
        let static_model: NgramModel<DoubleArrayTrieChar<NgramEntry>> =
            NgramModel::from_portable_static(model.to_portable());

        // Structural equality.
        assert_eq!(static_model.order(), model.order());
        assert_eq!(static_model.vocab_size(), model.vocab_size());
        assert_eq!(static_model.total_count(), model.total_count());
        assert_eq!(static_model.ngram_count(), model.ngram_count());

        // The static DAT-backed model must answer counts and log-probs identically to the
        // source for seen, unseen, and OOV n-grams.
        let cases: &[(&str, &[&str])] = &[
            ("fox", &["quick", "brown"]),
            ("brown", &["quick"]),
            ("the", &[]),
            ("dog", &["lazy"]),
            ("dog", &["the", "lazy"]),
            ("zzz_oov", &["the"]),
        ];
        for (w, ctx) in cases {
            let mut toks = ctx.to_vec();
            toks.push(w);
            assert_eq!(
                static_model.count(&toks),
                model.count(&toks),
                "count mismatch for {toks:?}"
            );
            let a = model.log_prob(w, ctx);
            let b = static_model.log_prob(w, ctx);
            assert!(
                (a - b).abs() < 1e-12,
                "log_prob mismatch for ({w:?}, {ctx:?}): {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_sentence_log_prob() {
        let model = create_test_ngram_model();

        let log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[cfg(feature = "serde-extras")]
    #[test]
    fn test_ngram_save_load_roundtrip() {
        let model = create_test_ngram_model();
        let temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");

        // Save the model
        model.save(temp_file.path()).expect("Failed to save model");

        // Verify file was created with content
        let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get file metadata");
        assert!(metadata.len() > 0, "Saved model file should not be empty");

        // Load the model
        let loaded: NgramModel<DynamicDawgChar<NgramEntry>> =
            NgramModel::load(temp_file.path()).expect("Failed to load model");

        // Verify properties match
        assert_eq!(model.order(), loaded.order());
        assert_eq!(model.vocab_size(), loaded.vocab_size());
        assert_eq!(model.total_count(), loaded.total_count());

        // Verify probabilities match
        let orig_prob = model.log_prob("fox", &["the", "quick"]);
        let loaded_prob = loaded.log_prob("fox", &["the", "quick"]);
        assert!(
            probs_equal(orig_prob, loaded_prob),
            "Log probabilities should match after roundtrip: {} vs {}",
            orig_prob,
            loaded_prob
        );

        // Verify sentence probabilities match
        let orig_sentence_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        let loaded_sentence_prob = loaded.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        assert!(
            probs_equal(orig_sentence_prob, loaded_sentence_prob),
            "Sentence log probabilities should match: {} vs {}",
            orig_sentence_prob,
            loaded_sentence_prob
        );
    }

    /// Helper to compare probabilities that may be -inf.
    #[cfg(feature = "serde-extras")]
    fn probs_equal(a: f64, b: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            a.signum() == b.signum() // Both -inf or both +inf
        } else if a.is_nan() || b.is_nan() {
            false
        } else {
            (a - b).abs() < 1e-10
        }
    }

    #[test]
    fn test_pathmap_model() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox";
        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let dictionary = PathMapDictionary::<NgramEntry>::new();
        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("N-gram training failed");

        // Basic functionality check
        let log_prob = model.log_prob("fox", &["brown"]);
        assert!(log_prob.is_finite());
    }
}
