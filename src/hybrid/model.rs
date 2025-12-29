//! Hybrid language model combining n-gram and embedding scores.
//!
//! This model provides:
//! - Combined scoring from n-gram and embedding models
//! - OOV handling via embedding similarity
//! - Configurable interpolation weights

use crate::embedding::SubwordEmbedding;
use crate::ngram::{NgramEntry, NgramModel};
use liblevenshtein::dictionary::MutableMappedDictionary;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;

#[cfg(feature = "serde-extras")]
use std::path::Path;

/// Interpolation strategy for combining n-gram and embedding scores.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InterpolationStrategy {
    /// Linear interpolation: α * ngram + (1-α) * embedding
    Linear {
        /// Weight for n-gram model (0.0 to 1.0).
        alpha: f64,
    },

    /// Log-linear combination: log P = α * log P_ngram + (1-α) * log P_emb
    LogLinear {
        /// Weight for n-gram model (0.0 to 1.0).
        alpha: f64,
    },

    /// N-gram only (use embedding only for OOV)
    NgramWithEmbeddingFallback,

    /// Dynamic weighting based on n-gram context length
    /// Higher weight to n-gram when more context is available
    Dynamic {
        /// Base weight for n-gram
        base_alpha: f64,
        /// Additional weight per context word
        alpha_per_context: f64,
        /// Maximum alpha
        max_alpha: f64,
    },
}

impl Default for InterpolationStrategy {
    fn default() -> Self {
        Self::Linear { alpha: 0.8 }
    }
}

/// Configuration for hybrid language model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridConfig {
    /// Interpolation strategy.
    pub strategy: InterpolationStrategy,

    /// Cache size for combined scores.
    pub cache_size: usize,

    /// Smoothing factor for embedding probabilities.
    pub embedding_smoothing: f64,

    /// Temperature for similarity-to-probability conversion.
    pub temperature: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            strategy: InterpolationStrategy::default(),
            cache_size: 50_000,
            embedding_smoothing: 1e-8,
            temperature: 1.0,
        }
    }
}

/// Cache key for combined scores.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CacheKey {
    word: String,
    context: Vec<String>,
}

/// Default cache constructor for serde deserialization.
/// Uses the default cache size from HybridConfig.
fn default_cache() -> Mutex<LruCache<CacheKey, f64>> {
    let cache_size = NonZeroUsize::new(HybridConfig::default().cache_size)
        .expect("Default cache size must be non-zero");
    Mutex::new(LruCache::new(cache_size))
}

/// Hybrid language model combining n-gram and embedding models.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig};
/// use libgrammstein::ngram::NgramModel;
/// use libgrammstein::embedding::SubwordEmbedding;
///
/// let ngram_model = NgramModel::load("ngram.bin")?;
/// let embedding_model = SubwordEmbedding::load("emb.bin")?;
///
/// let hybrid = HybridLanguageModel::new(ngram_model, embedding_model, HybridConfig::default());
///
/// // Score a word in context
/// let score = hybrid.score("fox", &["the", "quick", "brown"]);
/// ```
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "D: serde::Serialize + serde::de::DeserializeOwned")]
pub struct HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// N-gram model.
    ngram: NgramModel<D>,

    /// Embedding model.
    embedding: SubwordEmbedding,

    /// Configuration.
    config: HybridConfig,

    /// Score cache (not serialized - reconstructed on load).
    #[serde(skip, default = "default_cache")]
    cache: Mutex<LruCache<CacheKey, f64>>,
}

impl<D> HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// Create a new hybrid model.
    pub fn new(ngram: NgramModel<D>, embedding: SubwordEmbedding, config: HybridConfig) -> Self {
        let cache_size = NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            ngram,
            embedding,
            config,
            cache: Mutex::new(LruCache::new(cache_size)),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(ngram: NgramModel<D>, embedding: SubwordEmbedding) -> Self {
        Self::new(ngram, embedding, HybridConfig::default())
    }

    /// Get the n-gram model.
    pub fn ngram_model(&self) -> &NgramModel<D> {
        &self.ngram
    }

    /// Get the embedding model.
    pub fn embedding_model(&self) -> &SubwordEmbedding {
        &self.embedding
    }

    /// Get configuration.
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Score a word given context.
    ///
    /// Returns log probability of the word given the context.
    pub fn score(&self, word: &str, context: &[&str]) -> f64 {
        // Check cache
        let cache_key = CacheKey {
            word: word.to_string(),
            context: context.iter().map(|s| s.to_string()).collect(),
        };

        if let Ok(mut cache) = self.cache.lock() {
            if let Some(&cached_score) = cache.get(&cache_key) {
                return cached_score;
            }
        }

        // Compute score based on strategy
        let score = match self.config.strategy {
            InterpolationStrategy::Linear { alpha } => {
                self.score_linear(word, context, alpha)
            }
            InterpolationStrategy::LogLinear { alpha } => {
                self.score_log_linear(word, context, alpha)
            }
            InterpolationStrategy::NgramWithEmbeddingFallback => {
                self.score_with_fallback(word, context)
            }
            InterpolationStrategy::Dynamic { base_alpha, alpha_per_context, max_alpha } => {
                let alpha = (base_alpha + alpha_per_context * context.len() as f64).min(max_alpha);
                self.score_linear(word, context, alpha)
            }
        };

        // Update cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(cache_key, score);
        }

        score
    }

    /// Linear interpolation scoring.
    fn score_linear(&self, word: &str, context: &[&str], alpha: f64) -> f64 {
        let ngram_log_prob = self.ngram.log_prob(word, context);
        let embedding_log_prob = self.embedding_log_prob(word, context);

        // Handle -inf values by using a minimum log probability
        let min_log_prob = -50.0;
        let ngram_log_prob = ngram_log_prob.max(min_log_prob);
        let embedding_log_prob = embedding_log_prob.max(min_log_prob);

        // Convert to probabilities, interpolate, convert back to log
        let ngram_prob = ngram_log_prob.exp();
        let embedding_prob = embedding_log_prob.exp();

        let combined_prob = alpha * ngram_prob + (1.0 - alpha) * embedding_prob;
        combined_prob.max(f64::MIN_POSITIVE).ln()
    }

    /// Log-linear combination scoring.
    fn score_log_linear(&self, word: &str, context: &[&str], alpha: f64) -> f64 {
        let ngram_log_prob = self.ngram.log_prob(word, context);
        let embedding_log_prob = self.embedding_log_prob(word, context);

        // Handle -inf values by using a minimum log probability
        let min_log_prob = -50.0;
        let ngram_log_prob = ngram_log_prob.max(min_log_prob);
        let embedding_log_prob = embedding_log_prob.max(min_log_prob);

        alpha * ngram_log_prob + (1.0 - alpha) * embedding_log_prob
    }

    /// N-gram with embedding fallback for OOV.
    fn score_with_fallback(&self, word: &str, context: &[&str]) -> f64 {
        // Handle -inf values by using a minimum log probability
        let min_log_prob = -50.0;

        // Check if word is in n-gram vocabulary (has unigram count > 0)
        if self.ngram.count(&[word]) > 0 {
            self.ngram.log_prob(word, context).max(min_log_prob)
        } else {
            // OOV: use embedding-based probability
            self.embedding_log_prob(word, context).max(min_log_prob)
        }
    }

    /// Compute log probability using embeddings.
    ///
    /// Uses cosine similarity to estimate word probability in context.
    fn embedding_log_prob(&self, word: &str, context: &[&str]) -> f64 {
        if context.is_empty() {
            // No context: use uniform probability based on vocabulary
            return -(self.embedding.vocab_size() as f64).ln();
        }

        // Get word vector
        let word_vec = self.embedding.word_vector(word);

        // Get context vector (average of context word vectors)
        let context_vec = self.embedding.sentence_vector(context);

        // Compute cosine similarity
        let similarity = Self::cosine_similarity(&word_vec, &context_vec);

        // Convert similarity to probability via softmax-like scaling
        // P(w|c) ∝ exp(similarity / temperature)
        let scaled_sim = (similarity as f64) / self.config.temperature;

        // Normalize (approximate) using log-space
        // This is a simplification; full normalization would require summing over vocab
        let log_prob = scaled_sim - 1.0; // Subtract 1 as approximate normalization

        // Apply smoothing
        log_prob.max((self.config.embedding_smoothing).ln())
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Score a complete sentence.
    ///
    /// Returns the total log probability of the sentence.
    pub fn sentence_log_prob(&self, words: &[&str]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let order = self.ngram.order();
        let mut total_log_prob = 0.0;

        for (i, word) in words.iter().enumerate() {
            // Get context (up to order-1 previous words)
            let context_start = i.saturating_sub(order - 1);
            let context: Vec<&str> = words[context_start..i].iter().copied().collect();

            total_log_prob += self.score(word, &context);
        }

        total_log_prob
    }

    /// Compute perplexity of a sentence.
    pub fn perplexity(&self, words: &[&str]) -> f64 {
        if words.is_empty() {
            return f64::INFINITY;
        }

        let log_prob = self.sentence_log_prob(words);
        let avg_log_prob = log_prob / words.len() as f64;

        (-avg_log_prob).exp()
    }

    /// Find the most likely next word given context.
    ///
    /// Uses embedding similarity to rank candidates.
    pub fn predict_next(&self, context: &[&str], candidates: &[&str]) -> Option<(String, f64)> {
        if candidates.is_empty() {
            return None;
        }

        let mut best_word = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for &candidate in candidates {
            let score = self.score(candidate, context);
            if score > best_score {
                best_score = score;
                best_word = candidate.to_string();
            }
        }

        Some((best_word, best_score))
    }

    /// Clear the score cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

// Implement Send + Sync for the hybrid model
unsafe impl<D> Send for HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
}

unsafe impl<D> Sync for HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
}

// Serialization support (requires D: Serialize + DeserializeOwned and bincode via serde-extras)
#[cfg(feature = "serde-extras")]
impl<D> HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    /// Save the hybrid model to a binary file.
    ///
    /// Uses bincode for efficient binary serialization.
    /// Requires the dictionary to implement serde traits.
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.save("hybrid_model.bin")?;
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// Load a hybrid model from a binary file.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model: HybridLanguageModel<DynamicDawgChar<NgramEntry>> = HybridLanguageModel::load("hybrid_model.bin")?;
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let model = bincode::deserialize_from(reader)?;
        Ok(model)
    }
}

/// Portable hybrid model format for serialization.
///
/// This format doesn't require the dictionary to implement serde traits,
/// making it compatible with all dictionary backends.
#[cfg(feature = "serde-extras")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PortableHybridModel {
    /// N-gram model in portable format.
    pub ngram: crate::ngram::PortableNgramModel,
    /// Embedding model (already serializable).
    pub embedding: SubwordEmbedding,
    /// Hybrid configuration.
    pub config: HybridConfig,
}

// Portable serialization support (works with any dictionary, requires bincode via serde-extras)
#[cfg(feature = "serde-extras")]
impl<D> HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// Export to portable format for serialization.
    ///
    /// This method exports the model in a format that doesn't require
    /// the dictionary type to implement serde traits.
    pub fn to_portable(&self) -> PortableHybridModel
    where
        D: crate::ngram::IterableDictionary,
    {
        PortableHybridModel {
            ngram: self.ngram.to_portable(),
            embedding: self.embedding.clone(),
            config: self.config.clone(),
        }
    }

    /// Save model to a portable binary file.
    ///
    /// This format can be loaded into any dictionary backend.
    ///
    /// # Example
    ///
    /// ```ignore
    /// model.save_portable("hybrid_model.bin")?;
    /// ```
    pub fn save_portable<P: AsRef<Path>>(&self, path: P) -> crate::Result<()>
    where
        D: crate::ngram::IterableDictionary,
    {
        let portable = self.to_portable();
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &portable)?;
        Ok(())
    }

    /// Load model from a portable binary file.
    ///
    /// Reconstructs the model using the provided dictionary factory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    ///     HybridLanguageModel::load_portable("hybrid_model.bin", DynamicDawgChar::new)?;
    /// ```
    pub fn load_portable<P, F>(path: P, dictionary_factory: F) -> crate::Result<Self>
    where
        P: AsRef<Path>,
        F: FnOnce() -> D,
    {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let portable: PortableHybridModel = bincode::deserialize_from(reader)?;

        // Reconstruct N-gram model
        let ngram = crate::ngram::NgramModel::load_portable_from_portable(
            portable.ngram,
            dictionary_factory,
        )?;

        let cache_size = std::num::NonZeroUsize::new(portable.config.cache_size)
            .unwrap_or(std::num::NonZeroUsize::new(1).unwrap());

        Ok(Self {
            ngram,
            embedding: portable.embedding,
            config: portable.config,
            cache: Mutex::new(LruCache::new(cache_size)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::embedding::EmbeddingTrainerBuilder;
    use crate::ngram::TrainerBuilder;
    use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    fn create_test_models() -> (NgramModel<PathMapDictionary<NgramEntry>>, SubwordEmbedding) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        // Train n-gram model
        let dictionary = PathMapDictionary::<NgramEntry>::new();
        let ngram_model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("N-gram training failed");

        // Train embedding model
        let reader2 = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let embedding_model = EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(2)
            .train(reader2)
            .expect("Embedding training failed");

        (ngram_model, embedding_model)
    }

    #[test]
    fn test_hybrid_creation() {
        let (ngram, embedding) = create_test_models();
        let config = HybridConfig::default();
        let _hybrid = HybridLanguageModel::new(ngram, embedding, config);
    }

    #[test]
    fn test_hybrid_score() {
        let (ngram, embedding) = create_test_models();
        let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);

        let score = hybrid.score("fox", &["the", "quick"]);
        assert!(score.is_finite());
        assert!(score < 0.0); // Log probability should be negative
    }

    #[test]
    fn test_sentence_log_prob() {
        let (ngram, embedding) = create_test_models();
        let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);

        let log_prob = hybrid.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_perplexity() {
        let (ngram, embedding) = create_test_models();
        let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);

        let ppl = hybrid.perplexity(&["the", "quick", "brown", "fox"]);
        assert!(ppl.is_finite());
        assert!(ppl > 0.0);
    }

    #[test]
    fn test_interpolation_strategies() {
        let (ngram, embedding) = create_test_models();

        // Test linear interpolation
        let config1 = HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha: 0.5 },
            ..Default::default()
        };
        let hybrid1 = HybridLanguageModel::new(ngram.clone(), embedding.clone(), config1);
        let score1 = hybrid1.score("fox", &["the"]);
        assert!(score1.is_finite());

        // Test log-linear interpolation
        let config2 = HybridConfig {
            strategy: InterpolationStrategy::LogLinear { alpha: 0.5 },
            ..Default::default()
        };
        let hybrid2 = HybridLanguageModel::new(ngram.clone(), embedding.clone(), config2);
        let score2 = hybrid2.score("fox", &["the"]);
        assert!(score2.is_finite());

        // Test fallback strategy
        let config3 = HybridConfig {
            strategy: InterpolationStrategy::NgramWithEmbeddingFallback,
            ..Default::default()
        };
        let hybrid3 = HybridLanguageModel::new(ngram.clone(), embedding.clone(), config3);
        let score3 = hybrid3.score("fox", &["the"]);
        assert!(score3.is_finite());
    }

    #[test]
    fn test_cache() {
        let (ngram, embedding) = create_test_models();
        let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);

        // First call
        let score1 = hybrid.score("fox", &["the", "quick"]);

        // Second call should use cache
        let score2 = hybrid.score("fox", &["the", "quick"]);

        assert_eq!(score1, score2);

        // Clear cache
        hybrid.clear_cache();
    }

    #[test]
    fn test_predict_next() {
        let (ngram, embedding) = create_test_models();
        let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);

        let candidates = ["fox", "dog", "cat"];
        let result = hybrid.predict_next(&["the", "quick", "brown"], &candidates);

        assert!(result.is_some());
        let (word, score) = result.unwrap();
        assert!(candidates.contains(&word.as_str()));
        assert!(score.is_finite());
    }

    #[cfg(feature = "serde-extras")]
    mod serde_tests {
        use super::*;
        use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

        fn create_serializable_test_models() -> (NgramModel<DynamicDawgChar<NgramEntry>>, SubwordEmbedding) {
            let dir = TempDir::new().expect("Failed to create temp dir");
            let content = "the quick brown fox the quick brown dog the lazy fox \
                           the quick brown fox the quick brown dog the lazy fox \
                           the quick brown fox the quick brown dog the lazy fox";
            let path = create_test_corpus(dir.path(), content);
            let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

            // Train n-gram model with DynamicDawgChar (serializable backend)
            let dictionary = DynamicDawgChar::<NgramEntry>::new();
            let ngram_model = TrainerBuilder::new(dictionary)
                .order(3)
                .train(reader)
                .expect("N-gram training failed");

            // Train embedding model
            let reader2 = PlaintextReader::from_file(&path).expect("Failed to create reader");
            let embedding_model = EmbeddingTrainerBuilder::new()
                .dim(10)
                .window_size(2)
                .min_count(1)
                .epochs(2)
                .train(reader2)
                .expect("Embedding training failed");

            (ngram_model, embedding_model)
        }

        #[test]
        fn test_hybrid_save_load_roundtrip() {
            let (ngram, embedding) = create_serializable_test_models();
            let config = HybridConfig {
                strategy: InterpolationStrategy::Linear { alpha: 0.7 },
                cache_size: 1000,
                ..Default::default()
            };
            let hybrid = HybridLanguageModel::new(ngram, embedding, config);
            let temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");

            // Save the model
            hybrid.save(temp_file.path()).expect("Failed to save hybrid model");

            // Verify file was created with content
            let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get file metadata");
            assert!(metadata.len() > 0, "Saved model file should not be empty");

            // Load the model
            let loaded: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
                HybridLanguageModel::load(temp_file.path()).expect("Failed to load hybrid model");

            // Verify config matches
            assert_eq!(hybrid.config().cache_size, loaded.config().cache_size);
            match (hybrid.config().strategy, loaded.config().strategy) {
                (InterpolationStrategy::Linear { alpha: a1 }, InterpolationStrategy::Linear { alpha: a2 }) => {
                    assert!((a1 - a2).abs() < 1e-10, "Alpha should match");
                }
                _ => panic!("Strategy should match"),
            }

            // Verify scores match
            let orig_score = hybrid.score("fox", &["the", "quick"]);
            let loaded_score = loaded.score("fox", &["the", "quick"]);
            assert!(
                (orig_score - loaded_score).abs() < 1e-10,
                "Scores should match after roundtrip: {} vs {}",
                orig_score,
                loaded_score
            );

            // Verify sentence scores match
            let orig_sentence = hybrid.sentence_log_prob(&["the", "quick", "brown", "fox"]);
            let loaded_sentence = loaded.sentence_log_prob(&["the", "quick", "brown", "fox"]);
            assert!(
                (orig_sentence - loaded_sentence).abs() < 1e-10,
                "Sentence scores should match: {} vs {}",
                orig_sentence,
                loaded_sentence
            );

            // Verify perplexity matches
            let orig_ppl = hybrid.perplexity(&["the", "quick", "brown", "fox"]);
            let loaded_ppl = loaded.perplexity(&["the", "quick", "brown", "fox"]);
            assert!(
                (orig_ppl - loaded_ppl).abs() < 1e-8,
                "Perplexity should match: {} vs {}",
                orig_ppl,
                loaded_ppl
            );
        }
    }
}
