//! Phonetic-enhanced embeddings combining orthographic and phonetic similarity.
//!
//! This module extends the FastText-style subword embeddings with phonetic awareness,
//! using liblevenshtein's verified phonetic rewrite rules to cluster words that
//! sound similar together.
//!
//! # Approach
//!
//! The `PhoneticEmbedding` combines two similarity measures:
//!
//! 1. **Orthographic similarity**: Standard subword embedding cosine similarity
//! 2. **Phonetic similarity**: Similarity after phonetic normalization
//!
//! Words are phonetically normalized using Zompist-based rewrite rules (e.g., "gh" → "f"),
//! then their embeddings are compared. This helps cluster words that sound similar
//! but are spelled differently (e.g., "phone" and "fone").
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::embedding::{PhoneticEmbedding, SubwordEmbedding};
//!
//! let ortho_model = SubwordEmbedding::load("embeddings.bin")?;
//! let phonetic = PhoneticEmbedding::new(ortho_model)
//!     .with_phonetic_weight(0.3);  // 30% phonetic, 70% orthographic
//!
//! // "enough" and "enuf" should have high phonetic similarity
//! let sim = phonetic.similarity("enough", "enuf");
//! assert!(sim > 0.8);
//! ```
//!
//! # Verification
//!
//! The phonetic rules are formally verified in Coq/Rocq with proofs for:
//! - Bounded expansion (no unbounded memory growth)
//! - Termination (always reaches fixed point)
//! - Idempotence (stable transformations)
//!
//! See liblevenshtein's `docs/verification/phonetic/` for details.

use dashmap::DashMap;
use liblevenshtein::phonetic::{zompist_rules_char, OnlinePhoneticTransducerChar, RewriteRuleChar};
use std::sync::Arc;

use super::SubwordEmbedding;

/// Default weight for phonetic similarity component.
pub const DEFAULT_PHONETIC_WEIGHT: f64 = 0.3;

/// Maximum fuel for phonetic rule application (prevents infinite loops).
pub const DEFAULT_PHONETIC_FUEL: usize = 1000;

/// Phonetic-enhanced embedding model.
///
/// Combines orthographic (character n-gram) similarity with phonetic similarity
/// for improved OOV handling and error-tolerant matching.
///
/// # Thread Safety
///
/// This type is `Send + Sync` and can be safely shared across threads.
/// The internal phonetic normalization cache uses DashMap for lock-free access.
#[derive(Debug)]
pub struct PhoneticEmbedding {
    /// Underlying orthographic embedding model.
    orthographic: Arc<SubwordEmbedding>,

    /// Phonetic rewrite rules for normalization.
    rules: Vec<RewriteRuleChar>,

    /// Weight for phonetic vs orthographic similarity [0.0, 1.0].
    /// Combined similarity = (1 - weight) * ortho + weight * phonetic
    phonetic_weight: f64,

    /// Cache for phonetically normalized strings.
    /// Key: original word, Value: normalized form
    normalization_cache: DashMap<String, String>,

    /// Maximum cache size for normalized strings.
    max_cache_size: usize,
}

impl PhoneticEmbedding {
    /// Create a new phonetic embedding with default Zompist rules.
    ///
    /// Uses the 62 verified English phonetic rules from liblevenshtein,
    /// including orthography rules, vowel digraphs, and phonetic mappings.
    ///
    /// # Arguments
    ///
    /// * `orthographic` - The underlying subword embedding model
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = SubwordEmbedding::load("embeddings.bin")?;
    /// let phonetic = PhoneticEmbedding::new(model);
    /// ```
    pub fn new(orthographic: SubwordEmbedding) -> Self {
        Self {
            orthographic: Arc::new(orthographic),
            rules: zompist_rules_char(),
            phonetic_weight: DEFAULT_PHONETIC_WEIGHT,
            normalization_cache: DashMap::new(),
            max_cache_size: 100_000,
        }
    }

    /// Create from an Arc-wrapped orthographic model.
    ///
    /// Useful when sharing the orthographic model across multiple consumers.
    pub fn from_arc(orthographic: Arc<SubwordEmbedding>) -> Self {
        Self {
            orthographic,
            rules: zompist_rules_char(),
            phonetic_weight: DEFAULT_PHONETIC_WEIGHT,
            normalization_cache: DashMap::new(),
            max_cache_size: 100_000,
        }
    }

    /// Set custom phonetic rules.
    ///
    /// # Arguments
    ///
    /// * `rules` - Custom rewrite rules for phonetic normalization
    ///
    /// # Example
    ///
    /// ```ignore
    /// use liblevenshtein::phonetic::orthography_rules_char;
    ///
    /// let phonetic = PhoneticEmbedding::new(model)
    ///     .with_rules(orthography_rules_char());
    /// ```
    pub fn with_rules(mut self, rules: Vec<RewriteRuleChar>) -> Self {
        self.rules = rules;
        self.normalization_cache.clear();
        self
    }

    /// Set the phonetic weight.
    ///
    /// The weight controls the balance between orthographic and phonetic similarity:
    /// - 0.0: Pure orthographic similarity (no phonetic component)
    /// - 0.5: Equal weighting
    /// - 1.0: Pure phonetic similarity (no orthographic component)
    ///
    /// Default is 0.3 (30% phonetic, 70% orthographic).
    ///
    /// # Arguments
    ///
    /// * `weight` - Phonetic weight in range [0.0, 1.0]
    ///
    /// # Panics
    ///
    /// Panics if weight is not in [0.0, 1.0].
    pub fn with_phonetic_weight(mut self, weight: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&weight),
            "Phonetic weight must be in [0.0, 1.0], got {}",
            weight
        );
        self.phonetic_weight = weight;
        self
    }

    /// Set maximum cache size for normalized strings.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }

    /// Get the phonetic weight.
    #[inline]
    pub fn phonetic_weight(&self) -> f64 {
        self.phonetic_weight
    }

    /// Get a reference to the underlying orthographic embedding.
    #[inline]
    pub fn orthographic(&self) -> &SubwordEmbedding {
        &self.orthographic
    }

    /// Get the phonetic rules.
    #[inline]
    pub fn rules(&self) -> &[RewriteRuleChar] {
        &self.rules
    }

    /// Get embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.orthographic.dim()
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.orthographic.vocab_size()
    }

    /// Check if word is in vocabulary.
    #[inline]
    pub fn contains(&self, word: &str) -> bool {
        self.orthographic.contains(word)
    }

    /// Normalize a word using phonetic rules.
    ///
    /// Uses the online streaming transducer for efficient character-by-character
    /// processing with proper context handling.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to normalize
    ///
    /// # Returns
    ///
    /// The phonetically normalized form of the word.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let normalized = phonetic.normalize("enough");
    /// assert_eq!(normalized, "enuf");
    /// ```
    pub fn normalize(&self, word: &str) -> String {
        // Check cache first
        if let Some(cached) = self.normalization_cache.get(word) {
            return cached.clone();
        }

        // Use streaming transducer for normalization
        let mut transducer = OnlinePhoneticTransducerChar::new(self.rules.clone());
        let mut result = String::with_capacity(word.len());

        for c in word.chars() {
            for normalized_char in transducer.feed(c) {
                result.push(normalized_char);
            }
        }

        // Flush remaining buffer
        for c in transducer.finish() {
            result.push(c);
        }

        // Cache the result
        if self.normalization_cache.len() < self.max_cache_size {
            self.normalization_cache
                .insert(word.to_string(), result.clone());
        }

        result
    }

    /// Compute combined similarity between two words.
    ///
    /// Combines orthographic and phonetic similarity using the configured weight:
    /// ```text
    /// similarity = (1 - weight) * ortho_sim + weight * phonetic_sim
    /// ```
    ///
    /// # Arguments
    ///
    /// * `word1` - First word
    /// * `word2` - Second word
    ///
    /// # Returns
    ///
    /// Combined similarity score in range [-1.0, 1.0] (cosine similarity).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Words with same phonetic form should have high similarity
    /// let sim = phonetic.similarity("phone", "fone");
    /// assert!(sim > 0.9);
    /// ```
    pub fn similarity(&self, word1: &str, word2: &str) -> f64 {
        // Fast path: identical words
        if word1 == word2 {
            return 1.0;
        }

        // Compute orthographic similarity
        let ortho_sim = self.orthographic.similarity(word1, word2) as f64;

        // If pure orthographic, skip phonetic computation
        if self.phonetic_weight == 0.0 {
            return ortho_sim;
        }

        // Normalize both words phonetically
        let norm1 = self.normalize(word1);
        let norm2 = self.normalize(word2);

        // Compute phonetic similarity
        let phone_sim = if norm1 == norm2 {
            // Identical phonetic forms → maximum similarity
            1.0
        } else {
            // Compute embedding similarity of normalized forms
            self.orthographic.similarity(&norm1, &norm2) as f64
        };

        // Combine with weighted average
        (1.0 - self.phonetic_weight) * ortho_sim + self.phonetic_weight * phone_sim
    }

    /// Compute pure phonetic similarity (ignoring orthographic component).
    ///
    /// Normalizes both words and computes embedding similarity of the normalized forms.
    ///
    /// # Arguments
    ///
    /// * `word1` - First word
    /// * `word2` - Second word
    ///
    /// # Returns
    ///
    /// Phonetic similarity score in range [-1.0, 1.0].
    pub fn phonetic_similarity(&self, word1: &str, word2: &str) -> f64 {
        let norm1 = self.normalize(word1);
        let norm2 = self.normalize(word2);

        if norm1 == norm2 {
            1.0
        } else {
            self.orthographic.similarity(&norm1, &norm2) as f64
        }
    }

    /// Find most similar words using combined similarity.
    ///
    /// # Arguments
    ///
    /// * `word` - Query word
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of (word, similarity) pairs sorted by descending similarity.
    pub fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f64)> {
        // Get orthographic candidates (larger pool since we'll rerank)
        let candidates = self.orthographic.most_similar(word, k * 2);

        // Rerank with combined similarity
        let mut scored: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|(w, _)| {
                let sim = self.similarity(word, &w);
                (w, sim)
            })
            .collect();

        // Sort by combined similarity
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.truncate(k);
        scored
    }

    /// Find most similar words using pure phonetic similarity.
    ///
    /// Useful for finding homophones and near-homophones.
    ///
    /// # Arguments
    ///
    /// * `word` - Query word
    /// * `k` - Number of results to return
    ///
    /// # Returns
    ///
    /// Vector of (word, similarity) pairs sorted by descending phonetic similarity.
    pub fn most_similar_phonetically(&self, word: &str, k: usize) -> Vec<(String, f64)> {
        // Normalize query
        let normalized_query = self.normalize(word);

        // Get orthographic candidates from normalized form
        let candidates = self.orthographic.most_similar(&normalized_query, k * 3);

        // Rerank by phonetic similarity
        let mut scored: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|(w, _)| {
                let sim = self.phonetic_similarity(word, &w);
                (w, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.truncate(k);
        scored
    }

    /// Clear the normalization cache.
    pub fn clear_cache(&self) {
        self.normalization_cache.clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.normalization_cache.len()
    }
}

impl Clone for PhoneticEmbedding {
    fn clone(&self) -> Self {
        Self {
            orthographic: Arc::clone(&self.orthographic),
            rules: self.rules.clone(),
            phonetic_weight: self.phonetic_weight,
            normalization_cache: DashMap::new(), // Don't clone cache
            max_cache_size: self.max_cache_size,
        }
    }
}

// Ensure thread safety
unsafe impl Send for PhoneticEmbedding {}
unsafe impl Sync for PhoneticEmbedding {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::SubwordEmbedding;

    fn create_test_orthographic() -> SubwordEmbedding {
        let vocab = vec![
            "phone".to_string(),
            "fone".to_string(),
            "enough".to_string(),
            "enuf".to_string(),
            "knight".to_string(),
            "night".to_string(),
            "know".to_string(),
            "no".to_string(),
        ];
        let mut model = SubwordEmbedding::new(vocab, 50, 10000);

        // Set up embeddings such that similar words have similar vectors
        // This is simplified for testing
        let embeddings = model.word_embeddings_mut();

        // phone and fone (should be phonetically similar)
        embeddings[[0, 0]] = 1.0;
        embeddings[[0, 1]] = 0.5;
        embeddings[[1, 0]] = 1.0;
        embeddings[[1, 1]] = 0.5;

        // enough and enuf (should be phonetically similar)
        embeddings[[2, 2]] = 1.0;
        embeddings[[2, 3]] = 0.5;
        embeddings[[3, 2]] = 1.0;
        embeddings[[3, 3]] = 0.5;

        // knight and night (should be phonetically similar)
        embeddings[[4, 4]] = 1.0;
        embeddings[[4, 5]] = 0.5;
        embeddings[[5, 4]] = 1.0;
        embeddings[[5, 5]] = 0.5;

        // know and no (should be phonetically similar)
        embeddings[[6, 6]] = 1.0;
        embeddings[[6, 7]] = 0.5;
        embeddings[[7, 6]] = 1.0;
        embeddings[[7, 7]] = 0.5;

        model
    }

    #[test]
    fn test_phonetic_embedding_creation() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        assert_eq!(phonetic.dim(), 50);
        assert_eq!(phonetic.vocab_size(), 8);
        assert!(!phonetic.rules().is_empty());
    }

    #[test]
    fn test_phonetic_weight() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho).with_phonetic_weight(0.5);

        assert_eq!(phonetic.phonetic_weight(), 0.5);
    }

    #[test]
    #[should_panic(expected = "Phonetic weight must be in [0.0, 1.0]")]
    fn test_invalid_phonetic_weight() {
        let ortho = create_test_orthographic();
        let _ = PhoneticEmbedding::new(ortho).with_phonetic_weight(1.5);
    }

    #[test]
    fn test_normalize() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        // Test phonetic normalization
        // Note: Actual normalization depends on the Zompist rules
        let norm = phonetic.normalize("phone");
        // The normalization should be deterministic
        assert_eq!(norm, phonetic.normalize("phone"));
    }

    #[test]
    fn test_normalization_cache() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        // First call should populate cache
        let _ = phonetic.normalize("phone");
        assert_eq!(phonetic.cache_size(), 1);

        // Second call should use cache
        let _ = phonetic.normalize("phone");
        assert_eq!(phonetic.cache_size(), 1);

        // Different word should add to cache
        let _ = phonetic.normalize("enough");
        assert_eq!(phonetic.cache_size(), 2);

        // Clear cache
        phonetic.clear_cache();
        assert_eq!(phonetic.cache_size(), 0);
    }

    #[test]
    fn test_self_similarity() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        // Self-similarity should be 1.0
        assert_eq!(phonetic.similarity("phone", "phone"), 1.0);
        assert_eq!(phonetic.similarity("enough", "enough"), 1.0);
    }

    #[test]
    fn test_phonetic_similarity_identical_normalized() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        // Words that normalize to the same form should have phonetic similarity 1.0
        let norm1 = phonetic.normalize("phone");
        let norm2 = phonetic.normalize("phone");
        assert_eq!(norm1, norm2);

        let phone_sim = phonetic.phonetic_similarity("phone", "phone");
        assert_eq!(phone_sim, 1.0);
    }

    #[test]
    fn test_pure_orthographic_mode() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho.clone()).with_phonetic_weight(0.0);

        // With weight 0, should equal orthographic similarity
        let ortho_sim = ortho.similarity("phone", "fone");
        let combined_sim = phonetic.similarity("phone", "fone");

        assert!((ortho_sim as f64 - combined_sim).abs() < 1e-6);
    }

    #[test]
    fn test_clone() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho).with_phonetic_weight(0.5);

        // Populate cache
        let _ = phonetic.normalize("phone");

        // Clone should not share cache
        let cloned = phonetic.clone();
        assert_eq!(cloned.phonetic_weight(), 0.5);
        assert_eq!(cloned.cache_size(), 0); // Cache not cloned
    }

    #[test]
    fn test_most_similar() {
        let ortho = create_test_orthographic();
        let phonetic = PhoneticEmbedding::new(ortho);

        let similar = phonetic.most_similar("phone", 3);
        assert!(!similar.is_empty());
        assert!(similar.len() <= 3);
    }
}
