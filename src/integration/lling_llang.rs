//! lling-llang LanguageModel trait implementation.
//!
//! This module provides implementations of lling-llang's `LanguageModel` trait
//! for libgrammstein models, enabling their use in WFST-based text correction.

use crate::embedding::SubwordEmbedding;
use crate::hybrid::{HybridConfig, HybridLanguageModel};
use crate::ngram::{NgramEntry, NgramModel};
use liblevenshtein::dictionary::MutableMappedDictionary;
use lling_llang::layers::LanguageModel;
use std::sync::Arc;

/// Wrapper providing lling-llang LanguageModel trait implementation.
///
/// This enables libgrammstein models to be used with lling-llang's
/// `LanguageModelLayer` for lattice rescoring.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::integration::GrammsteinLanguageModel;
/// use libgrammstein::ngram::NgramModel;
/// use lling_llang::layers::{LanguageModel, LanguageModelLayer};
///
/// // Create from n-gram model
/// let ngram = NgramModel::load("model.bin")?;
/// let lm = GrammsteinLanguageModel::from_ngram(ngram);
///
/// // Use in lling-llang pipeline
/// let layer = LanguageModelLayer::new(Box::new(lm));
/// ```
pub enum GrammsteinLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// Pure n-gram language model.
    Ngram(Arc<NgramModel<D>>),

    /// Hybrid model combining n-gram and embeddings.
    Hybrid(Arc<HybridLanguageModel<D>>),
}

impl<D> GrammsteinLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// Create from an n-gram model.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lm = GrammsteinLanguageModel::from_ngram(ngram_model);
    /// ```
    pub fn from_ngram(model: NgramModel<D>) -> Self {
        Self::Ngram(Arc::new(model))
    }

    /// Create from an Arc-wrapped n-gram model.
    pub fn from_ngram_arc(model: Arc<NgramModel<D>>) -> Self {
        Self::Ngram(model)
    }

    /// Create from a hybrid model.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hybrid = HybridLanguageModel::new(ngram, embedding, config);
    /// let lm = GrammsteinLanguageModel::from_hybrid(hybrid);
    /// ```
    pub fn from_hybrid(model: HybridLanguageModel<D>) -> Self {
        Self::Hybrid(Arc::new(model))
    }

    /// Create from an Arc-wrapped hybrid model.
    pub fn from_hybrid_arc(model: Arc<HybridLanguageModel<D>>) -> Self {
        Self::Hybrid(model)
    }

    /// Create a hybrid model from separate n-gram and embedding models.
    ///
    /// Uses default hybrid configuration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lm = GrammsteinLanguageModel::from_components(ngram, embedding);
    /// ```
    pub fn from_components(ngram: NgramModel<D>, embedding: SubwordEmbedding) -> Self {
        Self::Hybrid(Arc::new(HybridLanguageModel::with_defaults(ngram, embedding)))
    }

    /// Create a hybrid model with custom configuration.
    pub fn from_components_with_config(
        ngram: NgramModel<D>,
        embedding: SubwordEmbedding,
        config: HybridConfig,
    ) -> Self {
        Self::Hybrid(Arc::new(HybridLanguageModel::new(ngram, embedding, config)))
    }

    /// Check if this is a hybrid model.
    pub fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    /// Get reference to the n-gram model (if using pure n-gram mode).
    pub fn ngram_model(&self) -> Option<&NgramModel<D>> {
        match self {
            Self::Ngram(model) => Some(model),
            Self::Hybrid(model) => Some(model.ngram_model()),
        }
    }

    /// Get reference to the hybrid model (if using hybrid mode).
    pub fn hybrid_model(&self) -> Option<&HybridLanguageModel<D>> {
        match self {
            Self::Ngram(_) => None,
            Self::Hybrid(model) => Some(model),
        }
    }
}

impl<D> LanguageModel for GrammsteinLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    fn score_sequence(&self, tokens: &[&str]) -> f64 {
        match self {
            Self::Ngram(model) => model.sentence_log_prob(tokens),
            Self::Hybrid(model) => model.sentence_log_prob(tokens),
        }
    }

    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64 {
        match self {
            Self::Ngram(model) => model.log_prob(next, prefix),
            Self::Hybrid(model) => model.score(next, prefix),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Ngram(model) => model.vocab_size(),
            Self::Hybrid(model) => model.ngram_model().vocab_size(),
        }
    }
}

// Implement Clone for sharing across threads
impl<D> Clone for GrammsteinLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    fn clone(&self) -> Self {
        match self {
            Self::Ngram(model) => Self::Ngram(Arc::clone(model)),
            Self::Hybrid(model) => Self::Hybrid(Arc::clone(model)),
        }
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

    fn create_test_models() -> (NgramModel<PathMapDictionary<NgramEntry>>, SubwordEmbedding) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        // Train n-gram model
        let dictionary = PathMapDictionary::<NgramEntry>::new();
        let ngram_model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(&reader)
            .expect("N-gram training failed");

        // Train embedding model
        let reader2 = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let embedding_model = EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(2)
            .train(&reader2)
            .expect("Embedding training failed");

        (ngram_model, embedding_model)
    }

    #[test]
    fn test_ngram_language_model() {
        let (ngram, _embedding) = create_test_models();
        let lm = GrammsteinLanguageModel::from_ngram(ngram);

        // Test LanguageModel trait methods
        let seq_score = lm.score_sequence(&["the", "quick", "brown"]);
        assert!(seq_score.is_finite());
        assert!(seq_score < 0.0); // Log probability should be negative

        let cont_score = lm.score_continuation(&["the", "quick"], "brown");
        assert!(cont_score.is_finite());

        assert!(lm.vocab_size() > 0);
    }

    #[test]
    fn test_hybrid_language_model() {
        let (ngram, embedding) = create_test_models();
        let lm = GrammsteinLanguageModel::from_components(ngram, embedding);

        assert!(lm.is_hybrid());

        // Test LanguageModel trait methods
        let seq_score = lm.score_sequence(&["the", "quick", "brown"]);
        assert!(seq_score.is_finite());

        let cont_score = lm.score_continuation(&["the", "quick"], "brown");
        assert!(cont_score.is_finite());
    }

    #[test]
    fn test_clone() {
        let (ngram, _embedding) = create_test_models();
        let lm1 = GrammsteinLanguageModel::from_ngram(ngram);
        let lm2 = lm1.clone();

        // Both should produce the same scores
        let score1 = lm1.score_sequence(&["the", "quick"]);
        let score2 = lm2.score_sequence(&["the", "quick"]);
        assert_eq!(score1, score2);
    }

    #[test]
    fn test_model_access() {
        let (ngram, embedding) = create_test_models();

        // N-gram mode
        let ngram_lm = GrammsteinLanguageModel::from_ngram(ngram.clone());
        assert!(ngram_lm.ngram_model().is_some());
        assert!(ngram_lm.hybrid_model().is_none());
        assert!(!ngram_lm.is_hybrid());

        // Hybrid mode
        let hybrid_lm = GrammsteinLanguageModel::from_components(ngram, embedding);
        assert!(hybrid_lm.ngram_model().is_some());
        assert!(hybrid_lm.hybrid_model().is_some());
        assert!(hybrid_lm.is_hybrid());
    }
}
