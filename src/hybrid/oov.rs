//! Out-of-vocabulary (OOV) handling strategies.
//!
//! Provides various approaches for handling words not seen during training:
//! - Embedding-based similarity
//! - Subword fallback
//! - Unknown word probability estimation

use crate::embedding::SubwordEmbedding;

/// OOV handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OovStrategy {
    /// Use subword embeddings to estimate probability.
    #[default]
    SubwordEmbedding,

    /// Use a fixed unknown word probability.
    FixedProbability {
        /// Fixed log probability for OOV words.
        log_prob: f64,
    },

    /// Use uniform probability over vocabulary.
    Uniform,

    /// Estimate from similar in-vocabulary words.
    SimilarWords {
        /// Number of similar words to consider.
        k: usize,
    },
}

/// OOV handler using embedding-based strategies.
pub struct OovHandler<'a> {
    /// Embedding model for similarity computation.
    embedding: &'a SubwordEmbedding,

    /// OOV strategy.
    strategy: OovStrategy,

    /// Vocabulary size (for uniform distribution).
    vocab_size: usize,
}

impl<'a> OovHandler<'a> {
    /// Create a new OOV handler.
    pub fn new(embedding: &'a SubwordEmbedding, strategy: OovStrategy) -> Self {
        let vocab_size = embedding.vocab_size();
        Self {
            embedding,
            strategy,
            vocab_size,
        }
    }

    /// Estimate log probability for an OOV word.
    pub fn estimate_log_prob(&self, word: &str, context: &[&str]) -> f64 {
        match self.strategy {
            OovStrategy::SubwordEmbedding => self.estimate_from_subwords(word, context),
            OovStrategy::FixedProbability { log_prob } => log_prob,
            OovStrategy::Uniform => -(self.vocab_size as f64).ln(),
            OovStrategy::SimilarWords { k } => self.estimate_from_similar(word, context, k),
        }
    }

    /// Estimate probability from subword embeddings.
    fn estimate_from_subwords(&self, word: &str, context: &[&str]) -> f64 {
        // Get word vector (composed from subwords for OOV)
        let word_vec = self.embedding.word_vector(word);

        // If no context, use uniform
        if context.is_empty() {
            return -(self.vocab_size as f64).ln();
        }

        // Get context vector
        let context_vec = self.embedding.sentence_vector(context);

        // Compute similarity
        let similarity = Self::cosine_similarity(&word_vec, &context_vec);

        // Convert to log probability (simplified)
        // Higher similarity = higher probability
        let log_prob = (similarity as f64) - 1.0;

        // Ensure reasonable bounds
        log_prob.clamp(-20.0, -1e-6)
    }

    /// Estimate probability from similar in-vocabulary words.
    fn estimate_from_similar(&self, word: &str, context: &[&str], k: usize) -> f64 {
        // Find similar words in vocabulary
        let similar = self.embedding.most_similar(word, k);

        if similar.is_empty() {
            return -(self.vocab_size as f64).ln();
        }

        // Average the similarities weighted by their similarity scores
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (sim_word, similarity) in &similar {
            if *similarity > 0.0 {
                // Get log prob of similar word in context (approximated)
                let sim_vec = self.embedding.word_vector(sim_word);
                let context_vec = if context.is_empty() {
                    ndarray::Array1::zeros(self.embedding.dim())
                } else {
                    self.embedding.sentence_vector(context)
                };

                let context_sim = Self::cosine_similarity(&sim_vec, &context_vec);
                weighted_sum += *similarity as f64 * context_sim as f64;
                weight_total += *similarity as f64;
            }
        }

        if weight_total > 0.0 {
            let avg_sim = weighted_sum / weight_total;
            (avg_sim - 1.0).clamp(-20.0, -1e-6)
        } else {
            -(self.vocab_size as f64).ln()
        }
    }

    /// Compute cosine similarity.
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

    /// Check if a word is OOV.
    pub fn is_oov(&self, word: &str) -> bool {
        !self.embedding.contains(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::embedding::EmbeddingTrainerBuilder;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_embedding() -> SubwordEmbedding {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        EmbeddingTrainerBuilder::new()
            .dim(10)
            .window_size(2)
            .min_count(1)
            .epochs(2)
            .train(reader)
            .expect("Training failed")
    }

    #[test]
    fn test_oov_subword_strategy() {
        let embedding = create_test_embedding();
        let handler = OovHandler::new(&embedding, OovStrategy::SubwordEmbedding);

        // OOV word
        let log_prob = handler.estimate_log_prob("unknown", &["the", "quick"]);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_oov_fixed_probability() {
        let embedding = create_test_embedding();
        let handler = OovHandler::new(
            &embedding,
            OovStrategy::FixedProbability { log_prob: -10.0 },
        );

        let log_prob = handler.estimate_log_prob("unknown", &["the"]);
        assert_eq!(log_prob, -10.0);
    }

    #[test]
    fn test_oov_uniform() {
        let embedding = create_test_embedding();
        let handler = OovHandler::new(&embedding, OovStrategy::Uniform);

        let log_prob = handler.estimate_log_prob("unknown", &[]);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_is_oov() {
        let embedding = create_test_embedding();
        let handler = OovHandler::new(&embedding, OovStrategy::default());

        // "the" should be in vocabulary
        assert!(!handler.is_oov("the"));

        // "xyz123" should be OOV
        assert!(handler.is_oov("xyz123"));
    }
}
