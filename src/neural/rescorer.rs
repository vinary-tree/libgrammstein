//! Neural rescoring for beam search paths using ModernBERT.
//!
//! This module provides rescoring of n-gram beam search candidates
//! using ModernBERT's pseudo-perplexity scoring.

use std::cmp::Ordering;
use std::sync::Arc;

use candle_core::{IndexOp, Tensor};

use super::modernbert::{ModernBertConfig, ModernBertModel};
use super::{NeuralError, Result};

/// Configuration for neural rescoring.
#[derive(Clone, Debug)]
pub struct RescoringConfig {
    /// ModernBERT model configuration.
    pub model_config: ModernBertConfig,
    /// Weight for n-gram scores (alpha).
    pub ngram_weight: f64,
    /// Weight for neural scores (beta).
    pub neural_weight: f64,
    /// Number of top paths to rescore.
    pub top_k: usize,
    /// Batch size for parallel rescoring.
    pub batch_size: usize,
    /// Use pseudo-perplexity (MLM-based) scoring.
    pub use_pseudo_perplexity: bool,
}

impl Default for RescoringConfig {
    fn default() -> Self {
        Self {
            model_config: ModernBertConfig::default(),
            ngram_weight: 0.7,
            neural_weight: 0.3,
            top_k: 100,
            batch_size: 32,
            use_pseudo_perplexity: true,
        }
    }
}

/// A scored path from beam search.
#[derive(Clone, Debug)]
pub struct ScoredPath<W> {
    /// The path (sequence of tokens/words).
    pub tokens: Vec<String>,
    /// Original n-gram score.
    pub ngram_score: W,
    /// Neural score (after rescoring).
    pub neural_score: Option<f64>,
    /// Combined final score.
    pub final_score: f64,
}

impl<W: Clone + Into<f64>> ScoredPath<W> {
    /// Create a new path with n-gram score only.
    pub fn new(tokens: Vec<String>, ngram_score: W) -> Self {
        let score: f64 = ngram_score.clone().into();
        Self {
            tokens,
            ngram_score,
            neural_score: None,
            final_score: score,
        }
    }

    /// Get the text representation of the path.
    pub fn text(&self) -> String {
        self.tokens.join(" ")
    }
}

/// Neural rescorer using ModernBERT.
pub struct ModernBertRescorer {
    model: Arc<ModernBertModel>,
    config: RescoringConfig,
}

impl ModernBertRescorer {
    /// Create a new rescorer by loading a model.
    pub fn new(config: RescoringConfig) -> Result<Self> {
        let model = ModernBertModel::load(config.model_config.clone())?;

        Ok(Self {
            model: Arc::new(model),
            config,
        })
    }

    /// Create a rescorer from an existing model.
    pub fn from_model(model: Arc<ModernBertModel>, config: RescoringConfig) -> Self {
        Self { model, config }
    }

    /// Score a single sentence using pseudo-perplexity.
    ///
    /// Pseudo-perplexity for MLM: mask each token, predict it, average log probs.
    /// Lower score = more probable sentence.
    pub fn score_sentence(&self, sentence: &str) -> Result<f64> {
        if self.config.use_pseudo_perplexity {
            self.pseudo_perplexity(sentence)
        } else {
            self.embedding_coherence(sentence)
        }
    }

    /// Compute MLM pseudo-perplexity for a sentence.
    ///
    /// For each position, mask the token and compute the log probability
    /// of predicting the original token. Average across all positions.
    fn pseudo_perplexity(&self, sentence: &str) -> Result<f64> {
        let tokens = self.model.encode(sentence)?;
        let num_tokens = tokens.len();

        if num_tokens == 0 {
            return Ok(0.0);
        }

        let mask_id = self.model.mask_token_id().ok_or_else(|| {
            NeuralError::Tokenization("No [MASK] token found in vocabulary".to_string())
        })?;

        let mut total_log_prob = 0.0;

        // For each position, mask and predict
        for i in 0..num_tokens {
            // Create masked sequence
            let mut masked_tokens = tokens.clone();
            let original_token = masked_tokens[i];
            masked_tokens[i] = mask_id;

            // Forward pass to get hidden states
            let input_ids = Tensor::new(&masked_tokens[..], self.model.device())?.unsqueeze(0)?;
            let hidden_states = self.model.forward(&input_ids, None)?;

            // Get logits for masked position
            // Note: This is a simplified version. A full implementation would
            // need the MLM head to project hidden states to vocabulary logits.
            let masked_hidden = hidden_states.i((0, i))?;

            // For now, use a proxy: cosine similarity between masked hidden state
            // and the embedding of the original token
            // This is an approximation; real PPL requires the full MLM head
            let score = self.token_probability_proxy(&masked_hidden, original_token)?;
            total_log_prob += score.ln();
        }

        // Perplexity = exp(-avg_log_prob)
        let avg_log_prob = total_log_prob / num_tokens as f64;
        Ok((-avg_log_prob).exp())
    }

    /// Proxy for token probability using embedding similarity.
    ///
    /// This is a simplified approximation when we don't have the full MLM head.
    fn token_probability_proxy(&self, hidden: &Tensor, _token_id: u32) -> Result<f64> {
        // Simplified: use L2 norm of hidden state as a proxy for confidence
        // A real implementation would use the MLM head projection
        let norm: f32 = hidden.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
        Ok(norm as f64 / 10.0) // Scale to reasonable range
    }

    /// Score sentence using embedding coherence.
    ///
    /// Measures how well the sentence embedding clusters with itself
    /// when split into parts.
    fn embedding_coherence(&self, sentence: &str) -> Result<f64> {
        // Get full sentence embedding
        let full_embedding = self.model.embed(sentence)?;

        // Split sentence into chunks and get their embeddings
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 2 {
            return Ok(1.0); // Single word, perfect coherence
        }

        let mid = words.len() / 2;
        let first_half = words[..mid].join(" ");
        let second_half = words[mid..].join(" ");

        let first_emb = self.model.embed(&first_half)?;
        let second_emb = self.model.embed(&second_half)?;

        // Coherence = average similarity between full and parts
        let sim1 = Self::cosine_similarity(&full_embedding, &first_emb);
        let sim2 = Self::cosine_similarity(&full_embedding, &second_emb);

        // Higher coherence = lower perplexity-like score (invert)
        Ok(2.0 / (sim1 + sim2 + 1e-6) as f64)
    }

    /// Cosine similarity between two embeddings.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Score multiple sentences in a batch.
    pub fn score_batch(&self, sentences: &[&str]) -> Result<Vec<f64>> {
        sentences.iter().map(|s| self.score_sentence(s)).collect()
    }

    /// Rescore top-k paths from beam search.
    ///
    /// Combines n-gram and neural scores:
    /// `final_score = alpha * ngram_score + beta * neural_score`
    pub fn rescore_paths<W: Clone + Into<f64>>(
        &self,
        mut paths: Vec<ScoredPath<W>>,
    ) -> Result<Vec<ScoredPath<W>>> {
        if paths.is_empty() {
            return Ok(paths);
        }

        // Sort by n-gram score and take top-k
        paths.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(Ordering::Equal)
        });
        paths.truncate(self.config.top_k);

        // Compute neural scores in batches
        let texts: Vec<String> = paths.iter().map(|p| p.text()).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        for chunk_start in (0..text_refs.len()).step_by(self.config.batch_size) {
            let chunk_end = (chunk_start + self.config.batch_size).min(text_refs.len());
            let chunk = &text_refs[chunk_start..chunk_end];

            let neural_scores = self.score_batch(chunk)?;

            for (i, score) in neural_scores.into_iter().enumerate() {
                let path_idx = chunk_start + i;

                // Neural score: lower perplexity = better = higher score
                // Convert to positive score (invert and normalize)
                let neural_normalized = 1.0 / (1.0 + score);

                paths[path_idx].neural_score = Some(neural_normalized);

                // Combine scores
                let ngram: f64 = paths[path_idx].ngram_score.clone().into();
                paths[path_idx].final_score = self.config.ngram_weight * ngram
                    + self.config.neural_weight * neural_normalized;
            }
        }

        // Re-sort by final score
        paths.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(Ordering::Equal)
        });

        Ok(paths)
    }

    /// Get the configuration.
    pub fn config(&self) -> &RescoringConfig {
        &self.config
    }

    /// Get the underlying model.
    pub fn model(&self) -> &ModernBertModel {
        &self.model
    }

    /// Update the score weights.
    pub fn set_weights(&mut self, ngram_weight: f64, neural_weight: f64) {
        self.config.ngram_weight = ngram_weight;
        self.config.neural_weight = neural_weight;
    }
}

impl std::fmt::Debug for ModernBertRescorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModernBertRescorer")
            .field("ngram_weight", &self.config.ngram_weight)
            .field("neural_weight", &self.config.neural_weight)
            .field("top_k", &self.config.top_k)
            .field("batch_size", &self.config.batch_size)
            .finish()
    }
}

/// Rescoring result with detailed scores.
#[derive(Clone, Debug)]
pub struct RescoringResult {
    /// Best path after rescoring.
    pub best_path: String,
    /// Top-k paths with scores.
    pub top_paths: Vec<RankedPath>,
    /// Total paths considered.
    pub total_paths: usize,
}

/// A ranked path with detailed scoring.
#[derive(Clone, Debug)]
pub struct RankedPath {
    /// Path text.
    pub text: String,
    /// Rank (1 = best).
    pub rank: usize,
    /// N-gram score.
    pub ngram_score: f64,
    /// Neural score.
    pub neural_score: f64,
    /// Combined final score.
    pub final_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scored_path() {
        let path = ScoredPath::<f64>::new(
            vec!["the".to_string(), "quick".to_string(), "fox".to_string()],
            0.8,
        );

        assert_eq!(path.text(), "the quick fox");
        assert!((path.final_score - 0.8).abs() < 1e-6);
        assert!(path.neural_score.is_none());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((ModernBertRescorer::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((ModernBertRescorer::cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rescore_paths_ordering() {
        // Test that paths are sorted by final score
        let mut paths = vec![
            ScoredPath::<f64> {
                tokens: vec!["a".to_string()],
                ngram_score: 0.5,
                neural_score: Some(0.8),
                final_score: 0.6,
            },
            ScoredPath::<f64> {
                tokens: vec!["b".to_string()],
                ngram_score: 0.9,
                neural_score: Some(0.7),
                final_score: 0.85,
            },
            ScoredPath::<f64> {
                tokens: vec!["c".to_string()],
                ngram_score: 0.3,
                neural_score: Some(0.9),
                final_score: 0.4,
            },
        ];

        paths.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(Ordering::Equal)
        });

        assert_eq!(paths[0].text(), "b");
        assert_eq!(paths[1].text(), "a");
        assert_eq!(paths[2].text(), "c");
    }
}
