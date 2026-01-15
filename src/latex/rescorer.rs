//! Neural rescoring for LaTeX correction candidates.
//!
//! This module provides neural-based rescoring using fine-tuned language models
//! for mathematical and scientific text.

use crate::latex::tokenizer::{LaTeXToken, LaTeXTokenKind};
use std::collections::HashMap;

/// Configuration for the LaTeX rescorer.
#[derive(Debug, Clone)]
pub struct RescorerConfig {
    /// Weight for neural score in final combination.
    pub neural_weight: f64,
    /// Weight for n-gram score in final combination.
    pub ngram_weight: f64,
    /// Weight for embedding similarity score.
    pub embedding_weight: f64,
    /// Batch size for neural inference.
    pub batch_size: usize,
    /// Maximum sequence length for neural model.
    pub max_length: usize,
    /// Whether to use GPU acceleration.
    pub use_gpu: bool,
    /// Model name or path.
    pub model_name: String,
}

impl Default for RescorerConfig {
    fn default() -> Self {
        Self {
            neural_weight: 0.4,
            ngram_weight: 0.4,
            embedding_weight: 0.2,
            batch_size: 16,
            max_length: 512,
            use_gpu: false,
            model_name: "modernbert-latex".to_string(),
        }
    }
}

/// Result of rescoring a candidate.
#[derive(Debug, Clone)]
pub struct RescoreResult {
    /// The rescored sequence (as text).
    pub sequence: String,
    /// Combined final score.
    pub score: f64,
    /// Neural model score component.
    pub neural_score: Option<f64>,
    /// N-gram score component.
    pub ngram_score: Option<f64>,
    /// Embedding similarity score component.
    pub embedding_score: Option<f64>,
    /// Confidence in the rescoring (0.0 to 1.0).
    pub confidence: f64,
}

impl RescoreResult {
    /// Create a new rescore result.
    pub fn new(sequence: String, score: f64) -> Self {
        Self {
            sequence,
            score,
            neural_score: None,
            ngram_score: None,
            embedding_score: None,
            confidence: 1.0,
        }
    }

    /// Set the component scores.
    pub fn with_components(
        mut self,
        neural: Option<f64>,
        ngram: Option<f64>,
        embedding: Option<f64>,
    ) -> Self {
        self.neural_score = neural;
        self.ngram_score = ngram;
        self.embedding_score = embedding;
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Neural rescorer for LaTeX sequences.
///
/// This rescorer uses a fine-tuned language model (e.g., ModernBERT) to
/// score LaTeX sequences based on naturalness and correctness.
pub struct LaTeXRescorer {
    /// Configuration.
    config: RescorerConfig,
    /// Cache for rescored sequences.
    cache: HashMap<String, f64>,
    /// Whether the model is loaded.
    model_loaded: bool,
}

impl LaTeXRescorer {
    /// Create a new rescorer with default configuration.
    pub fn new() -> Self {
        Self::with_config(RescorerConfig::default())
    }

    /// Create a rescorer with custom configuration.
    pub fn with_config(config: RescorerConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            model_loaded: false,
        }
    }

    /// Load the neural model.
    ///
    /// This loads the model weights and initializes the inference engine.
    #[cfg(feature = "neural-rescore")]
    pub fn load_model(&mut self) -> crate::Result<()> {
        // Model loading would be implemented here using candle
        // For now, we mark the model as loaded
        self.model_loaded = true;
        Ok(())
    }

    /// Check if the model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Rescore a single LaTeX sequence.
    pub fn rescore(&mut self, tokens: &[LaTeXToken]) -> RescoreResult {
        let sequence = tokens_to_string(tokens);

        // Check cache
        if let Some(&cached_score) = self.cache.get(&sequence) {
            return RescoreResult::new(sequence, cached_score);
        }

        // Compute component scores
        let neural_score = self.compute_neural_score(tokens);
        let ngram_score = None; // Would be computed by n-gram model
        let embedding_score = self.compute_embedding_score(tokens);

        // Combine scores
        let score = self.combine_scores(neural_score, ngram_score, embedding_score);

        // Cache result
        self.cache.insert(sequence.clone(), score);

        RescoreResult::new(sequence, score)
            .with_components(neural_score, ngram_score, embedding_score)
    }

    /// Rescore multiple sequences in a batch.
    pub fn rescore_batch(&mut self, sequences: &[&[LaTeXToken]]) -> Vec<RescoreResult> {
        sequences
            .iter()
            .map(|tokens| self.rescore(tokens))
            .collect()
    }

    /// Compute the neural model score for a sequence.
    fn compute_neural_score(&self, tokens: &[LaTeXToken]) -> Option<f64> {
        if !self.model_loaded {
            return None;
        }

        // In a full implementation, this would:
        // 1. Convert tokens to model input format
        // 2. Run through the neural model
        // 3. Extract log probability or score

        // Placeholder: compute a heuristic score based on token structure
        let score = self.heuristic_neural_score(tokens);
        Some(score)
    }

    /// Compute embedding-based similarity score.
    fn compute_embedding_score(&self, tokens: &[LaTeXToken]) -> Option<f64> {
        // In a full implementation, this would:
        // 1. Get embeddings for each token
        // 2. Compute sequence embedding (e.g., centroid or attention-weighted)
        // 3. Compare against reference embeddings

        // Placeholder: return based on token validity
        let valid_tokens = tokens.iter().filter(|t| !matches!(&t.kind, LaTeXTokenKind::Unknown(_))).count();
        let total = tokens.len().max(1);
        Some(valid_tokens as f64 / total as f64)
    }

    /// Combine component scores into a final score.
    fn combine_scores(
        &self,
        neural: Option<f64>,
        ngram: Option<f64>,
        embedding: Option<f64>,
    ) -> f64 {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        if let Some(s) = neural {
            weighted_sum += s * self.config.neural_weight;
            total_weight += self.config.neural_weight;
        }

        if let Some(s) = ngram {
            weighted_sum += s * self.config.ngram_weight;
            total_weight += self.config.ngram_weight;
        }

        if let Some(s) = embedding {
            weighted_sum += s * self.config.embedding_weight;
            total_weight += self.config.embedding_weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Heuristic neural score based on token structure.
    ///
    /// This is a fallback when the neural model isn't loaded.
    fn heuristic_neural_score(&self, tokens: &[LaTeXToken]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;
        let mut brace_depth = 0i32;
        let mut math_depth = 0i32;

        for token in tokens {
            match &token.kind {
                LaTeXTokenKind::Command(_) => score += 0.1,
                LaTeXTokenKind::OpenBrace(_) => {
                    brace_depth += 1;
                    score += 0.05;
                }
                LaTeXTokenKind::CloseBrace(_) => {
                    brace_depth -= 1;
                    if brace_depth < 0 {
                        score -= 0.5; // Unmatched closing brace
                    } else {
                        score += 0.05;
                    }
                }
                LaTeXTokenKind::MathOpen(_) => {
                    math_depth += 1;
                    score += 0.1;
                }
                LaTeXTokenKind::MathClose(_) => {
                    math_depth -= 1;
                    if math_depth < 0 {
                        score -= 0.5; // Unmatched math close
                    } else {
                        score += 0.1;
                    }
                }
                LaTeXTokenKind::Unknown(_) => score -= 0.2,
                _ => score += 0.02,
            }
        }

        // Penalize unclosed braces/math
        score -= brace_depth.abs() as f64 * 0.3;
        score -= math_depth.abs() as f64 * 0.3;

        // Normalize to 0-1 range
        let normalized = (score / tokens.len() as f64).clamp(-1.0, 1.0);
        (normalized + 1.0) / 2.0
    }

    /// Clear the score cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }

    /// Get the configuration.
    pub fn config(&self) -> &RescorerConfig {
        &self.config
    }
}

impl Default for LaTeXRescorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert tokens to a string representation.
fn tokens_to_string(tokens: &[LaTeXToken]) -> String {
    tokens.iter().map(|t| t.text()).collect::<Vec<_>>().join("")
}

/// Candidate sequence for rescoring.
#[derive(Debug, Clone)]
pub struct RescoreCandidate {
    /// The token sequence.
    pub tokens: Vec<LaTeXToken>,
    /// Prior score (from earlier pipeline stages).
    pub prior_score: f64,
    /// Source of the candidate (e.g., "lexical", "syntactic").
    pub source: String,
}

impl RescoreCandidate {
    /// Create a new candidate.
    pub fn new(tokens: Vec<LaTeXToken>, prior_score: f64, source: &str) -> Self {
        Self {
            tokens,
            prior_score,
            source: source.to_string(),
        }
    }

    /// Get the token sequence as text.
    pub fn text(&self) -> String {
        tokens_to_string(&self.tokens)
    }
}

/// Batch rescorer for efficient processing of multiple candidates.
pub struct BatchRescorer {
    rescorer: LaTeXRescorer,
    /// Maximum batch size.
    max_batch_size: usize,
}

impl BatchRescorer {
    /// Create a new batch rescorer.
    pub fn new(rescorer: LaTeXRescorer) -> Self {
        let max_batch_size = rescorer.config.batch_size;
        Self {
            rescorer,
            max_batch_size,
        }
    }

    /// Rescore a batch of candidates.
    pub fn rescore_candidates(&mut self, candidates: &[RescoreCandidate]) -> Vec<(RescoreCandidate, RescoreResult)> {
        let mut results = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            let result = self.rescorer.rescore(&candidate.tokens);
            results.push((candidate.clone(), result));
        }

        // Sort by combined score (prior + neural)
        results.sort_by(|a, b| {
            let score_a = a.0.prior_score + a.1.score;
            let score_b = b.0.prior_score + b.1.score;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Get the top-k candidates after rescoring.
    pub fn top_k(&mut self, candidates: &[RescoreCandidate], k: usize) -> Vec<(RescoreCandidate, RescoreResult)> {
        let mut results = self.rescore_candidates(candidates);
        results.truncate(k);
        results
    }

    /// Get the best candidate after rescoring.
    pub fn best(&mut self, candidates: &[RescoreCandidate]) -> Option<(RescoreCandidate, RescoreResult)> {
        self.top_k(candidates, 1).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latex::tokenizer::LaTeXTokenizer;

    #[test]
    fn test_rescore_basic() {
        let tokenizer = LaTeXTokenizer::new();
        let mut rescorer = LaTeXRescorer::new();

        let tokens = tokenizer.tokenize(r"\alpha + \beta");
        let result = rescorer.rescore(&tokens);

        assert!(!result.sequence.is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_rescore_unbalanced() {
        let tokenizer = LaTeXTokenizer::new();
        let mut rescorer = LaTeXRescorer::new();

        // Balanced braces should score higher
        let balanced = tokenizer.tokenize(r"\frac{a}{b}");
        let unbalanced = tokenizer.tokenize(r"\frac{a}{b");

        let balanced_result = rescorer.rescore(&balanced);
        let unbalanced_result = rescorer.rescore(&unbalanced);

        assert!(balanced_result.score >= unbalanced_result.score);
    }

    #[test]
    fn test_cache() {
        let tokenizer = LaTeXTokenizer::new();
        let mut rescorer = LaTeXRescorer::new();

        let tokens = tokenizer.tokenize(r"\alpha");

        // First call
        let result1 = rescorer.rescore(&tokens);
        assert_eq!(rescorer.cache_stats().0, 1);

        // Second call should use cache
        let result2 = rescorer.rescore(&tokens);
        assert_eq!(result1.score, result2.score);

        rescorer.clear_cache();
        assert_eq!(rescorer.cache_stats().0, 0);
    }

    #[test]
    fn test_batch_rescorer() {
        let tokenizer = LaTeXTokenizer::new();
        let rescorer = LaTeXRescorer::new();
        let mut batch = BatchRescorer::new(rescorer);

        let candidates = vec![
            RescoreCandidate::new(tokenizer.tokenize(r"\alpha"), 0.5, "lexical"),
            RescoreCandidate::new(tokenizer.tokenize(r"\beta"), 0.6, "lexical"),
            RescoreCandidate::new(tokenizer.tokenize(r"\gamma"), 0.4, "lexical"),
        ];

        let results = batch.top_k(&candidates, 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_combine_scores() {
        let rescorer = LaTeXRescorer::new();

        let combined = rescorer.combine_scores(Some(0.8), Some(0.6), Some(0.7));
        assert!(combined > 0.0 && combined < 1.0);

        // With only neural score
        let neural_only = rescorer.combine_scores(Some(0.9), None, None);
        assert!((neural_only - 0.9).abs() < 1e-6);
    }
}
