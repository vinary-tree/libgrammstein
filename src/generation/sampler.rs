//! Text generation via autoregressive sampling.
//!
//! This module provides text generation capabilities using trained language models.
//! It supports multiple sampling strategies:
//!
//! - **Greedy decoding**: Always select the highest probability token
//! - **Nucleus (top-p) sampling**: Sample from the smallest set with cumulative probability >= p
//! - **Top-k sampling**: Sample from the k highest probability tokens
//! - **Temperature scaling**: Adjust the sharpness of the probability distribution
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::generation::{TextGenerator, GenerationConfig};
//! use libgrammstein::hybrid::HybridLanguageModel;
//!
//! let model = HybridLanguageModel::load("model.bin")?;
//! let generator = TextGenerator::new(model, GenerationConfig::default());
//!
//! let text = generator.generate(&["the", "quick"]);
//! println!("Generated: {}", text.join(" "));
//! ```

use crate::ngram::store::IterableNgramStore;
use crate::ngram::NgramModel;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashSet;
use std::sync::Arc;

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,

    /// Temperature for sampling (higher = more random, lower = more deterministic).
    /// 1.0 is neutral, < 1.0 sharpens, > 1.0 flattens the distribution.
    pub temperature: f64,

    /// Nucleus sampling threshold (top-p). Sample from smallest set with cumulative prob >= p.
    /// 1.0 disables nucleus sampling. 0.9 is a common value.
    pub top_p: f64,

    /// Top-k sampling. Only consider the k highest probability tokens.
    /// None disables top-k sampling.
    pub top_k: Option<usize>,

    /// Minimum probability for a token to be considered.
    pub min_prob: f64,

    /// Stop tokens that end generation.
    pub stop_tokens: Vec<String>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 50,
            temperature: 1.0,
            top_p: 0.9,
            top_k: None,
            min_prob: 1e-10,
            stop_tokens: vec![".".to_string(), "!".to_string(), "?".to_string()],
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create a new configuration for greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: Some(1),
            ..Default::default()
        }
    }

    /// Create a new configuration for nucleus sampling.
    pub fn nucleus(top_p: f64) -> Self {
        Self {
            top_p,
            ..Default::default()
        }
    }

    /// Set the maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add stop tokens.
    pub fn with_stop_tokens(mut self, tokens: Vec<String>) -> Self {
        self.stop_tokens = tokens;
        self
    }
}

/// Text generator using an n-gram language model.
///
/// Generates text autoregressively by sampling from the probability distribution
/// over next tokens given the context.
pub struct TextGenerator<S>
where
    S: IterableNgramStore + Send + Sync,
{
    /// The language model.
    model: Arc<NgramModel<S>>,

    /// Generation configuration.
    config: GenerationConfig,

    /// Cached vocabulary (unigrams from the model).
    vocabulary: Vec<String>,
}

impl<S> TextGenerator<S>
where
    S: IterableNgramStore + Send + Sync,
{
    /// Create a new text generator.
    pub fn new(model: NgramModel<S>, config: GenerationConfig) -> Self {
        let vocabulary = Self::extract_vocabulary(&model);
        Self {
            model: Arc::new(model),
            config,
            vocabulary,
        }
    }

    /// Create from an Arc-wrapped model.
    pub fn from_arc(model: Arc<NgramModel<S>>, config: GenerationConfig) -> Self {
        let vocabulary = Self::extract_vocabulary(&model);
        Self {
            model,
            config,
            vocabulary,
        }
    }

    /// Extract vocabulary (unigrams) from the model.
    ///
    /// The result is sorted so generation is *reproducible across model
    /// instances*: greedy decoding breaks score ties by vocabulary order, and the
    /// backend's raw iteration order (and term-id assignment) is not stable across
    /// independently-trained but statistically identical models. Sorting pins the
    /// tie-break to a deterministic (lexicographic) choice.
    fn extract_vocabulary(model: &NgramModel<S>) -> Vec<String> {
        let mut vocab: HashSet<String> = HashSet::new();

        for (words, _) in model.iter_ngrams() {
            // A unigram is a single-word n-gram.
            if words.len() == 1 {
                vocab.insert(words.into_iter().next().expect("unigram has one word"));
            }
        }

        let mut vocab: Vec<String> = vocab.into_iter().collect();
        vocab.sort_unstable();
        vocab
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Generate text starting from a prompt.
    ///
    /// Returns the generated tokens (not including the prompt).
    pub fn generate(&self, prompt: &[&str]) -> Vec<String> {
        match self.config.temperature {
            t if t <= 0.0 => self.generate_greedy(prompt),
            _ => self.generate_sampling(prompt),
        }
    }

    /// Generate using greedy decoding (always pick highest probability token).
    pub fn generate_greedy(&self, prompt: &[&str]) -> Vec<String> {
        let mut context: Vec<String> = prompt.iter().map(|s| s.to_string()).collect();
        let mut generated = Vec::new();
        // Hardening: `order` flows from model state; a degenerate `order == 0`
        // would make `order - 1` underflow `usize` (panic in debug). Clamp the
        // context-window width to `order.saturating_sub(1)` so the generator
        // never panics regardless of the model's reported order.
        let context_width = self.model.order().saturating_sub(1);

        for _ in 0..self.config.max_tokens {
            // Get context window (last n-1 tokens)
            let ctx_start = context.len().saturating_sub(context_width);
            let ctx: Vec<&str> = context[ctx_start..].iter().map(|s| s.as_str()).collect();

            // Find highest probability token
            let next = self.best_token(&ctx);

            if let Some(token) = next {
                // Check for stop token
                if self.config.stop_tokens.contains(&token) {
                    generated.push(token);
                    break;
                }

                context.push(token.clone());
                generated.push(token);
            } else {
                break;
            }
        }

        generated
    }

    /// Generate using sampling with temperature and nucleus/top-k filtering.
    pub fn generate_sampling(&self, prompt: &[&str]) -> Vec<String> {
        let mut rng: Box<dyn RngCore> = match self.config.seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::rng()),
        };

        let mut context: Vec<String> = prompt.iter().map(|s| s.to_string()).collect();
        let mut generated = Vec::new();
        // Hardening: clamp context-window width (see `generate_greedy`); avoids a
        // `usize` underflow panic if the model reports `order == 0`.
        let context_width = self.model.order().saturating_sub(1);

        for _ in 0..self.config.max_tokens {
            // Get context window (last n-1 tokens)
            let ctx_start = context.len().saturating_sub(context_width);
            let ctx: Vec<&str> = context[ctx_start..].iter().map(|s| s.as_str()).collect();

            // Sample next token
            let next = self.sample_token(&ctx, &mut rng);

            if let Some(token) = next {
                // Check for stop token
                if self.config.stop_tokens.contains(&token) {
                    generated.push(token);
                    break;
                }

                context.push(token.clone());
                generated.push(token);
            } else {
                break;
            }
        }

        generated
    }

    /// Find the highest probability token given context.
    fn best_token(&self, context: &[&str]) -> Option<String> {
        let mut best_token = None;
        let mut best_score = f64::NEG_INFINITY;

        for word in &self.vocabulary {
            let score = self.model.log_prob(word, context);
            if score > best_score {
                best_score = score;
                best_token = Some(word.clone());
            }
        }

        best_token
    }

    /// Sample a token from the distribution given context.
    fn sample_token(&self, context: &[&str], rng: &mut dyn RngCore) -> Option<String> {
        // Compute log probabilities for all vocabulary tokens
        let mut candidates: Vec<(String, f64)> = self
            .vocabulary
            .iter()
            .map(|word| {
                let log_prob = self.model.log_prob(word, context);
                (word.clone(), log_prob)
            })
            .filter(|(_, lp)| lp.is_finite())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Apply temperature scaling
        if self.config.temperature != 1.0 {
            let inv_temp = 1.0 / self.config.temperature;
            for (_, log_prob) in &mut candidates {
                *log_prob *= inv_temp;
            }
        }

        // Convert to probabilities
        let max_log_prob = candidates
            .iter()
            .map(|(_, lp)| *lp)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut probs: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|(word, lp)| {
                // Subtract max for numerical stability before exp
                let prob = (lp - max_log_prob).exp();
                (word, prob)
            })
            .filter(|(_, p)| *p > self.config.min_prob)
            .collect();

        if probs.is_empty() {
            return None;
        }

        // Normalize
        let total: f64 = probs.iter().map(|(_, p)| *p).sum();
        for (_, p) in &mut probs {
            *p /= total;
        }

        // Sort by probability (descending) for top-k and nucleus sampling
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k filtering
        if let Some(k) = self.config.top_k {
            probs.truncate(k);
        }

        // Apply nucleus (top-p) filtering
        if self.config.top_p < 1.0 {
            probs = self.nucleus_filter(probs);
        }

        // Re-normalize after filtering
        let total: f64 = probs.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 {
            return probs.first().map(|(w, _)| w.clone());
        }

        let weights: Vec<f64> = probs.iter().map(|(_, p)| *p / total).collect();

        // Sample from distribution
        match WeightedIndex::new(&weights) {
            Ok(dist) => {
                let idx = dist.sample(rng);
                Some(probs[idx].0.clone())
            }
            Err(_) => probs.first().map(|(w, _)| w.clone()),
        }
    }

    /// Apply nucleus (top-p) filtering.
    ///
    /// Returns the smallest set of tokens whose cumulative probability >= top_p.
    fn nucleus_filter(&self, probs: Vec<(String, f64)>) -> Vec<(String, f64)> {
        let mut cumulative = 0.0;
        let mut filtered = Vec::new();

        for (word, prob) in probs {
            cumulative += prob;
            filtered.push((word, prob));

            if cumulative >= self.config.top_p {
                break;
            }
        }

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::store::TermIdStore;
    use crate::ngram::{NgramEntry, TrainerBuilder};
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_model() -> NgramModel<TermIdStore<DynamicDawg<NgramEntry>>> {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox jumps over the lazy dog. \
                       the quick brown fox runs in the park. \
                       the lazy dog sleeps under the tree.";
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let dictionary = DynamicDawg::<NgramEntry>::new();

        TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("Training failed")
    }

    #[test]
    fn test_greedy_generation() {
        let model = create_test_model();
        let config = GenerationConfig::greedy().with_max_tokens(5);
        let generator = TextGenerator::new(model, config);

        let result = generator.generate(&["the", "quick"]);
        assert!(!result.is_empty());
        // Greedy should be deterministic
        let result2 = generator.generate(&["the", "quick"]);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_sampling_generation() {
        let model = create_test_model();
        let config = GenerationConfig::nucleus(0.9)
            .with_max_tokens(5)
            .with_seed(42);
        let generator = TextGenerator::new(model, config);

        let result = generator.generate(&["the"]);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_stop_tokens() {
        // The corpus has no standalone "." token (periods are attached, e.g. "dog."),
        // so a "." stop token can never be generated. The previous version of this test
        // only "passed" because greedy generation used to dead-end on a -inf log_prob
        // for unseen continuations; that Kneser-Ney bug is fixed, so generation now
        // continues normally to max_tokens.
        //
        // Exercise the stop-token mechanism deterministically instead: greedy generation
        // is reproducible, so discover the first token it emits, then verify that using
        // that token as a stop token terminates generation immediately.
        let baseline = TextGenerator::new(
            create_test_model(),
            GenerationConfig::greedy()
                .with_max_tokens(20)
                .with_stop_tokens(vec![]),
        )
        .generate(&["the"]);
        assert!(
            !baseline.is_empty(),
            "greedy generation should produce output"
        );

        let stop = baseline[0].clone();
        let stopped = TextGenerator::new(
            create_test_model(),
            GenerationConfig::greedy()
                .with_max_tokens(20)
                .with_stop_tokens(vec![stop.clone()]),
        )
        .generate(&["the"]);

        assert!(
            stopped.len() <= 1,
            "generation should stop at the first token when it is a stop token, got {stopped:?}"
        );
        if let Some(last) = stopped.last() {
            assert_eq!(last, &stop);
        }
    }

    #[test]
    fn test_vocabulary_extraction() {
        let model = create_test_model();
        let config = GenerationConfig::default();
        let generator = TextGenerator::new(model, config);

        assert!(generator.vocab_size() > 0);
    }
}
