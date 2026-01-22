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

#[allow(deprecated)]
use crate::ngram::{IterableDictionary, NgramEntry, NgramModel, NGRAM_SEPARATOR};
use liblevenshtein::dictionary::MutableMappedDictionary;
use rand::distributions::WeightedIndex;
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
pub struct TextGenerator<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync,
{
    /// The language model.
    model: Arc<NgramModel<D>>,

    /// Generation configuration.
    config: GenerationConfig,

    /// Cached vocabulary (unigrams from the model).
    vocabulary: Vec<String>,
}

impl<D> TextGenerator<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync,
{
    /// Create a new text generator.
    pub fn new(model: NgramModel<D>, config: GenerationConfig) -> Self {
        let vocabulary = Self::extract_vocabulary(&model);
        Self {
            model: Arc::new(model),
            config,
            vocabulary,
        }
    }

    /// Create from an Arc-wrapped model.
    pub fn from_arc(model: Arc<NgramModel<D>>, config: GenerationConfig) -> Self {
        let vocabulary = Self::extract_vocabulary(&model);
        Self {
            model,
            config,
            vocabulary,
        }
    }

    /// Extract vocabulary (unigrams) from the model.
    #[allow(deprecated)]
    fn extract_vocabulary(model: &NgramModel<D>) -> Vec<String> {
        let mut vocab: HashSet<String> = HashSet::new();

        for (key, _) in model.trie().iter_entries() {
            // Check if this is a unigram (no separator)
            if !key.contains(NGRAM_SEPARATOR) {
                vocab.insert(key);
            }
        }

        vocab.into_iter().collect()
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
        let order = self.model.order();

        for _ in 0..self.config.max_tokens {
            // Get context window (last n-1 tokens)
            let ctx_start = context.len().saturating_sub(order - 1);
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
            None => Box::new(rand::thread_rng()),
        };

        let mut context: Vec<String> = prompt.iter().map(|s| s.to_string()).collect();
        let mut generated = Vec::new();
        let order = self.model.order();

        for _ in 0..self.config.max_tokens {
            // Get context window (last n-1 tokens)
            let ctx_start = context.len().saturating_sub(order - 1);
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
    use crate::ngram::TrainerBuilder;
    use liblevenshtein::dictionary::pathmap::PathMapDictionary;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_model() -> NgramModel<PathMapDictionary<NgramEntry>> {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox jumps over the lazy dog. \
                       the quick brown fox runs in the park. \
                       the lazy dog sleeps under the tree.";
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
        let model = create_test_model();
        let config = GenerationConfig::greedy()
            .with_max_tokens(100)
            .with_stop_tokens(vec![".".to_string()]);
        let generator = TextGenerator::new(model, config);

        let result = generator.generate(&["the"]);
        // Should stop at or before reaching max_tokens due to stop token
        assert!(result.len() < 100 || result.last() == Some(&".".to_string()));
    }

    #[test]
    fn test_vocabulary_extraction() {
        let model = create_test_model();
        let config = GenerationConfig::default();
        let generator = TextGenerator::new(model, config);

        assert!(generator.vocab_size() > 0);
    }
}
