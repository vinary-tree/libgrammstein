//! Mode-aware N-gram language model for LaTeX documents.
//!
//! This module provides n-gram models that understand the different contexts
//! within LaTeX documents: commands, math expressions, and natural text.

use crate::latex::tokenizer::{LaTeXToken, LaTeXTokenKind};
use crate::ngram::{NgramEntry, NgramModel};
use liblevenshtein::dictionary::MutableMappedDictionary;

/// Configuration for LaTeX n-gram model training and scoring.
#[derive(Debug, Clone)]
pub struct NgramConfig {
    /// N-gram order (e.g., 3 for trigrams).
    pub order: usize,
    /// Weight for command sequence scoring.
    pub command_weight: f64,
    /// Weight for math expression scoring.
    pub math_weight: f64,
    /// Weight for natural text scoring.
    pub text_weight: f64,
    /// Minimum count threshold for n-gram inclusion.
    pub min_count: u64,
    /// Whether to use separate models for each mode.
    pub mode_separation: bool,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            order: 5,
            command_weight: 1.5,
            math_weight: 2.0,
            text_weight: 1.0,
            min_count: 1,
            mode_separation: true,
        }
    }
}

/// The mode/context of a LaTeX token sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaTeXMode {
    /// LaTeX command sequences (\begin, \frac, etc.).
    Command,
    /// Mathematical expressions.
    Math,
    /// Natural language text.
    Text,
    /// Mixed or indeterminate mode.
    Mixed,
}

impl LaTeXMode {
    /// Get the weight multiplier for this mode.
    pub fn weight(&self, config: &NgramConfig) -> f64 {
        match self {
            LaTeXMode::Command => config.command_weight,
            LaTeXMode::Math => config.math_weight,
            LaTeXMode::Text => config.text_weight,
            LaTeXMode::Mixed => 1.0,
        }
    }
}

/// Mode detector for LaTeX token sequences.
#[derive(Debug, Clone, Default)]
pub struct ModeDetector {
    /// Number of consecutive command tokens to trigger command mode.
    command_threshold: usize,
    /// Number of consecutive math tokens to trigger math mode.
    math_threshold: usize,
}

impl ModeDetector {
    /// Create a new mode detector with default thresholds.
    pub fn new() -> Self {
        Self {
            command_threshold: 2,
            math_threshold: 2,
        }
    }

    /// Create a mode detector with custom thresholds.
    pub fn with_thresholds(command_threshold: usize, math_threshold: usize) -> Self {
        Self {
            command_threshold,
            math_threshold,
        }
    }

    /// Detect the mode of a single token.
    pub fn token_mode(&self, token: &LaTeXToken) -> LaTeXMode {
        if token.in_math {
            LaTeXMode::Math
        } else {
            match &token.kind {
                LaTeXTokenKind::Command(_) => LaTeXMode::Command,
                LaTeXTokenKind::Environment(_) => LaTeXMode::Command,
                LaTeXTokenKind::Text(_) => LaTeXMode::Text,
                LaTeXTokenKind::MathOpen(_) | LaTeXTokenKind::MathClose(_) => LaTeXMode::Math,
                LaTeXTokenKind::Subscript | LaTeXTokenKind::Superscript => LaTeXMode::Math,
                LaTeXTokenKind::Number(_) => LaTeXMode::Math, // Numbers often in math context
                LaTeXTokenKind::Operator(_) => LaTeXMode::Math,
                LaTeXTokenKind::Identifier(_) => LaTeXMode::Math,
                _ => LaTeXMode::Mixed,
            }
        }
    }

    /// Detect the dominant mode of a token sequence.
    pub fn sequence_mode(&self, tokens: &[LaTeXToken]) -> LaTeXMode {
        if tokens.is_empty() {
            return LaTeXMode::Mixed;
        }

        let mut command_count = 0;
        let mut math_count = 0;
        let mut text_count = 0;

        for token in tokens {
            match self.token_mode(token) {
                LaTeXMode::Command => command_count += 1,
                LaTeXMode::Math => math_count += 1,
                LaTeXMode::Text => text_count += 1,
                LaTeXMode::Mixed => {}
            }
        }

        let total = tokens.len();
        let threshold = total / 2;

        if math_count > threshold || tokens.iter().any(|t| t.in_math) {
            LaTeXMode::Math
        } else if command_count > threshold {
            LaTeXMode::Command
        } else if text_count > threshold {
            LaTeXMode::Text
        } else {
            LaTeXMode::Mixed
        }
    }

    /// Detect mode transitions in a token sequence.
    /// Returns a vector of (start_idx, end_idx, mode) tuples.
    pub fn detect_transitions(&self, tokens: &[LaTeXToken]) -> Vec<(usize, usize, LaTeXMode)> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut regions = Vec::new();
        let mut current_mode = self.token_mode(&tokens[0]);
        let mut start_idx = 0;

        for (i, token) in tokens.iter().enumerate().skip(1) {
            let token_mode = self.token_mode(token);
            if token_mode != current_mode && token_mode != LaTeXMode::Mixed {
                if i > start_idx {
                    regions.push((start_idx, i, current_mode));
                }
                current_mode = token_mode;
                start_idx = i;
            }
        }

        // Push final region
        if tokens.len() > start_idx {
            regions.push((start_idx, tokens.len(), current_mode));
        }

        regions
    }
}

/// Mode-aware N-gram language model for LaTeX.
///
/// This model maintains separate n-gram statistics for different LaTeX contexts
/// (commands, math, text) and combines them using configurable weights.
pub struct LaTeXNgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// N-gram model for command sequences.
    command_model: NgramModel<D>,
    /// N-gram model for math expressions.
    math_model: NgramModel<D>,
    /// N-gram model for natural text.
    text_model: NgramModel<D>,
    /// Combined model for all contexts.
    combined_model: NgramModel<D>,
    /// Mode detector.
    mode_detector: ModeDetector,
    /// Configuration.
    config: NgramConfig,
}

impl<D> LaTeXNgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// Create a new LaTeX n-gram model from pre-trained component models.
    pub fn new(
        command_model: NgramModel<D>,
        math_model: NgramModel<D>,
        text_model: NgramModel<D>,
        combined_model: NgramModel<D>,
        config: NgramConfig,
    ) -> Self {
        Self {
            command_model,
            math_model,
            text_model,
            combined_model,
            mode_detector: ModeDetector::new(),
            config,
        }
    }

    /// Get the n-gram order.
    pub fn order(&self) -> usize {
        self.config.order
    }

    /// Get the configuration.
    pub fn config(&self) -> &NgramConfig {
        &self.config
    }

    /// Get a reference to the mode detector.
    pub fn mode_detector(&self) -> &ModeDetector {
        &self.mode_detector
    }

    /// Score a sequence of LaTeX tokens.
    ///
    /// Returns the log probability of the sequence, weighted by mode.
    pub fn score(&self, tokens: &[LaTeXToken]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        if !self.config.mode_separation {
            // Use combined model only
            let texts: Vec<String> = tokens.iter().map(|t| t.text()).collect();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            return self.combined_model.sentence_log_prob(&refs);
        }

        // Detect mode transitions and score each region
        let regions = self.mode_detector.detect_transitions(tokens);
        let mut total_score = 0.0;
        let mut total_weight = 0.0;

        for (start, end, mode) in regions {
            let region_tokens = &tokens[start..end];
            let texts: Vec<String> = region_tokens.iter().map(|t| t.text()).collect();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            let model = match mode {
                LaTeXMode::Command => &self.command_model,
                LaTeXMode::Math => &self.math_model,
                LaTeXMode::Text => &self.text_model,
                LaTeXMode::Mixed => &self.combined_model,
            };

            let score = model.sentence_log_prob(&refs);
            let weight = mode.weight(&self.config);

            total_score += score * weight;
            total_weight += weight * (end - start) as f64;
        }

        if total_weight > 0.0 {
            total_score / total_weight * tokens.len() as f64
        } else {
            total_score
        }
    }

    /// Score a token given its context.
    ///
    /// Returns the log probability of the token given the context.
    pub fn score_token(&self, token: &LaTeXToken, context: &[LaTeXToken]) -> f64 {
        let mode = if context.is_empty() {
            self.mode_detector.token_mode(token)
        } else {
            self.mode_detector.sequence_mode(context)
        };

        let model = match mode {
            LaTeXMode::Command => &self.command_model,
            LaTeXMode::Math => &self.math_model,
            LaTeXMode::Text => &self.text_model,
            LaTeXMode::Mixed => &self.combined_model,
        };

        let token_text = token.text();
        let context_texts: Vec<String> = context.iter().map(|t| t.text()).collect();
        let context_refs: Vec<&str> = context_texts.iter().map(|s| s.as_str()).collect();

        model.log_prob(&token_text, &context_refs)
    }

    /// Get the vocabulary size for a specific mode.
    pub fn vocab_size(&self, mode: LaTeXMode) -> usize {
        match mode {
            LaTeXMode::Command => self.command_model.vocab_size(),
            LaTeXMode::Math => self.math_model.vocab_size(),
            LaTeXMode::Text => self.text_model.vocab_size(),
            LaTeXMode::Mixed => self.combined_model.vocab_size(),
        }
    }

    /// Check if a token is in the vocabulary for a specific mode.
    pub fn in_vocabulary(&self, token: &str, mode: LaTeXMode) -> bool {
        match mode {
            LaTeXMode::Command => self.command_model.in_vocabulary(token),
            LaTeXMode::Math => self.math_model.in_vocabulary(token),
            LaTeXMode::Text => self.text_model.in_vocabulary(token),
            LaTeXMode::Mixed => self.combined_model.in_vocabulary(token),
        }
    }

    /// Get the underlying model for a specific mode.
    pub fn model_for_mode(&self, mode: LaTeXMode) -> &NgramModel<D> {
        match mode {
            LaTeXMode::Command => &self.command_model,
            LaTeXMode::Math => &self.math_model,
            LaTeXMode::Text => &self.text_model,
            LaTeXMode::Mixed => &self.combined_model,
        }
    }
}

/// Builder for training LaTeX n-gram models.
pub struct LaTeXNgramTrainer<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// Configuration.
    config: NgramConfig,
    /// Command sequence buffer.
    command_buffer: Vec<String>,
    /// Math expression buffer.
    math_buffer: Vec<String>,
    /// Text buffer.
    text_buffer: Vec<String>,
    /// Combined buffer.
    combined_buffer: Vec<String>,
    /// Mode detector.
    mode_detector: ModeDetector,
    /// Dictionary factory.
    _marker: std::marker::PhantomData<D>,
}

impl<D> LaTeXNgramTrainer<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Default,
{
    /// Create a new trainer with default configuration.
    pub fn new() -> Self {
        Self::with_config(NgramConfig::default())
    }

    /// Create a new trainer with custom configuration.
    pub fn with_config(config: NgramConfig) -> Self {
        Self {
            config,
            command_buffer: Vec::new(),
            math_buffer: Vec::new(),
            text_buffer: Vec::new(),
            combined_buffer: Vec::new(),
            mode_detector: ModeDetector::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Add tokens to the training data.
    pub fn add_tokens(&mut self, tokens: &[LaTeXToken]) {
        for token in tokens {
            let text = token.text();
            let mode = self.mode_detector.token_mode(token);

            // Add to mode-specific buffer
            match mode {
                LaTeXMode::Command => self.command_buffer.push(text.clone()),
                LaTeXMode::Math => self.math_buffer.push(text.clone()),
                LaTeXMode::Text => self.text_buffer.push(text.clone()),
                LaTeXMode::Mixed => {}
            }

            // Always add to combined buffer
            self.combined_buffer.push(text);
        }
    }

    /// Get the current buffer sizes.
    pub fn buffer_sizes(&self) -> (usize, usize, usize, usize) {
        (
            self.command_buffer.len(),
            self.math_buffer.len(),
            self.text_buffer.len(),
            self.combined_buffer.len(),
        )
    }
}

impl<D> Default for LaTeXNgramTrainer<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Sliding window iterator for n-gram extraction.
pub struct NgramWindow<'a> {
    tokens: &'a [LaTeXToken],
    order: usize,
    position: usize,
}

impl<'a> NgramWindow<'a> {
    /// Create a new n-gram window iterator.
    pub fn new(tokens: &'a [LaTeXToken], order: usize) -> Self {
        Self {
            tokens,
            order,
            position: 0,
        }
    }
}

impl<'a> Iterator for NgramWindow<'a> {
    type Item = (&'a [LaTeXToken], &'a LaTeXToken);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.tokens.len() {
            return None;
        }

        let context_start = self.position.saturating_sub(self.order - 1);
        let context = &self.tokens[context_start..self.position];
        let token = &self.tokens[self.position];

        self.position += 1;
        Some((context, token))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latex::tokenizer::LaTeXTokenizer;

    #[test]
    fn test_mode_detector_token() {
        let tokenizer = LaTeXTokenizer::new();
        let detector = ModeDetector::new();

        let tokens = tokenizer.tokenize(r"\alpha");
        assert_eq!(detector.token_mode(&tokens[0]), LaTeXMode::Command);

        let tokens = tokenizer.tokenize(r"$x$");
        assert_eq!(detector.token_mode(&tokens[1]), LaTeXMode::Math);
    }

    #[test]
    fn test_mode_detector_sequence() {
        let tokenizer = LaTeXTokenizer::new();
        let detector = ModeDetector::new();

        let tokens = tokenizer.tokenize(r"\alpha \beta \gamma");
        assert_eq!(detector.sequence_mode(&tokens), LaTeXMode::Command);

        let tokens = tokenizer.tokenize(r"$x + y + z$");
        assert_eq!(detector.sequence_mode(&tokens), LaTeXMode::Math);
    }

    #[test]
    fn test_mode_transitions() {
        let tokenizer = LaTeXTokenizer::new();
        let detector = ModeDetector::new();

        let tokens = tokenizer.tokenize(r"\textbf{text} $x^2$");
        let regions = detector.detect_transitions(&tokens);

        // Should detect at least two distinct regions
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_ngram_window() {
        let tokenizer = LaTeXTokenizer::new();
        let tokens = tokenizer.tokenize(r"$a + b$");

        let window = NgramWindow::new(&tokens, 3);
        let ngrams: Vec<_> = window.collect();

        assert_eq!(ngrams.len(), tokens.len());
    }
}
