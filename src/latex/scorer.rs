//! Combined LaTeX scorer integrating n-gram, embedding, and neural components.
//!
//! This module provides a unified scoring interface that combines multiple
//! scoring components for comprehensive LaTeX sequence evaluation.

use crate::latex::embedding::CommandCategory;
use crate::latex::ngram::{LaTeXMode, ModeDetector};
use crate::latex::tokenizer::{LaTeXToken, LaTeXTokenKind};
use std::collections::HashMap;

/// Configuration for the combined scorer.
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Weight for n-gram score component.
    pub ngram_weight: f64,
    /// Weight for embedding similarity score.
    pub embedding_weight: f64,
    /// Weight for neural rescorer score.
    pub neural_weight: f64,
    /// Weight for structural validity score.
    pub structural_weight: f64,
    /// Weight for RAG retrieval score.
    pub rag_weight: f64,
    /// Whether to normalize component scores before combining.
    pub normalize_components: bool,
    /// Minimum score threshold for acceptance.
    pub min_score: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            ngram_weight: 0.30,
            embedding_weight: 0.15,
            neural_weight: 0.25,
            structural_weight: 0.20,
            rag_weight: 0.10,
            normalize_components: true,
            min_score: 0.0,
        }
    }
}

impl ScorerConfig {
    /// Create a config emphasizing statistical scoring.
    pub fn statistical() -> Self {
        Self {
            ngram_weight: 0.50,
            embedding_weight: 0.20,
            neural_weight: 0.10,
            structural_weight: 0.15,
            rag_weight: 0.05,
            ..Default::default()
        }
    }

    /// Create a config emphasizing neural scoring.
    pub fn neural() -> Self {
        Self {
            ngram_weight: 0.20,
            embedding_weight: 0.15,
            neural_weight: 0.45,
            structural_weight: 0.15,
            rag_weight: 0.05,
            ..Default::default()
        }
    }

    /// Create a config emphasizing structural validity.
    pub fn structural() -> Self {
        Self {
            ngram_weight: 0.20,
            embedding_weight: 0.10,
            neural_weight: 0.15,
            structural_weight: 0.45,
            rag_weight: 0.10,
            ..Default::default()
        }
    }
}

/// Individual component score with metadata.
#[derive(Debug, Clone)]
pub struct ComponentScore {
    /// Name of the scoring component.
    pub name: String,
    /// Raw score value.
    pub raw_score: f64,
    /// Normalized score (0.0 to 1.0).
    pub normalized_score: f64,
    /// Weight used in combination.
    pub weight: f64,
    /// Additional details about the score.
    pub details: HashMap<String, String>,
}

impl ComponentScore {
    /// Create a new component score.
    pub fn new(name: &str, raw_score: f64, weight: f64) -> Self {
        // Default normalization: assume raw_score is already in a reasonable range
        let normalized = raw_score.clamp(0.0, 1.0);
        Self {
            name: name.to_string(),
            raw_score,
            normalized_score: normalized,
            weight,
            details: HashMap::new(),
        }
    }

    /// Set the normalized score explicitly.
    pub fn with_normalized(mut self, normalized: f64) -> Self {
        self.normalized_score = normalized.clamp(0.0, 1.0);
        self
    }

    /// Add a detail.
    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the weighted score.
    pub fn weighted_score(&self) -> f64 {
        self.normalized_score * self.weight
    }
}

/// Result of scoring a LaTeX sequence.
#[derive(Debug, Clone)]
pub struct ScoringResult {
    /// The scored sequence as text.
    pub sequence: String,
    /// Combined final score.
    pub score: f64,
    /// Individual component scores.
    pub components: Vec<ComponentScore>,
    /// Detected dominant mode.
    pub mode: LaTeXMode,
    /// Whether the sequence passes minimum score threshold.
    pub passes_threshold: bool,
    /// Confidence in the score (based on component agreement).
    pub confidence: f64,
}

impl ScoringResult {
    /// Create a new scoring result.
    pub fn new(sequence: String, score: f64, mode: LaTeXMode) -> Self {
        Self {
            sequence,
            score,
            components: Vec::new(),
            mode,
            passes_threshold: true,
            confidence: 1.0,
        }
    }

    /// Add a component score.
    pub fn add_component(&mut self, component: ComponentScore) {
        self.components.push(component);
    }

    /// Get a component by name.
    pub fn component(&self, name: &str) -> Option<&ComponentScore> {
        self.components.iter().find(|c| c.name == name)
    }

    /// Compute confidence based on component agreement.
    pub fn compute_confidence(&mut self) {
        if self.components.len() < 2 {
            self.confidence = 1.0;
            return;
        }

        // Compute standard deviation of normalized scores
        let scores: Vec<f64> = self.components.iter().map(|c| c.normalized_score).collect();
        let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance: f64 =
            scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        // Lower std_dev = higher confidence (components agree)
        self.confidence = (1.0 - std_dev.min(0.5) * 2.0).max(0.0);
    }
}

/// Combined LaTeX scorer.
pub struct LaTeXScorer {
    /// Configuration.
    config: ScorerConfig,
    /// Mode detector for context analysis.
    mode_detector: ModeDetector,
    /// Cache for recent scores.
    cache: HashMap<String, ScoringResult>,
    /// Maximum cache size.
    max_cache_size: usize,
}

impl LaTeXScorer {
    /// Create a new scorer with default configuration.
    pub fn new() -> Self {
        Self::with_config(ScorerConfig::default())
    }

    /// Create a scorer with custom configuration.
    pub fn with_config(config: ScorerConfig) -> Self {
        Self {
            config,
            mode_detector: ModeDetector::new(),
            cache: HashMap::new(),
            max_cache_size: 10000,
        }
    }

    /// Create a builder for more complex configuration.
    pub fn builder() -> LaTeXScorerBuilder {
        LaTeXScorerBuilder::new()
    }

    /// Score a sequence of LaTeX tokens.
    pub fn score(&mut self, tokens: &[LaTeXToken]) -> ScoringResult {
        let sequence = tokens_to_string(tokens);

        // Check cache
        if let Some(cached) = self.cache.get(&sequence) {
            return cached.clone();
        }

        // Detect mode
        let mode = self.mode_detector.sequence_mode(tokens);

        // Compute component scores
        let mut components = Vec::new();

        // Structural score
        let structural = self.compute_structural_score(tokens);
        components.push(
            ComponentScore::new("structural", structural, self.config.structural_weight)
                .with_normalized(structural),
        );

        // Local token fluency score.
        let ngram = self.compute_local_fluency_score(tokens);
        components.push(
            ComponentScore::new("ngram", ngram, self.config.ngram_weight).with_normalized(ngram),
        );

        // Semantic coherence score.
        let embedding = self.compute_semantic_coherence_score(tokens);
        components.push(
            ComponentScore::new("embedding", embedding, self.config.embedding_weight)
                .with_normalized(embedding),
        );

        // Combine scores
        let combined = self.combine_scores(&components);

        let mut result = ScoringResult::new(sequence.clone(), combined, mode);
        result.components = components;
        result.passes_threshold = combined >= self.config.min_score;
        result.compute_confidence();

        // Cache result
        if self.cache.len() >= self.max_cache_size {
            self.cache.clear();
        }
        self.cache.insert(sequence, result.clone());

        result
    }

    /// Score multiple sequences and return sorted results.
    pub fn score_candidates(&mut self, candidates: &[&[LaTeXToken]]) -> Vec<ScoringResult> {
        let mut results: Vec<ScoringResult> =
            candidates.iter().map(|tokens| self.score(tokens)).collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Get the best scoring candidate.
    pub fn best_candidate(&mut self, candidates: &[&[LaTeXToken]]) -> Option<ScoringResult> {
        self.score_candidates(candidates).into_iter().next()
    }

    /// Compute structural validity score.
    fn compute_structural_score(&self, tokens: &[LaTeXToken]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let mut score = 1.0;
        let mut brace_stack: Vec<char> = Vec::new();
        let mut math_stack: Vec<char> = Vec::new();
        let mut penalties = 0.0;

        for token in tokens {
            match &token.kind {
                // Track braces
                LaTeXTokenKind::OpenBrace(kind) => {
                    let c = match kind {
                        crate::latex::tokenizer::BraceKind::Curly => '{',
                        crate::latex::tokenizer::BraceKind::Square => '[',
                        crate::latex::tokenizer::BraceKind::Paren => '(',
                    };
                    brace_stack.push(c);
                }
                LaTeXTokenKind::CloseBrace(kind) => {
                    let expected = match kind {
                        crate::latex::tokenizer::BraceKind::Curly => '{',
                        crate::latex::tokenizer::BraceKind::Square => '[',
                        crate::latex::tokenizer::BraceKind::Paren => '(',
                    };
                    if brace_stack.pop() != Some(expected) {
                        penalties += 0.2; // Mismatched brace
                    }
                }

                // Track math delimiters
                LaTeXTokenKind::MathOpen(mode) => {
                    let c = match mode {
                        crate::latex::tokenizer::MathMode::InlineDollar => '$',
                        crate::latex::tokenizer::MathMode::DisplayDoubleDollar => 'D',
                        crate::latex::tokenizer::MathMode::InlineParen => '(',
                        crate::latex::tokenizer::MathMode::DisplayBracket => '[',
                        crate::latex::tokenizer::MathMode::Environment => 'E',
                    };
                    math_stack.push(c);
                }
                LaTeXTokenKind::MathClose(mode) => {
                    let expected = match mode {
                        crate::latex::tokenizer::MathMode::InlineDollar => '$',
                        crate::latex::tokenizer::MathMode::DisplayDoubleDollar => 'D',
                        crate::latex::tokenizer::MathMode::InlineParen => '(',
                        crate::latex::tokenizer::MathMode::DisplayBracket => '[',
                        crate::latex::tokenizer::MathMode::Environment => 'E',
                    };
                    if math_stack.pop() != Some(expected) {
                        penalties += 0.3; // Mismatched math delimiter
                    }
                }

                // Penalize unknown tokens
                LaTeXTokenKind::Unknown(_) => {
                    penalties += 0.1;
                }

                _ => {}
            }
        }

        // Penalize unclosed structures
        penalties += brace_stack.len() as f64 * 0.15;
        penalties += math_stack.len() as f64 * 0.2;

        score = (score - penalties).max(0.0);
        score
    }

    /// Compute local token fluency from adjacent LaTeX token transitions.
    fn compute_local_fluency_score(&self, tokens: &[LaTeXToken]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        if tokens.len() == 1 {
            return match tokens[0].kind {
                LaTeXTokenKind::Unknown(_) => 0.25,
                _ => 0.70,
            };
        }

        let mut total = 0.0;
        let mut transitions = 0usize;
        for pair in tokens.windows(2) {
            total += self.transition_fluency(&pair[0], &pair[1]);
            transitions += 1;
        }

        let transition_score = total / transitions as f64;
        let command_ratio = tokens
            .iter()
            .filter(|t| matches!(t.kind, LaTeXTokenKind::Command(_)))
            .count() as f64
            / tokens.len() as f64;
        let density_score = (1.0 - (command_ratio - 0.20).abs() * 1.5).clamp(0.35, 1.0);

        (transition_score * 0.85 + density_score * 0.15).clamp(0.0, 1.0)
    }

    fn transition_fluency(&self, previous: &LaTeXToken, current: &LaTeXToken) -> f64 {
        use LaTeXTokenKind::*;

        match (&previous.kind, &current.kind) {
            (Command(cmd), OpenBrace(_)) if command_takes_group(cmd) => 1.0,
            (Command(cmd), _) if command_takes_group(cmd) => 0.45,
            (OpenBrace(_), CloseBrace(_)) => 0.55,
            (OpenBrace(_), _) | (_, CloseBrace(_)) => 0.85,
            (MathOpen(_), MathClose(_)) => 0.40,
            (MathOpen(_), _) | (_, MathClose(_)) => 0.95,
            (Identifier(_), Operator(_)) | (Number(_), Operator(_)) => 0.95,
            (Operator(_), Identifier(_)) | (Operator(_), Number(_)) | (Operator(_), Command(_)) => {
                0.95
            }
            (Subscript | Superscript, Identifier(_))
            | (Subscript | Superscript, Number(_))
            | (Subscript | Superscript, Command(_))
            | (Subscript | Superscript, OpenBrace(_)) => 0.95,
            (Subscript | Superscript, _) => 0.35,
            (Command(left), Command(right)) => command_pair_fluency(left, right),
            (Unknown(_), _) | (_, Unknown(_)) => 0.20,
            (Text(_), Text(_)) | (Identifier(_), Identifier(_)) => 0.75,
            _ => 0.70,
        }
    }

    /// Compute semantic coherence from token modes and command categories.
    fn compute_semantic_coherence_score(&self, tokens: &[LaTeXToken]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let mode = self.mode_detector.sequence_mode(tokens);
        let mode_matches = tokens
            .iter()
            .filter(|t| self.mode_detector.token_mode(t) == mode)
            .count();
        let mode_score = mode_matches as f64 / tokens.len() as f64;

        let command_categories: Vec<CommandCategory> = tokens
            .iter()
            .filter_map(|token| match &token.kind {
                LaTeXTokenKind::Command(command) => Some(CommandCategory::from_command(command)),
                _ => None,
            })
            .collect();

        let category_score = if command_categories.len() < 2 {
            1.0
        } else {
            let coherent_pairs = command_categories
                .windows(2)
                .filter(|pair| command_categories_are_compatible(pair[0], pair[1]))
                .count();
            coherent_pairs as f64 / (command_categories.len() - 1) as f64
        };

        let unknown_penalty = tokens
            .iter()
            .filter(|token| matches!(token.kind, LaTeXTokenKind::Unknown(_)))
            .count() as f64
            / tokens.len() as f64;

        (mode_score * 0.55 + category_score * 0.45 - unknown_penalty * 0.35).clamp(0.0, 1.0)
    }

    /// Combine component scores according to weights.
    fn combine_scores(&self, components: &[ComponentScore]) -> f64 {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for component in components {
            weighted_sum += component.weighted_score();
            total_weight += component.weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Clear the score cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.max_cache_size)
    }

    /// Get the configuration.
    pub fn config(&self) -> &ScorerConfig {
        &self.config
    }
}

impl Default for LaTeXScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for LaTeXScorer with fluent API.
pub struct LaTeXScorerBuilder {
    config: ScorerConfig,
}

impl LaTeXScorerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: ScorerConfig::default(),
        }
    }

    /// Set n-gram weight.
    pub fn ngram_weight(mut self, weight: f64) -> Self {
        self.config.ngram_weight = weight;
        self
    }

    /// Set embedding weight.
    pub fn embedding_weight(mut self, weight: f64) -> Self {
        self.config.embedding_weight = weight;
        self
    }

    /// Set neural weight.
    pub fn neural_weight(mut self, weight: f64) -> Self {
        self.config.neural_weight = weight;
        self
    }

    /// Set structural weight.
    pub fn structural_weight(mut self, weight: f64) -> Self {
        self.config.structural_weight = weight;
        self
    }

    /// Set RAG weight.
    pub fn rag_weight(mut self, weight: f64) -> Self {
        self.config.rag_weight = weight;
        self
    }

    /// Set minimum score threshold.
    pub fn min_score(mut self, min: f64) -> Self {
        self.config.min_score = min;
        self
    }

    /// Enable/disable component normalization.
    pub fn normalize_components(mut self, normalize: bool) -> Self {
        self.config.normalize_components = normalize;
        self
    }

    /// Use statistical preset.
    pub fn statistical_preset(mut self) -> Self {
        self.config = ScorerConfig::statistical();
        self
    }

    /// Use neural preset.
    pub fn neural_preset(mut self) -> Self {
        self.config = ScorerConfig::neural();
        self
    }

    /// Use structural preset.
    pub fn structural_preset(mut self) -> Self {
        self.config = ScorerConfig::structural();
        self
    }

    /// Build the scorer.
    pub fn build(self) -> LaTeXScorer {
        LaTeXScorer::with_config(self.config)
    }
}

impl Default for LaTeXScorerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert tokens to a string representation.
fn tokens_to_string(tokens: &[LaTeXToken]) -> String {
    tokens.iter().map(|t| t.text()).collect::<Vec<_>>().join("")
}

fn command_takes_group(command: &str) -> bool {
    matches!(
        command,
        "frac"
            | "sqrt"
            | "binom"
            | "overline"
            | "underline"
            | "hat"
            | "bar"
            | "vec"
            | "text"
            | "textbf"
            | "textit"
            | "emph"
            | "section"
            | "subsection"
            | "subsubsection"
            | "begin"
            | "end"
    )
}

fn command_pair_fluency(left: &str, right: &str) -> f64 {
    match (left, right) {
        ("left", "right") | ("begin", "end") => 0.20,
        ("left", _) | (_, "right") => 0.60,
        _ => {
            let left_category = CommandCategory::from_command(left);
            let right_category = CommandCategory::from_command(right);
            if command_categories_are_compatible(left_category, right_category) {
                0.85
            } else {
                0.55
            }
        }
    }
}

fn command_categories_are_compatible(left: CommandCategory, right: CommandCategory) -> bool {
    use CommandCategory::*;

    matches!(
        (left, right),
        (GreekLetter, GreekLetter)
            | (Operator, GreekLetter)
            | (Operator, Function)
            | (Function, GreekLetter)
            | (Function, Operator)
            | (Relation, GreekLetter)
            | (Relation, Function)
            | (Accent, GreekLetter)
            | (Accent, Function)
            | (Delimiter, Delimiter)
            | (Environment, Environment)
            | (Formatting, Formatting)
            | (Structure, Structure)
            | (Arrow, Arrow)
            | (Spacing, _)
            | (_, Spacing)
    ) || left == right
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latex::tokenizer::LaTeXTokenizer;

    #[test]
    fn test_basic_scoring() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        let tokens = tokenizer.tokenize(r"\alpha + \beta");
        let result = scorer.score(&tokens);

        assert!(!result.sequence.is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.components.is_empty());
    }

    #[test]
    fn test_structural_scoring() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        // Well-formed should score higher
        let well_formed = tokenizer.tokenize(r"\frac{a}{b}");
        let malformed = tokenizer.tokenize(r"\frac{a}{b");

        let score_well = scorer.score(&well_formed);
        let score_mal = scorer.score(&malformed);

        assert!(score_well.score >= score_mal.score);
    }

    #[test]
    fn test_builder() {
        let scorer = LaTeXScorer::builder()
            .ngram_weight(0.5)
            .structural_weight(0.5)
            .min_score(0.3)
            .build();

        assert_eq!(scorer.config().ngram_weight, 0.5);
        assert_eq!(scorer.config().structural_weight, 0.5);
        assert_eq!(scorer.config().min_score, 0.3);
    }

    #[test]
    fn test_presets() {
        let statistical = ScorerConfig::statistical();
        assert!(statistical.ngram_weight > statistical.neural_weight);

        let neural = ScorerConfig::neural();
        assert!(neural.neural_weight > neural.ngram_weight);

        let structural = ScorerConfig::structural();
        assert!(structural.structural_weight > structural.ngram_weight);
    }

    #[test]
    fn test_candidate_ranking() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        let candidates: Vec<Vec<LaTeXToken>> = vec![
            tokenizer.tokenize(r"\alpha"),
            tokenizer.tokenize(r"\frac{1}{2}"),
            tokenizer.tokenize(r"$x^2$"),
        ];

        let refs: Vec<&[LaTeXToken]> = candidates.iter().map(|v| v.as_slice()).collect();
        let results = scorer.score_candidates(&refs);

        assert_eq!(results.len(), 3);
        // Results should be sorted by score (descending)
        for i in 1..results.len() {
            assert!(results[i - 1].score >= results[i].score);
        }
    }

    #[test]
    fn test_caching() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        let tokens = tokenizer.tokenize(r"\alpha");

        // First call
        let result1 = scorer.score(&tokens);
        assert_eq!(scorer.cache_stats().0, 1);

        // Second call should use cache
        let result2 = scorer.score(&tokens);
        assert_eq!(result1.score, result2.score);

        scorer.clear_cache();
        assert_eq!(scorer.cache_stats().0, 0);
    }

    #[test]
    fn test_confidence() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        let tokens = tokenizer.tokenize(r"\frac{a}{b}");
        let result = scorer.score(&tokens);

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_component_scores() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::new();

        let tokens = tokenizer.tokenize(r"\alpha + \beta = \gamma");
        let result = scorer.score(&tokens);

        assert!(result.component("structural").is_some());
        assert!(result.component("ngram").is_some());
        assert!(result.component("embedding").is_some());
    }

    #[test]
    fn test_local_fluency_rewards_latex_argument_structure() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::builder()
            .ngram_weight(1.0)
            .embedding_weight(0.0)
            .neural_weight(0.0)
            .structural_weight(0.0)
            .rag_weight(0.0)
            .build();

        let fluent = scorer.score(&tokenizer.tokenize(r"\frac{a}{b}"));
        let abrupt = scorer.score(&tokenizer.tokenize(r"\frac \alpha \beta"));

        assert!(
            fluent.component("ngram").unwrap().normalized_score
                > abrupt.component("ngram").unwrap().normalized_score
        );
    }

    #[test]
    fn test_semantic_coherence_penalizes_unknown_tokens() {
        let tokenizer = LaTeXTokenizer::new();
        let mut scorer = LaTeXScorer::builder()
            .ngram_weight(0.0)
            .embedding_weight(1.0)
            .neural_weight(0.0)
            .structural_weight(0.0)
            .rag_weight(0.0)
            .build();

        let coherent = scorer.score(&tokenizer.tokenize(r"\alpha + \beta"));
        let mut noisy_tokens = tokenizer.tokenize(r"\alpha + \beta");
        noisy_tokens.push(LaTeXToken::new(
            LaTeXTokenKind::Unknown("@@".to_string()),
            0,
            2,
            false,
        ));
        let noisy = scorer.score(&noisy_tokens);

        assert!(
            coherent.component("embedding").unwrap().normalized_score
                > noisy.component("embedding").unwrap().normalized_score
        );
    }
}
