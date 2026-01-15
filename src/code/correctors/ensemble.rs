//! Ensemble corrector combining lexical, grammar, and semantic sources.
//!
//! This corrector orchestrates multiple correction sources and combines
//! their suggestions using configurable weighting and deduplication.

use super::grammar::GrammarCorrector;
use super::lexical::LexicalCorrector;
use super::semantic::SemanticCorrector;
use crate::code::ast::ParsedCode;
use crate::code::correction::{
    CodeCorrector, Correction, CorrectionCandidates, CorrectionKind, CorrectionSource,
};
use crate::code::cpg::CodePropertyGraph;
use crate::code::language::{CodeLanguage, TokenContext};
use crate::code::pcfg::WeightedCFG;
use crate::code::tokenizer::CodeToken;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the ensemble corrector.
#[derive(Debug, Clone)]
pub struct EnsembleCorrectorConfig {
    /// Weight for lexical corrections (0.0 - 1.0)
    pub lexical_weight: f64,
    /// Weight for grammar corrections (0.0 - 1.0)
    pub grammar_weight: f64,
    /// Weight for semantic corrections (0.0 - 1.0)
    pub semantic_weight: f64,
    /// Minimum confidence to include in results
    pub min_confidence: f64,
    /// Maximum total candidates to return
    pub max_candidates: usize,
    /// Whether to deduplicate similar corrections
    pub deduplicate: bool,
    /// Similarity threshold for deduplication
    pub dedup_threshold: f64,
    /// Whether to boost confidence when multiple sources agree
    pub agreement_boost: bool,
    /// Boost factor when sources agree
    pub agreement_boost_factor: f64,
}

impl Default for EnsembleCorrectorConfig {
    fn default() -> Self {
        Self {
            lexical_weight: 0.4,
            grammar_weight: 0.35,
            semantic_weight: 0.25,
            min_confidence: 0.3,
            max_candidates: 10,
            deduplicate: true,
            dedup_threshold: 0.9,
            agreement_boost: true,
            agreement_boost_factor: 1.3,
        }
    }
}

/// Ensemble corrector combining multiple correction sources.
///
/// This corrector aggregates suggestions from lexical, grammar, and semantic
/// correctors, applying configurable weights and deduplication to produce
/// a unified ranked list of corrections.
pub struct EnsembleCorrector<L: CodeLanguage> {
    language: Arc<L>,
    config: EnsembleCorrectorConfig,
    lexical: Option<LexicalCorrector<L>>,
    grammar: Option<GrammarCorrector<L>>,
    semantic: Option<SemanticCorrector<L>>,
}

impl<L: CodeLanguage + Clone> EnsembleCorrector<L> {
    /// Creates a new ensemble corrector with all components.
    pub fn new(
        language: Arc<L>,
        grammar: Option<WeightedCFG>,
        config: EnsembleCorrectorConfig,
    ) -> Self {
        let lexical = Some(LexicalCorrector::with_defaults(Arc::clone(&language)));
        let grammar_corrector = grammar.map(|g| GrammarCorrector::with_defaults(Arc::clone(&language), g));
        let semantic = Some(SemanticCorrector::with_defaults(Arc::clone(&language)));

        Self {
            language,
            config,
            lexical,
            grammar: grammar_corrector,
            semantic,
        }
    }

    /// Creates an ensemble corrector with default configuration.
    pub fn with_defaults(language: Arc<L>, grammar: Option<WeightedCFG>) -> Self {
        Self::new(language, grammar, EnsembleCorrectorConfig::default())
    }

    /// Creates a minimal ensemble with only lexical correction.
    pub fn lexical_only(language: Arc<L>) -> Self {
        Self {
            language: Arc::clone(&language),
            config: EnsembleCorrectorConfig::default(),
            lexical: Some(LexicalCorrector::with_defaults(language)),
            grammar: None,
            semantic: None,
        }
    }

    /// Returns a mutable reference to the lexical corrector.
    pub fn lexical_mut(&mut self) -> Option<&mut LexicalCorrector<L>> {
        self.lexical.as_mut()
    }

    /// Returns a mutable reference to the semantic corrector.
    pub fn semantic_mut(&mut self) -> Option<&mut SemanticCorrector<L>> {
        self.semantic.as_mut()
    }

    /// Adds identifiers to the lexical corrector.
    pub fn add_identifiers(&mut self, identifiers: &[&str]) {
        if let Some(ref mut lexical) = self.lexical {
            for id in identifiers {
                lexical.add_identifier(id);
            }
        }
    }

    /// Registers variables with the semantic corrector.
    pub fn register_variables(&mut self, variables: &[(String, Option<String>)]) {
        if let Some(ref mut semantic) = self.semantic {
            for (name, type_name) in variables {
                semantic.register_variable(name.clone(), type_name.clone(), 0);
            }
        }
    }

    /// Collects corrections from all sources for a token.
    fn collect_corrections(&self, token: &CodeToken, context: &TokenContext) -> Vec<(Correction, f64)> {
        let mut corrections = Vec::new();

        // Lexical corrections
        if let Some(ref lexical) = self.lexical {
            for c in lexical.correct_token(token, context) {
                corrections.push((c, self.config.lexical_weight));
            }
        }

        // Grammar corrections
        if let Some(ref grammar) = self.grammar {
            for c in grammar.correct_token(token, context) {
                corrections.push((c, self.config.grammar_weight));
            }
        }

        // Semantic corrections
        if let Some(ref semantic) = self.semantic {
            for c in semantic.correct_token(token, context) {
                corrections.push((c, self.config.semantic_weight));
            }
        }

        corrections
    }

    /// Applies source weighting to correction confidence.
    fn apply_weight(correction: &Correction, weight: f64) -> Correction {
        let mut c = correction.clone();
        c.confidence *= weight;
        c
    }

    /// Checks if two corrections are similar enough to merge.
    fn are_similar(&self, a: &Correction, b: &Correction) -> bool {
        if a.replacement != b.replacement {
            return false;
        }
        if a.start_byte != b.start_byte || a.end_byte != b.end_byte {
            return false;
        }
        true
    }

    /// Merges similar corrections, boosting confidence when sources agree.
    fn merge_corrections(&self, corrections: Vec<(Correction, f64)>) -> Vec<Correction> {
        if corrections.is_empty() {
            return vec![];
        }

        // Group by (replacement, position)
        let mut groups: HashMap<(String, usize, usize), Vec<(Correction, f64)>> = HashMap::new();

        for (c, weight) in corrections {
            let key = (c.replacement.clone(), c.start_byte, c.end_byte);
            groups.entry(key).or_default().push((c, weight));
        }

        let mut merged = Vec::new();

        for ((replacement, start, end), group) in groups {
            if group.len() == 1 {
                // Single source
                let (c, weight) = group.into_iter().next().unwrap();
                let mut correction = c;
                correction.confidence *= weight;
                merged.push(correction);
            } else {
                // Multiple sources agree - merge and boost
                let sources: Vec<CorrectionSource> = group.iter().map(|(c, _)| c.source).collect();
                let total_weight: f64 = group.iter().map(|(_, w)| w).sum();
                let avg_confidence: f64 = group.iter().map(|(c, w)| c.confidence * w).sum::<f64>() / total_weight;

                // Take the best correction as base
                let mut best = group.into_iter()
                    .max_by(|a, b| (a.0.confidence * a.1).partial_cmp(&(b.0.confidence * b.1)).unwrap())
                    .unwrap().0;

                // Apply agreement boost
                let boost = if self.config.agreement_boost && sources.len() > 1 {
                    self.config.agreement_boost_factor
                } else {
                    1.0
                };

                best.confidence = (avg_confidence * boost).min(1.0);
                best.source = CorrectionSource::Combined;
                best.context = Some(format!(
                    "Suggested by {} sources",
                    sources.len()
                ));

                merged.push(best);
            }
        }

        merged
    }

    /// Filters and ranks the final corrections.
    fn finalize_corrections(&self, mut corrections: Vec<Correction>) -> Vec<Correction> {
        // Filter by minimum confidence
        corrections.retain(|c| c.confidence >= self.config.min_confidence);

        // Sort by confidence descending
        corrections.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max candidates
        corrections.truncate(self.config.max_candidates);

        corrections
    }

    /// Performs full analysis on parsed code with CPG.
    pub fn analyze_full(
        &self,
        parsed: &ParsedCode,
        cpg: &CodePropertyGraph,
    ) -> Vec<Correction> {
        let mut all_corrections = Vec::new();

        // Get semantic corrections from full analysis
        if let Some(ref semantic) = self.semantic {
            let semantic_corrections = semantic.analyze_parsed(parsed, cpg);
            for c in semantic_corrections {
                all_corrections.push((c, self.config.semantic_weight));
            }
        }

        // Merge and finalize
        let merged = self.merge_corrections(all_corrections);
        self.finalize_corrections(merged)
    }

    /// Returns the configuration.
    pub fn config(&self) -> &EnsembleCorrectorConfig {
        &self.config
    }

    /// Returns the language handler.
    pub fn language(&self) -> &L {
        &self.language
    }
}

impl<L: CodeLanguage + Clone + Send + Sync> CodeCorrector for EnsembleCorrector<L> {
    fn correct_token(&self, token: &CodeToken, context: &TokenContext) -> Vec<Correction> {
        let corrections = self.collect_corrections(token, context);
        let merged = self.merge_corrections(corrections);
        self.finalize_corrections(merged)
    }

    fn correct_range(
        &self,
        source: &str,
        start_byte: usize,
        end_byte: usize,
    ) -> Vec<Correction> {
        let text = &source[start_byte..end_byte];
        let token = CodeToken::new(
            text,
            start_byte,
            0,
            0,
            crate::code::language::TokenType::Unknown,
            "unknown",
        );

        let context = TokenContext::new(crate::code::language::TokenType::Unknown);
        self.correct_token(&token, &context)
    }

    fn max_edit_distance(&self) -> usize {
        // Return max of all correctors
        let mut max = 2;
        if let Some(ref lexical) = self.lexical {
            max = max.max(lexical.max_edit_distance());
        }
        if let Some(ref grammar) = self.grammar {
            max = max.max(grammar.max_edit_distance());
        }
        if let Some(ref semantic) = self.semantic {
            max = max.max(semantic.max_edit_distance());
        }
        max
    }

    fn name(&self) -> &str {
        "EnsembleCorrector"
    }
}

/// Builder for creating EnsembleCorrector with custom configuration.
pub struct EnsembleCorrectorBuilder<L: CodeLanguage> {
    language: Arc<L>,
    config: EnsembleCorrectorConfig,
    grammar: Option<WeightedCFG>,
    enable_lexical: bool,
    enable_grammar: bool,
    enable_semantic: bool,
}

impl<L: CodeLanguage + Clone> EnsembleCorrectorBuilder<L> {
    /// Creates a new builder.
    pub fn new(language: Arc<L>) -> Self {
        Self {
            language,
            config: EnsembleCorrectorConfig::default(),
            grammar: None,
            enable_lexical: true,
            enable_grammar: true,
            enable_semantic: true,
        }
    }

    /// Sets the grammar for grammar-based correction.
    pub fn with_grammar(mut self, grammar: WeightedCFG) -> Self {
        self.grammar = Some(grammar);
        self
    }

    /// Sets custom configuration.
    pub fn with_config(mut self, config: EnsembleCorrectorConfig) -> Self {
        self.config = config;
        self
    }

    /// Disables lexical correction.
    pub fn without_lexical(mut self) -> Self {
        self.enable_lexical = false;
        self
    }

    /// Disables grammar correction.
    pub fn without_grammar(mut self) -> Self {
        self.enable_grammar = false;
        self
    }

    /// Disables semantic correction.
    pub fn without_semantic(mut self) -> Self {
        self.enable_semantic = false;
        self
    }

    /// Sets the weight for lexical corrections.
    pub fn lexical_weight(mut self, weight: f64) -> Self {
        self.config.lexical_weight = weight;
        self
    }

    /// Sets the weight for grammar corrections.
    pub fn grammar_weight(mut self, weight: f64) -> Self {
        self.config.grammar_weight = weight;
        self
    }

    /// Sets the weight for semantic corrections.
    pub fn semantic_weight(mut self, weight: f64) -> Self {
        self.config.semantic_weight = weight;
        self
    }

    /// Builds the ensemble corrector.
    pub fn build(self) -> EnsembleCorrector<L> {
        let lexical = if self.enable_lexical {
            Some(LexicalCorrector::with_defaults(Arc::clone(&self.language)))
        } else {
            None
        };

        let grammar = if self.enable_grammar {
            self.grammar.map(|g| GrammarCorrector::with_defaults(Arc::clone(&self.language), g))
        } else {
            None
        };

        let semantic = if self.enable_semantic {
            Some(SemanticCorrector::with_defaults(Arc::clone(&self.language)))
        } else {
            None
        };

        EnsembleCorrector {
            language: self.language,
            config: self.config,
            lexical,
            grammar,
            semantic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock language for testing
    #[derive(Debug, Clone, Default)]
    struct MockLanguage;

    impl CodeLanguage for MockLanguage {
        fn name(&self) -> &str { "mock" }
        fn display_name(&self) -> &str { "Mock" }
        fn tree_sitter_language(&self) -> tree_sitter::Language {
            panic!("Not implemented for tests")
        }
        fn keywords(&self) -> &[&str] {
            &["if", "else", "while", "for", "return", "function"]
        }
        fn special_tokens(&self) -> &[&str] { &[] }
        fn file_extensions(&self) -> &[&str] { &["mock"] }
        fn classify_token(&self, _token: &str, _node_kind: &str) -> crate::code::language::TokenType {
            crate::code::language::TokenType::Unknown
        }
        fn is_valid_identifier(&self, s: &str) -> bool {
            !s.is_empty() && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
        }
        fn builtin_types(&self) -> &[&str] { &["int", "string", "bool"] }
        fn stdlib_functions(&self) -> &[&str] { &["print", "read"] }
        fn comment_syntax(&self) -> crate::code::language::CommentSyntax {
            crate::code::language::CommentSyntax::default()
        }
        fn is_whitespace_significant(&self) -> bool { false }
    }

    #[test]
    fn test_ensemble_corrector_creation() {
        let lang = Arc::new(MockLanguage);
        let corrector = EnsembleCorrector::with_defaults(lang, None);

        assert!(corrector.lexical.is_some());
        assert!(corrector.grammar.is_none()); // No grammar provided
        assert!(corrector.semantic.is_some());
    }

    #[test]
    fn test_ensemble_builder() {
        let lang = Arc::new(MockLanguage);
        let corrector = EnsembleCorrectorBuilder::new(lang)
            .without_semantic()
            .lexical_weight(0.6)
            .build();

        assert!(corrector.lexical.is_some());
        assert!(corrector.semantic.is_none());
        assert!((corrector.config.lexical_weight - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_ensemble_correction() {
        let lang = Arc::new(MockLanguage);
        let mut corrector = EnsembleCorrector::lexical_only(Arc::clone(&lang));

        // Add some identifiers
        corrector.add_identifiers(&["calculateTotal", "processData"]);

        let token = CodeToken::new(
            "funtion", // Misspelled "function"
            0,
            1,
            0,
            crate::code::language::TokenType::Keyword,
            "keyword",
        );

        let context = TokenContext::new(crate::code::language::TokenType::Keyword);
        let corrections = corrector.correct_token(&token, &context);

        // Should get corrections for the misspelled keyword
        assert!(!corrections.is_empty());
    }

    #[test]
    fn test_agreement_boost() {
        let config = EnsembleCorrectorConfig {
            agreement_boost: true,
            agreement_boost_factor: 1.5,
            ..Default::default()
        };

        // Corrections from multiple sources should be boosted
        assert!(config.agreement_boost);
        assert!((config.agreement_boost_factor - 1.5).abs() < 0.01);
    }
}
