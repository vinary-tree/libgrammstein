//! Grammar-aware corrector using PCFG and Earley parsing.
//!
//! This corrector uses probabilistic context-free grammars to suggest
//! syntactically valid corrections. It leverages the Earley parser for
//! incremental grammar checking and can suggest insertions/deletions
//! based on grammar constraints.

use crate::code::constrained_decoding::{
    ConstrainedDecodingConfig, EarleyParser, GrammarConstraint,
};
use crate::code::correction::{CodeCorrector, Correction, CorrectionKind, CorrectionSource};
use crate::code::language::{CodeLanguage, TokenContext, TokenType};
use crate::code::pcfg::{Production, Symbol, WeightedCFG};
use crate::code::tokenizer::CodeToken;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for the grammar corrector.
#[derive(Debug, Clone)]
pub struct GrammarCorrectorConfig {
    /// Maximum number of candidates per error
    pub max_candidates: usize,
    /// Minimum probability for rule to be considered
    pub min_rule_probability: f64,
    /// Whether to suggest insertions for missing tokens
    pub suggest_insertions: bool,
    /// Whether to suggest deletions for extra tokens
    pub suggest_deletions: bool,
    /// Maximum lookahead for completion suggestions
    pub max_lookahead: usize,
    /// Base confidence for grammar-based corrections
    pub base_confidence: f64,
}

impl Default for GrammarCorrectorConfig {
    fn default() -> Self {
        Self {
            max_candidates: 5,
            min_rule_probability: 0.01,
            suggest_insertions: true,
            suggest_deletions: true,
            max_lookahead: 3,
            base_confidence: 0.8,
        }
    }
}

/// Grammar-aware corrector using PCFG.
///
/// This corrector maintains a trained PCFG and uses it to:
/// 1. Validate token sequences against the grammar
/// 2. Suggest completions based on grammar rules
/// 3. Identify and correct syntax errors
pub struct GrammarCorrector<L: CodeLanguage> {
    language: Arc<L>,
    config: GrammarCorrectorConfig,
    grammar: WeightedCFG,
    /// Parser for grammar-constrained decoding
    parser: EarleyParser,
    /// Cache of valid next tokens per state
    completion_cache: HashMap<String, Vec<(String, f64)>>,
}

impl<L: CodeLanguage> GrammarCorrector<L> {
    /// Creates a new grammar corrector with a pre-trained grammar.
    pub fn new(language: Arc<L>, grammar: WeightedCFG, config: GrammarCorrectorConfig) -> Self {
        let parser = EarleyParser::new(grammar.clone());

        Self {
            language,
            config,
            grammar,
            parser,
            completion_cache: HashMap::new(),
        }
    }

    /// Creates a corrector with default configuration.
    pub fn with_defaults(language: Arc<L>, grammar: WeightedCFG) -> Self {
        Self::new(language, grammar, GrammarCorrectorConfig::default())
    }

    /// Creates a grammar constraint for validation.
    pub fn create_constraint(&self) -> GrammarConstraint {
        GrammarConstraint::new(self.grammar.clone(), ConstrainedDecodingConfig::default())
    }

    /// Returns valid next tokens given current parse state.
    pub fn valid_next_tokens(&self, token_history: &[&str]) -> HashSet<String> {
        let mut constraint = self.create_constraint();

        // Advance through history
        for token in token_history {
            if !constraint.advance(token) {
                // Invalid prefix - return empty
                return HashSet::new();
            }
        }

        constraint.valid_tokens()
    }

    /// Suggests completions based on grammar rules.
    pub fn suggest_completions(
        &self,
        context: &[&str],
        max_suggestions: usize,
    ) -> Vec<(String, f64)> {
        let valid = self.valid_next_tokens(context);

        // Score each valid token by rule probability
        let mut suggestions: Vec<(String, f64)> = valid
            .into_iter()
            .map(|token| {
                let prob = self.token_probability(&token, context);
                (token, prob)
            })
            .collect();

        // Sort by probability descending
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(max_suggestions);
        suggestions
    }

    /// Estimates probability of a token given context.
    fn token_probability(&self, token: &str, _context: &[&str]) -> f64 {
        // Look for rules that produce this terminal
        let mut max_prob: f64 = 0.0;

        for (production, _) in self.grammar.iter_rules() {
            // Check if this rule produces the token as a terminal
            for symbol in &production.rhs {
                if let Symbol::Terminal(t) = symbol {
                    if t == token {
                        let prob = self.grammar.probability(production);
                        max_prob = max_prob.max(prob);
                    }
                }
            }
        }

        max_prob
    }

    /// Identifies syntax errors in a token sequence.
    pub fn find_syntax_errors(&self, tokens: &[&str]) -> Vec<SyntaxError> {
        let mut errors = Vec::new();
        let mut constraint = self.create_constraint();

        for (i, token) in tokens.iter().enumerate() {
            if !constraint.is_valid_token(token) {
                // This token is invalid at this position
                let valid_tokens = constraint.valid_tokens();
                let is_empty = valid_tokens.is_empty();

                errors.push(SyntaxError {
                    position: i,
                    token: token.to_string(),
                    expected: valid_tokens,
                    error_type: if is_empty {
                        SyntaxErrorType::UnexpectedToken
                    } else {
                        SyntaxErrorType::InvalidToken
                    },
                });

                // Try to recover by skipping the token
                // (Error recovery strategy)
            }

            if !constraint.advance(token) {
                // Could not advance - stuck in error state
                break;
            }
        }

        errors
    }

    /// Generates insertion corrections for missing tokens.
    fn suggest_insertions(
        &self,
        position: usize,
        context: &[&str],
        _source: &str,
        byte_position: usize,
    ) -> Vec<Correction> {
        if !self.config.suggest_insertions {
            return vec![];
        }

        let suggestions = self.suggest_completions(context, self.config.max_candidates);

        suggestions
            .into_iter()
            .map(|(token, prob)| {
                let confidence = self.config.base_confidence * prob;
                Correction::new(
                    CorrectionKind::Insertion,
                    byte_position,
                    byte_position,
                    "",
                    &token,
                )
                .with_confidence(confidence)
                .with_source(CorrectionSource::Grammar)
                .with_context(format!("Expected token at position {}", position))
            })
            .collect()
    }

    /// Generates deletion corrections for extra tokens.
    fn suggest_deletions(
        &self,
        token: &CodeToken,
        valid_tokens: &HashSet<String>,
    ) -> Vec<Correction> {
        if !self.config.suggest_deletions {
            return vec![];
        }

        let end_byte = token.byte_offset + token.text.len();

        // If no tokens are valid, suggest deletion
        if valid_tokens.is_empty() || !valid_tokens.contains(&token.text) {
            vec![Correction::new(
                CorrectionKind::Deletion,
                token.byte_offset,
                end_byte,
                &token.text,
                "",
            )
            .with_confidence(self.config.base_confidence * 0.7)
            .with_source(CorrectionSource::Grammar)
            .with_context("Unexpected token")]
        } else {
            vec![]
        }
    }

    /// Generates replacement corrections based on valid alternatives.
    fn suggest_replacements(
        &self,
        token: &CodeToken,
        valid_tokens: &HashSet<String>,
    ) -> Vec<Correction> {
        let mut corrections = Vec::new();
        let end_byte = token.byte_offset + token.text.len();

        for valid in valid_tokens {
            if valid == &token.text {
                continue;
            }

            // Calculate similarity-based confidence
            let similarity = self.string_similarity(&token.text, valid);
            if similarity < 0.3 {
                continue; // Too different
            }

            let prob = self.token_probability(valid, &[]);
            let confidence = self.config.base_confidence * similarity * (0.5 + 0.5 * prob);

            corrections.push(
                Correction::new(
                    CorrectionKind::Replacement,
                    token.byte_offset,
                    end_byte,
                    &token.text,
                    valid,
                )
                .with_confidence(confidence)
                .with_source(CorrectionSource::Grammar)
                .with_context(format!("Grammar suggests '{}'", valid)),
            );
        }

        corrections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        corrections.truncate(self.config.max_candidates);
        corrections
    }

    /// Simple string similarity metric (Jaccard on character bigrams).
    ///
    /// Builds HashSets directly from char iterators without intermediate Vec allocation.
    fn string_similarity(&self, a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Build bigrams directly from char iterator without Vec allocation.
        // Pre-size the HashSets to the upper bound of bigrams: an N-byte ASCII
        // string has at most N-1 bigrams; using `a.len()` overestimates for
        // multi-byte chars but never under-sizes, eliminating mid-loop rehash.
        let bigrams_a: HashSet<(char, char)> = {
            let mut set = HashSet::with_capacity(a.len().saturating_sub(1).max(1));
            let mut chars = a.chars().peekable();
            while let Some(c1) = chars.next() {
                if let Some(&c2) = chars.peek() {
                    set.insert((c1, c2));
                }
            }
            set
        };

        let bigrams_b: HashSet<(char, char)> = {
            let mut set = HashSet::with_capacity(b.len().saturating_sub(1).max(1));
            let mut chars = b.chars().peekable();
            while let Some(c1) = chars.next() {
                if let Some(&c2) = chars.peek() {
                    set.insert((c1, c2));
                }
            }
            set
        };

        // Handle single-character strings (no bigrams)
        if bigrams_a.is_empty() || bigrams_b.is_empty() {
            return if a == b { 1.0 } else { 0.0 };
        }

        let intersection = bigrams_a.intersection(&bigrams_b).count();
        let union = bigrams_a.union(&bigrams_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Returns the grammar.
    pub fn grammar(&self) -> &WeightedCFG {
        &self.grammar
    }

    /// Returns the language handler.
    pub fn language(&self) -> &L {
        &self.language
    }
}

impl<L: CodeLanguage + Send + Sync> CodeCorrector for GrammarCorrector<L> {
    fn correct_token(&self, token: &CodeToken, _context: &TokenContext) -> Vec<Correction> {
        let mut corrections = Vec::new();

        // For grammar correction without full context, we check against all valid start tokens.
        // In practice, the pipeline should provide token history for better accuracy.
        // Get valid tokens from initial state (empty context)
        let valid_tokens = self.valid_next_tokens(&[]);

        // Check if current token is valid
        if !valid_tokens.contains(&token.text) {
            // Suggest replacements
            corrections.extend(self.suggest_replacements(token, &valid_tokens));

            // Suggest deletions
            corrections.extend(self.suggest_deletions(token, &valid_tokens));
        }

        corrections
    }

    fn correct_range(&self, source: &str, start_byte: usize, end_byte: usize) -> Vec<Correction> {
        let text = &source[start_byte..end_byte];
        let token = CodeToken::new(text, start_byte, 0, 0, TokenType::Unknown, "unknown");

        let context = TokenContext::new(TokenType::Unknown);
        self.correct_token(&token, &context)
    }

    fn max_edit_distance(&self) -> usize {
        2 // Grammar corrections can involve larger structural changes
    }

    fn name(&self) -> &str {
        "GrammarCorrector"
    }
}

/// Type of syntax error detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntaxErrorType {
    /// Token is not valid at this position
    InvalidToken,
    /// Token is completely unexpected (no valid tokens possible)
    UnexpectedToken,
    /// Missing required token
    MissingToken,
    /// Unclosed bracket/delimiter
    UnclosedDelimiter,
}

/// A syntax error detected by the grammar checker.
#[derive(Debug, Clone)]
pub struct SyntaxError {
    /// Position in the token stream
    pub position: usize,
    /// The problematic token
    pub token: String,
    /// Valid tokens expected at this position
    pub expected: HashSet<String>,
    /// Type of error
    pub error_type: SyntaxErrorType,
}

impl SyntaxError {
    /// Returns a human-readable error message.
    pub fn message(&self) -> String {
        match self.error_type {
            SyntaxErrorType::InvalidToken => {
                if self.expected.len() <= 3 {
                    let expected: Vec<_> = self.expected.iter().take(3).collect();
                    format!(
                        "Invalid token '{}', expected one of: {}",
                        self.token,
                        expected
                            .iter()
                            .map(|s| format!("'{}'", s))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    format!(
                        "Invalid token '{}' (expected {} possible tokens)",
                        self.token,
                        self.expected.len()
                    )
                }
            }
            SyntaxErrorType::UnexpectedToken => {
                format!("Unexpected token '{}'", self.token)
            }
            SyntaxErrorType::MissingToken => {
                format!("Missing token before '{}'", self.token)
            }
            SyntaxErrorType::UnclosedDelimiter => {
                format!("Unclosed delimiter before '{}'", self.token)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grammar() -> WeightedCFG {
        let mut cfg = WeightedCFG::new("stmt");

        // stmt -> "if" "(" expr ")" stmt
        cfg.add_rule(
            Production::new(
                "stmt",
                vec![
                    Symbol::Terminal("if".to_string()),
                    Symbol::Terminal("(".to_string()),
                    Symbol::NonTerminal("expr".to_string()),
                    Symbol::Terminal(")".to_string()),
                    Symbol::NonTerminal("stmt".to_string()),
                ],
            ),
            0.3,
        );

        // stmt -> "while" "(" expr ")" stmt
        cfg.add_rule(
            Production::new(
                "stmt",
                vec![
                    Symbol::Terminal("while".to_string()),
                    Symbol::Terminal("(".to_string()),
                    Symbol::NonTerminal("expr".to_string()),
                    Symbol::Terminal(")".to_string()),
                    Symbol::NonTerminal("stmt".to_string()),
                ],
            ),
            0.2,
        );

        // stmt -> "return" expr ";"
        cfg.add_rule(
            Production::new(
                "stmt",
                vec![
                    Symbol::Terminal("return".to_string()),
                    Symbol::NonTerminal("expr".to_string()),
                    Symbol::Terminal(";".to_string()),
                ],
            ),
            0.3,
        );

        // stmt -> expr ";"
        cfg.add_rule(
            Production::new(
                "stmt",
                vec![
                    Symbol::NonTerminal("expr".to_string()),
                    Symbol::Terminal(";".to_string()),
                ],
            ),
            0.2,
        );

        // expr -> "x"
        cfg.add_rule(
            Production::new("expr", vec![Symbol::Terminal("x".to_string())]),
            0.5,
        );

        // expr -> "y"
        cfg.add_rule(
            Production::new("expr", vec![Symbol::Terminal("y".to_string())]),
            0.5,
        );

        cfg
    }

    // Mock language for testing
    #[derive(Debug, Clone, Default)]
    struct MockLanguage;

    impl CodeLanguage for MockLanguage {
        fn name(&self) -> &str {
            "mock"
        }
        fn display_name(&self) -> &str {
            "Mock"
        }
        fn tree_sitter_language(&self) -> tree_sitter::Language {
            panic!("Not implemented for tests")
        }
        fn keywords(&self) -> &[&str] {
            &["if", "else", "while", "return"]
        }
        fn special_tokens(&self) -> &[&str] {
            &[]
        }
        fn file_extensions(&self) -> &[&str] {
            &["mock"]
        }
        fn classify_token(&self, _token: &str, _node_kind: &str) -> TokenType {
            TokenType::Unknown
        }
        fn is_valid_identifier(&self, s: &str) -> bool {
            !s.is_empty()
        }
        fn builtin_types(&self) -> &[&str] {
            &[]
        }
        fn stdlib_functions(&self) -> &[&str] {
            &[]
        }
        fn comment_syntax(&self) -> crate::code::language::CommentSyntax {
            crate::code::language::CommentSyntax::default()
        }
        fn is_whitespace_significant(&self) -> bool {
            false
        }
    }

    #[test]
    #[ignore = "Requires full Earley parser integration - test grammar constraint separately"]
    fn test_grammar_corrector_valid_tokens() {
        let lang = Arc::new(MockLanguage);
        let grammar = create_test_grammar();
        let corrector = GrammarCorrector::with_defaults(lang, grammar);

        // At the start, valid tokens should include statement starters
        let valid = corrector.valid_next_tokens(&[]);
        assert!(valid.contains("if") || valid.contains("while") || valid.contains("return"));
    }

    #[test]
    #[ignore = "Requires full Earley parser integration - test grammar constraint separately"]
    fn test_grammar_corrector_completions() {
        let lang = Arc::new(MockLanguage);
        let grammar = create_test_grammar();
        let corrector = GrammarCorrector::with_defaults(lang, grammar);

        // After "if (", we should expect expressions
        let completions = corrector.suggest_completions(&["if", "("], 5);
        // Should suggest valid expressions
        assert!(!completions.is_empty());
    }

    #[test]
    fn test_syntax_error_message() {
        let mut expected = HashSet::new();
        expected.insert("if".to_string());
        expected.insert("while".to_string());

        let error = SyntaxError {
            position: 0,
            token: "fi".to_string(),
            expected,
            error_type: SyntaxErrorType::InvalidToken,
        };

        let msg = error.message();
        assert!(msg.contains("Invalid token"));
        assert!(msg.contains("fi"));
    }
}
