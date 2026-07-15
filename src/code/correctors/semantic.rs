//! Semantic corrector using GNN and code embeddings.
//!
//! This corrector analyzes code semantics using:
//! - Code Property Graphs (CPG) for structural analysis
//! - Graph Neural Networks for semantic pattern detection
//! - Code embeddings for similarity-based suggestions

use crate::code::ast::ParsedCode;
use crate::code::correction::{CodeCorrector, Correction, CorrectionKind, CorrectionSource};
use crate::code::cpg::CodePropertyGraph;
use crate::code::gnn::{GnnConfig, GnnSemanticScorer, IssueType, SemanticIssue};
use crate::code::language::{CodeLanguage, TokenContext, TokenType};
use crate::code::tokenizer::CodeToken;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the semantic corrector.
#[derive(Debug, Clone)]
pub struct SemanticCorrectorConfig {
    /// Minimum confidence threshold for reporting issues
    pub min_confidence: f64,
    /// Maximum candidates per issue
    pub max_candidates: usize,
    /// Whether to check for variable misuse
    pub check_variable_misuse: bool,
    /// Whether to check for unused bindings
    pub check_unused_bindings: bool,
    /// Whether to check for type errors (when type info available)
    pub check_type_errors: bool,
    /// GNN configuration
    pub gnn_config: GnnConfig,
}

impl Default for SemanticCorrectorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_candidates: 5,
            check_variable_misuse: true,
            check_unused_bindings: true,
            check_type_errors: true,
            gnn_config: GnnConfig::default(),
        }
    }
}

/// Semantic corrector using GNN analysis.
///
/// This corrector builds a Code Property Graph from parsed code and uses
/// graph neural network-based analysis to detect semantic issues like:
/// - Variable misuse (wrong variable in context)
/// - Unused bindings
/// - Type mismatches
/// - API misuse patterns
pub struct SemanticCorrector<L: CodeLanguage> {
    language: Arc<L>,
    config: SemanticCorrectorConfig,
    gnn_scorer: GnnSemanticScorer,
    /// Cache of variable names seen in the project
    known_variables: HashMap<String, VariableInfo>,
    /// Cache of function signatures
    known_functions: HashMap<String, FunctionInfo>,
}

/// Information about a known variable.
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Inferred or declared type (if known)
    pub type_name: Option<String>,
    /// Scope level where defined
    pub scope_level: usize,
    /// Number of times used
    pub use_count: usize,
}

/// Information about a known function.
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Parameter types (if known)
    pub param_types: Vec<Option<String>>,
    /// Return type (if known)
    pub return_type: Option<String>,
    /// Number of parameters
    pub arity: usize,
}

impl<L: CodeLanguage> SemanticCorrector<L> {
    /// Creates a new semantic corrector.
    pub fn new(language: Arc<L>, config: SemanticCorrectorConfig) -> Self {
        let gnn_scorer = GnnSemanticScorer::new(config.gnn_config.clone());

        Self {
            language,
            config,
            gnn_scorer,
            known_variables: HashMap::new(),
            known_functions: HashMap::new(),
        }
    }

    /// Creates a corrector with default configuration.
    pub fn with_defaults(language: Arc<L>) -> Self {
        Self::new(language, SemanticCorrectorConfig::default())
    }

    /// Registers a variable from the codebase.
    pub fn register_variable(
        &mut self,
        name: String,
        type_name: Option<String>,
        scope_level: usize,
    ) {
        let info = self
            .known_variables
            .entry(name.clone())
            .or_insert(VariableInfo {
                name,
                type_name: None,
                scope_level,
                use_count: 0,
            });
        if type_name.is_some() {
            info.type_name = type_name;
        }
        info.use_count += 1;
    }

    /// Registers a function from the codebase.
    pub fn register_function(&mut self, name: String, arity: usize, return_type: Option<String>) {
        self.known_functions.insert(
            name.clone(),
            FunctionInfo {
                name,
                param_types: vec![None; arity],
                return_type,
                arity,
            },
        );
    }

    /// Analyzes a Code Property Graph for semantic issues.
    pub fn analyze_cpg(&self, cpg: &CodePropertyGraph) -> Vec<SemanticIssue> {
        let mut issues = self.gnn_scorer.detect_issues(cpg);

        // Filter by confidence
        issues.retain(|issue| issue.confidence >= self.config.min_confidence);

        // Filter by enabled checks
        issues.retain(|issue| match issue.issue_type {
            IssueType::VariableMisuse => self.config.check_variable_misuse,
            IssueType::UnusedBinding => self.config.check_unused_bindings,
            IssueType::TypeError => self.config.check_type_errors,
            _ => true,
        });

        issues
    }

    /// Finds variable misuse candidates using GNN analysis.
    pub fn find_variable_misuse(
        &self,
        cpg: &CodePropertyGraph,
        variable_name: &str,
        node_idx: usize,
    ) -> Vec<(String, f64)> {
        // Get candidates from GNN
        let mut candidates = self.gnn_scorer.variable_misuse_candidates(cpg, node_idx);

        // Enhance with known variables from the project
        for (name, info) in &self.known_variables {
            if name == variable_name {
                continue;
            }

            // Calculate similarity based on:
            // 1. Name similarity
            // 2. Type compatibility (if known)
            // 3. Usage frequency

            let name_sim = self.name_similarity(variable_name, name);
            let usage_boost = (info.use_count as f64).ln().max(0.0) / 10.0;
            let score = name_sim * 0.7 + usage_boost * 0.3;

            if score > 0.3 && !candidates.iter().any(|(n, _)| n == name) {
                candidates.push((name.clone(), score));
            }
        }

        // Sort by score
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.config.max_candidates);
        candidates
    }

    /// Calculates name similarity using Levenshtein distance.
    fn name_similarity(&self, a: &str, b: &str) -> f64 {
        if a == b {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let distance = Self::levenshtein_distance(a, b);
        let max_len = a.len().max(b.len());

        // Convert distance to similarity (0.0 to 1.0)
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Computes the Levenshtein distance between two strings.
    fn levenshtein_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        let mut dp = vec![vec![0usize; n + 1]; m + 1];

        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in dp[0].iter_mut().enumerate() {
            *cell = j;
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }

    /// Converts a semantic issue to corrections.
    fn issue_to_corrections(
        &self,
        issue: &SemanticIssue,
        cpg: &CodePropertyGraph,
        source: &str,
    ) -> Vec<Correction> {
        let mut corrections = Vec::new();

        // Find the node in the CPG
        let node = cpg.all_nodes().find(|n| n.id == issue.node_idx);

        let (start_byte, end_byte, original) = if let Some(node) = node {
            (
                node.location.0,
                node.location.1,
                source
                    .get(node.location.0..node.location.1)
                    .unwrap_or("")
                    .to_string(),
            )
        } else {
            return corrections;
        };

        match issue.issue_type {
            IssueType::VariableMisuse => {
                // Get candidates for variable replacement
                let candidates = self.find_variable_misuse(cpg, &original, issue.node_idx);

                for (replacement, score) in candidates {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::VariableMisuse,
                            start_byte,
                            end_byte,
                            &original,
                            &replacement,
                        )
                        .with_confidence(issue.confidence * score)
                        .with_source(CorrectionSource::Neural)
                        .with_context(format!(
                            "Possible variable misuse: did you mean '{}'?",
                            replacement
                        )),
                    );
                }
            }
            IssueType::UnusedBinding => {
                // Suggest removal or usage
                if let Some(suggestion) = &issue.suggestion {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::Deletion,
                            start_byte,
                            end_byte,
                            &original,
                            "",
                        )
                        .with_confidence(issue.confidence * 0.6)
                        .with_source(CorrectionSource::DataFlow)
                        .with_context(suggestion.clone()),
                    );
                }
            }
            IssueType::TypeError => {
                if let Some(suggestion) = &issue.suggestion {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::TypeError,
                            start_byte,
                            end_byte,
                            &original,
                            suggestion,
                        )
                        .with_confidence(issue.confidence)
                        .with_source(CorrectionSource::TypeInference)
                        .with_context("Type mismatch detected".to_string()),
                    );
                }
            }
            IssueType::MissingErrorHandling => {
                // Suggest wrapping in error handling
                corrections.push(
                    Correction::new(
                        CorrectionKind::Other,
                        start_byte,
                        end_byte,
                        &original,
                        &original, // Would need language-specific error handling
                    )
                    .with_confidence(issue.confidence * 0.5)
                    .with_source(CorrectionSource::ControlFlow)
                    .with_context("Consider adding error handling".to_string()),
                );
            }
            _ => {
                // Generic issue - use suggestion if available
                if let Some(suggestion) = &issue.suggestion {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::Other,
                            start_byte,
                            end_byte,
                            &original,
                            suggestion,
                        )
                        .with_confidence(issue.confidence)
                        .with_source(CorrectionSource::Neural)
                        .with_context(format!("{:?} detected", issue.issue_type)),
                    );
                }
            }
        }

        corrections
    }

    /// Analyzes parsed code and returns corrections.
    pub fn analyze_parsed(&self, parsed: &ParsedCode, cpg: &CodePropertyGraph) -> Vec<Correction> {
        let issues = self.analyze_cpg(cpg);
        let source = &parsed.source;

        let mut all_corrections = Vec::new();
        for issue in &issues {
            let corrections = self.issue_to_corrections(issue, cpg, source);
            all_corrections.extend(corrections);
        }

        // Sort by confidence
        all_corrections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_corrections
    }

    /// Returns the GNN scorer.
    pub fn gnn_scorer(&self) -> &GnnSemanticScorer {
        &self.gnn_scorer
    }

    /// Returns the language handler.
    pub fn language(&self) -> &L {
        &self.language
    }
}

impl<L: CodeLanguage + Send + Sync> CodeCorrector for SemanticCorrector<L> {
    fn correct_token(&self, token: &CodeToken, _context: &TokenContext) -> Vec<Correction> {
        // Semantic correction requires full AST context, so token-level
        // correction is limited. We mainly check against known variables.
        let mut corrections = Vec::new();

        if token.token_type == TokenType::Identifier {
            // Check if this identifier exists in known variables
            if !self.known_variables.contains_key(&token.text) {
                // Find similar known variables
                let mut candidates: Vec<_> = self
                    .known_variables
                    .keys()
                    .map(|name| {
                        let sim = self.name_similarity(&token.text, name);
                        (name.clone(), sim)
                    })
                    .filter(|(_, sim)| *sim > 0.5)
                    .collect();

                candidates
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let end_byte = token.byte_offset + token.text.len();
                for (replacement, score) in candidates.into_iter().take(self.config.max_candidates)
                {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::VariableMisuse,
                            token.byte_offset,
                            end_byte,
                            &token.text,
                            &replacement,
                        )
                        .with_confidence(score * 0.7)
                        .with_source(CorrectionSource::Neural)
                        .with_context(format!(
                            "Unknown identifier, did you mean '{}'?",
                            replacement
                        )),
                    );
                }
            }
        }

        corrections
    }

    fn correct_range(&self, source: &str, start_byte: usize, end_byte: usize) -> Vec<Correction> {
        let text = &source[start_byte..end_byte];
        let token = CodeToken::new(text, start_byte, 0, 0, TokenType::Identifier, "identifier");

        let context = TokenContext::new(TokenType::Identifier);
        self.correct_token(&token, &context)
    }

    fn max_edit_distance(&self) -> usize {
        3 // Semantic corrections can involve larger changes
    }

    fn name(&self) -> &str {
        "SemanticCorrector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            tree_sitter_rust::LANGUAGE.into()
        }
        fn keywords(&self) -> &[&str] {
            &[]
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
    fn test_name_similarity() {
        let lang = Arc::new(MockLanguage);
        let corrector = SemanticCorrector::with_defaults(lang);

        // Identical names
        assert!((corrector.name_similarity("count", "count") - 1.0).abs() < 0.01);

        // Similar names
        let sim = corrector.name_similarity("count", "counter");
        assert!(sim > 0.5);

        // Different names
        let sim = corrector.name_similarity("foo", "bar");
        assert!(sim < 0.3);
    }

    #[test]
    fn test_variable_registration() {
        let lang = Arc::new(MockLanguage);
        let mut corrector = SemanticCorrector::with_defaults(lang);

        corrector.register_variable("userCount".to_string(), Some("int".to_string()), 0);
        corrector.register_variable("userName".to_string(), Some("string".to_string()), 0);

        assert!(corrector.known_variables.contains_key("userCount"));
        assert!(corrector.known_variables.contains_key("userName"));
        assert_eq!(
            corrector.known_variables["userCount"].type_name,
            Some("int".to_string())
        );
    }

    #[test]
    fn test_unknown_identifier_correction() {
        let lang = Arc::new(MockLanguage);
        let mut corrector = SemanticCorrector::with_defaults(lang);

        // Register some known variables
        corrector.register_variable("calculateTotal".to_string(), None, 0);
        corrector.register_variable("calculateAverage".to_string(), None, 0);

        // Try to correct a typo
        let token = CodeToken::new(
            "calulateTotal", // Misspelled
            0,
            1,
            0,
            TokenType::Identifier,
            "identifier",
        );

        let context = TokenContext::new(TokenType::Identifier);
        let corrections = corrector.correct_token(&token, &context);

        assert!(!corrections.is_empty());
        assert!(corrections
            .iter()
            .any(|c| c.replacement == "calculateTotal"));
    }
}
