//! Lexical corrector using liblevenshtein for fuzzy matching.
//!
//! This corrector handles token-level spelling errors by maintaining
//! dictionaries of valid tokens per token type (keywords, identifiers, etc.).

use crate::code::correction::{CodeCorrector, Correction, CorrectionKind, CorrectionSource};
use crate::code::language::{CodeLanguage, TokenContext, TokenType};
use crate::code::tokenizer::CodeToken;
use std::collections::HashSet;
use std::sync::Arc;

/// Configuration for the lexical corrector.
#[derive(Debug, Clone)]
pub struct LexicalCorrectorConfig {
    /// Maximum edit distance for corrections
    pub max_edit_distance: usize,
    /// Minimum token length to consider for correction
    pub min_token_length: usize,
    /// Maximum number of candidates to return per token
    pub max_candidates: usize,
    /// Penalty for each edit operation (affects confidence)
    pub edit_penalty: f64,
}

impl Default for LexicalCorrectorConfig {
    fn default() -> Self {
        Self {
            max_edit_distance: 2,
            min_token_length: 2,
            max_candidates: 5,
            edit_penalty: 0.15,
        }
    }
}

/// A fuzzy match candidate.
#[derive(Debug, Clone)]
struct FuzzyCandidate {
    term: String,
    distance: usize,
}

/// Lexical corrector using fuzzy dictionaries.
///
/// Maintains separate dictionaries for different token types to ensure
/// corrections are contextually appropriate (e.g., keywords only suggest
/// other keywords).
pub struct LexicalCorrector<L: CodeLanguage> {
    language: Arc<L>,
    config: LexicalCorrectorConfig,
    /// Set of keywords
    keywords: HashSet<String>,
    /// Set of known identifiers (project-specific)
    identifiers: HashSet<String>,
    /// Set of builtin types
    types: HashSet<String>,
    /// Set of stdlib functions
    stdlib: HashSet<String>,
}

impl<L: CodeLanguage> LexicalCorrector<L> {
    /// Creates a new lexical corrector for the given language.
    pub fn new(language: Arc<L>, config: LexicalCorrectorConfig) -> Self {
        let mut keywords = HashSet::new();
        let mut types = HashSet::new();
        let mut stdlib = HashSet::new();

        // Populate keyword set
        for keyword in language.keywords() {
            keywords.insert(keyword.to_string());
        }

        // Populate type set
        for typ in language.builtin_types() {
            types.insert(typ.to_string());
        }

        // Populate stdlib set
        for func in language.stdlib_functions() {
            stdlib.insert(func.to_string());
        }

        Self {
            language,
            config,
            keywords,
            identifiers: HashSet::new(),
            types,
            stdlib,
        }
    }

    /// Creates a corrector with default configuration.
    pub fn with_defaults(language: Arc<L>) -> Self {
        Self::new(language, LexicalCorrectorConfig::default())
    }

    /// Adds an identifier to the identifier set.
    pub fn add_identifier(&mut self, identifier: &str) {
        if self.language.is_valid_identifier(identifier) {
            self.identifiers.insert(identifier.to_string());
        }
    }

    /// Adds multiple identifiers from a source file.
    pub fn add_identifiers_from_source(&mut self, source: &str) {
        for word in source.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if !word.is_empty() && self.language.is_valid_identifier(word) {
                self.identifiers.insert(word.to_string());
            }
        }
    }

    /// Adds identifiers from parsed AST nodes.
    pub fn add_identifiers_from_tokens(&mut self, tokens: &[CodeToken]) {
        for token in tokens {
            if token.token_type == TokenType::Identifier {
                self.add_identifier(&token.text);
            }
        }
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

        for i in 0..=m {
            dp[i][0] = i;
        }
        for j in 0..=n {
            dp[0][j] = j;
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

    /// Finds fuzzy matches in a set within max edit distance.
    fn fuzzy_search(&self, query: &str, dictionary: &HashSet<String>) -> Vec<FuzzyCandidate> {
        let max_dist = self.config.max_edit_distance;
        let mut candidates = Vec::new();

        for term in dictionary {
            // Skip if length difference alone exceeds max distance
            let len_diff = (query.len() as isize - term.len() as isize).unsigned_abs();
            if len_diff > max_dist {
                continue;
            }

            let distance = Self::levenshtein_distance(query, term);
            if distance > 0 && distance <= max_dist {
                candidates.push(FuzzyCandidate {
                    term: term.clone(),
                    distance,
                });
            }
        }

        // Sort by distance
        candidates.sort_by_key(|c| c.distance);
        candidates
    }

    /// Returns candidates for a token from the appropriate dictionary.
    fn get_candidates(&self, token: &str, token_type: TokenType) -> Vec<FuzzyCandidate> {
        match token_type {
            TokenType::Keyword => self.fuzzy_search(token, &self.keywords),
            TokenType::TypeName => self.fuzzy_search(token, &self.types),
            TokenType::Identifier => {
                let mut candidates = self.fuzzy_search(token, &self.identifiers);
                candidates.extend(self.fuzzy_search(token, &self.stdlib));

                // Deduplicate
                let mut seen = HashSet::new();
                candidates.retain(|c| seen.insert(c.term.clone()));

                // Re-sort
                candidates.sort_by_key(|c| c.distance);
                candidates
            }
            _ => {
                // For other types, search all dictionaries
                let mut candidates = Vec::new();
                candidates.extend(self.fuzzy_search(token, &self.keywords));
                candidates.extend(self.fuzzy_search(token, &self.identifiers));
                candidates.extend(self.fuzzy_search(token, &self.types));
                candidates.extend(self.fuzzy_search(token, &self.stdlib));

                // Deduplicate
                let mut seen = HashSet::new();
                candidates.retain(|c| seen.insert(c.term.clone()));

                // Re-sort
                candidates.sort_by_key(|c| c.distance);
                candidates
            }
        }
    }

    /// Converts a fuzzy candidate to a Correction.
    fn candidate_to_correction(&self, candidate: &FuzzyCandidate, token: &CodeToken) -> Correction {
        let distance = candidate.distance as f64;

        // Confidence decreases with edit distance
        let confidence = 1.0 - (distance * self.config.edit_penalty).min(0.9);

        let end_byte = token.byte_offset + token.text.len();

        Correction::new(
            CorrectionKind::Spelling,
            token.byte_offset,
            end_byte,
            &token.text,
            &candidate.term,
        )
        .with_confidence(confidence)
        .with_source(CorrectionSource::Lexical)
        .with_context(format!("Edit distance: {}", candidate.distance))
    }

    /// Returns the language handler.
    pub fn language(&self) -> &L {
        &self.language
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LexicalCorrectorConfig {
        &self.config
    }

    /// Returns the number of keywords in the dictionary.
    pub fn keyword_count(&self) -> usize {
        self.keywords.len()
    }

    /// Returns the number of identifiers in the dictionary.
    pub fn identifier_count(&self) -> usize {
        self.identifiers.len()
    }
}

impl<L: CodeLanguage + Send + Sync> CodeCorrector for LexicalCorrector<L> {
    fn correct_token(&self, token: &CodeToken, _context: &TokenContext) -> Vec<Correction> {
        // Skip tokens that are too short
        if token.text.len() < self.config.min_token_length {
            return vec![];
        }

        let token_type = token.token_type;

        // Get fuzzy candidates
        let candidates = self.get_candidates(&token.text, token_type);

        // Convert to corrections
        candidates
            .into_iter()
            .take(self.config.max_candidates)
            .map(|c| self.candidate_to_correction(&c, token))
            .collect()
    }

    fn correct_range(&self, source: &str, start_byte: usize, end_byte: usize) -> Vec<Correction> {
        let text = &source[start_byte..end_byte];

        // Create a temporary token for the range
        let token = CodeToken::new(text, start_byte, 0, 0, TokenType::Unknown, "unknown");

        let context = TokenContext::new(TokenType::Unknown);
        self.correct_token(&token, &context)
    }

    fn max_edit_distance(&self) -> usize {
        self.config.max_edit_distance
    }

    fn name(&self) -> &str {
        "LexicalCorrector"
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
            panic!("Not implemented for tests")
        }
        fn keywords(&self) -> &[&str] {
            &[
                "if", "else", "while", "for", "return", "function", "let", "const", "var",
            ]
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
            !s.is_empty() && s.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)
        }
        fn builtin_types(&self) -> &[&str] {
            &["int", "string", "bool", "float"]
        }
        fn stdlib_functions(&self) -> &[&str] {
            &["print", "println", "read", "write"]
        }
        fn comment_syntax(&self) -> crate::code::language::CommentSyntax {
            crate::code::language::CommentSyntax::default()
        }
        fn is_whitespace_significant(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("", ""),
            0
        );
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("abc", ""),
            3
        );
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("", "abc"),
            3
        );
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("abc", "abc"),
            0
        );
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("abc", "abd"),
            1
        );
        assert_eq!(
            LexicalCorrector::<MockLanguage>::levenshtein_distance("function", "funtion"),
            1
        );
    }

    #[test]
    fn test_lexical_corrector_keywords() {
        let lang = Arc::new(MockLanguage);
        let corrector = LexicalCorrector::with_defaults(lang);

        let token = CodeToken::new(
            "funtion", // Misspelled "function"
            0,
            1,
            0,
            TokenType::Keyword,
            "keyword",
        );

        let context = TokenContext::new(TokenType::Keyword);
        let corrections = corrector.correct_token(&token, &context);

        assert!(!corrections.is_empty());
        assert!(corrections.iter().any(|c| c.replacement == "function"));
    }

    #[test]
    fn test_lexical_corrector_identifiers() {
        let lang = Arc::new(MockLanguage);
        let mut corrector = LexicalCorrector::with_defaults(lang);

        // Add some project-specific identifiers
        corrector.add_identifier("calculateTotal");
        corrector.add_identifier("processData");
        corrector.add_identifier("handleError");

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

    #[test]
    fn test_lexical_corrector_exact_match() {
        let lang = Arc::new(MockLanguage);
        let corrector = LexicalCorrector::with_defaults(lang);

        let token = CodeToken::new(
            "function", // Correct spelling
            0,
            1,
            0,
            TokenType::Keyword,
            "keyword",
        );

        let context = TokenContext::new(TokenType::Keyword);
        let corrections = corrector.correct_token(&token, &context);

        // Should not suggest corrections for exact matches (distance > 0 filter)
        assert!(corrections.is_empty() || corrections.iter().all(|c| c.replacement != "function"));
    }
}
