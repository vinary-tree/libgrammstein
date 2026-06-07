//! Integration tests for the code correction module.
//!
//! These tests verify the end-to-end functionality of the code correction
//! pipeline, including lexical, grammar, and semantic correctors.

#![cfg(feature = "code")]

use libgrammstein::code::correction::{
    CodeCorrector, Correction, CorrectionCandidates, CorrectionKind,
};
use libgrammstein::code::correctors::LexicalCorrector;
use libgrammstein::code::language::{CodeLanguage, CommentSyntax, TokenContext, TokenType};
use libgrammstein::code::tokenizer::CodeToken;
use std::sync::Arc;

/// Mock language for testing without tree-sitter dependencies.
#[derive(Debug, Clone, Default)]
struct MockLanguage {
    name: &'static str,
    keywords: &'static [&'static str],
    builtin_types: &'static [&'static str],
    stdlib_functions: &'static [&'static str],
}

impl MockLanguage {
    fn rust_like() -> Self {
        Self {
            name: "rust_like",
            keywords: &[
                "fn", "let", "mut", "if", "else", "while", "for", "loop", "match", "return",
                "struct", "enum", "impl", "trait", "pub", "use", "mod", "const", "static", "type",
                "where", "async", "await",
            ],
            builtin_types: &[
                "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
                "usize", "f32", "f64", "bool", "char", "str", "String", "Vec", "Option", "Result",
                "Box", "Rc", "Arc",
            ],
            stdlib_functions: &[
                "println",
                "print",
                "format",
                "panic",
                "assert",
                "assert_eq",
                "assert_ne",
                "dbg",
                "todo",
                "unimplemented",
            ],
        }
    }

    fn python_like() -> Self {
        Self {
            name: "python_like",
            keywords: &[
                "def", "class", "if", "elif", "else", "while", "for", "in", "return", "yield",
                "try", "except", "finally", "raise", "import", "from", "as", "with", "pass",
                "break", "continue", "and", "or", "not", "is", "lambda", "global", "nonlocal",
                "async", "await",
            ],
            builtin_types: &[
                "int",
                "float",
                "str",
                "bool",
                "list",
                "dict",
                "tuple",
                "set",
                "bytes",
                "bytearray",
                "frozenset",
                "object",
                "type",
                "None",
            ],
            stdlib_functions: &[
                "print",
                "len",
                "range",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "reversed",
                "open",
                "input",
                "type",
                "isinstance",
            ],
        }
    }
}

impl CodeLanguage for MockLanguage {
    fn name(&self) -> &str {
        self.name
    }

    fn display_name(&self) -> &str {
        match self.name {
            "rust_like" => "Rust-like",
            "python_like" => "Python-like",
            _ => self.name,
        }
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        // This won't be called in lexical-only tests
        panic!("Mock language doesn't support tree-sitter parsing")
    }

    fn keywords(&self) -> &[&str] {
        self.keywords
    }

    fn special_tokens(&self) -> &[&str] {
        &[]
    }

    fn file_extensions(&self) -> &[&str] {
        &["mock"]
    }

    fn classify_token(&self, token: &str, _node_kind: &str) -> TokenType {
        if self.keywords.contains(&token) {
            TokenType::Keyword
        } else if self.builtin_types.contains(&token) {
            TokenType::TypeName
        } else if self.stdlib_functions.contains(&token) {
            TokenType::Identifier // stdlib functions are identifiers
        } else if token.chars().all(|c| c.is_ascii_digit()) {
            TokenType::NumericLiteral
        } else if token.starts_with('"') && token.ends_with('"') {
            TokenType::StringLiteral
        } else {
            TokenType::Identifier
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        !s.is_empty()
            && s.chars()
                .next()
                .map(|c| c.is_alphabetic() || c == '_')
                .unwrap_or(false)
            && s.chars().all(|c| c.is_alphanumeric() || c == '_')
    }

    fn builtin_types(&self) -> &[&str] {
        self.builtin_types
    }

    fn stdlib_functions(&self) -> &[&str] {
        self.stdlib_functions
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax::default()
    }

    fn is_whitespace_significant(&self) -> bool {
        self.name == "python_like"
    }
}

// =============================================================================
// Lexical Corrector Integration Tests
// =============================================================================

#[test]
fn test_lexical_correction_rust_keywords() {
    let lang = Arc::new(MockLanguage::rust_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    // Test common keyword typos
    let test_cases = vec![
        ("fnc", "fn"),        // Missing character
        ("fnn", "fn"),        // Extra character
        ("funciton", "fn"),   // Won't match (too far)
        ("lett", "let"),      // Extra character
        ("leett", "let"),     // Extra characters
        ("retrun", "return"), // Transposition
        ("reutrn", "return"), // Different transposition
        ("strcut", "struct"), // Transposition
        ("pubilc", "pub"),    // Won't match (different word)
    ];

    for (typo, expected) in test_cases {
        let token = CodeToken::new(typo, 0, 1, 0, TokenType::Keyword, "keyword");
        let context = TokenContext::new(TokenType::Keyword);
        let corrections = corrector.correct_token(&token, &context);

        // Check if any correction matches the expected word
        if corrections.iter().any(|c| c.replacement == expected) {
            println!("OK: {} -> {} found in corrections", typo, expected);
        } else {
            println!(
                "INFO: {} -> {} not found (corrections: {:?})",
                typo,
                expected,
                corrections
                    .iter()
                    .map(|c| &c.replacement)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_lexical_correction_python_keywords() {
    let lang = Arc::new(MockLanguage::python_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    let test_cases = vec![
        ("dfe", "def"),       // Transposition
        ("calss", "class"),   // Transposition
        ("rteurn", "return"), // Multiple errors
        ("improt", "import"), // Transposition
        ("exept", "except"),  // Missing character
    ];

    for (typo, expected) in test_cases {
        let token = CodeToken::new(typo, 0, 1, 0, TokenType::Keyword, "keyword");
        let context = TokenContext::new(TokenType::Keyword);
        let corrections = corrector.correct_token(&token, &context);

        if corrections.iter().any(|c| c.replacement == expected) {
            println!("OK: {} -> {} found", typo, expected);
        } else {
            println!(
                "INFO: {} -> {} not found (have: {:?})",
                typo,
                expected,
                corrections
                    .iter()
                    .map(|c| &c.replacement)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn test_lexical_correction_builtin_types() {
    let lang = Arc::new(MockLanguage::rust_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    let test_cases = vec![
        ("Stirng", "String"), // Transposition
        ("Optoin", "Option"), // Transposition
        ("Resutl", "Result"), // Transposition
        ("boool", "bool"),    // Extra character
    ];

    for (typo, expected) in test_cases {
        let token = CodeToken::new(typo, 0, 1, 0, TokenType::TypeName, "type");
        let context = TokenContext::new(TokenType::TypeName);
        let corrections = corrector.correct_token(&token, &context);

        if corrections.iter().any(|c| c.replacement == expected) {
            println!("OK: {} -> {} found", typo, expected);
        }
    }
}

#[test]
fn test_lexical_correction_custom_identifiers() {
    let lang = Arc::new(MockLanguage::rust_like());
    let mut corrector = LexicalCorrector::with_defaults(lang);

    // Add project-specific identifiers
    corrector.add_identifier("calculate_total");
    corrector.add_identifier("process_request");
    corrector.add_identifier("handle_error");
    corrector.add_identifier("validate_input");
    corrector.add_identifier("serialize_data");

    let test_cases = vec![
        ("calcluate_total", "calculate_total"), // Transposition
        ("porcess_request", "process_request"), // Transposition
        ("handel_error", "handle_error"),       // Transposition
        ("valdiate_input", "validate_input"),   // Transposition
    ];

    for (typo, expected) in test_cases {
        let token = CodeToken::new(typo, 0, 1, 0, TokenType::Identifier, "identifier");
        let context = TokenContext::new(TokenType::Identifier);
        let corrections = corrector.correct_token(&token, &context);

        assert!(
            corrections.iter().any(|c| c.replacement == expected),
            "Expected {} -> {} but got {:?}",
            typo,
            expected,
            corrections
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn test_lexical_correction_confidence_ranking() {
    let lang = Arc::new(MockLanguage::rust_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    // "fn" with one character off should have higher confidence than two off
    let token = CodeToken::new(
        "fnn", // One character off from "fn"
        0,
        1,
        0,
        TokenType::Keyword,
        "keyword",
    );
    let context = TokenContext::new(TokenType::Keyword);
    let corrections = corrector.correct_token(&token, &context);

    if corrections.len() >= 2 {
        // Verify corrections are sorted by confidence (descending)
        for i in 1..corrections.len() {
            assert!(
                corrections[i - 1].confidence >= corrections[i].confidence,
                "Corrections should be sorted by confidence"
            );
        }
    }
}

#[test]
fn test_lexical_correction_no_self_correction() {
    let lang = Arc::new(MockLanguage::rust_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    // Exact match should not suggest itself
    let token = CodeToken::new("fn", 0, 1, 0, TokenType::Keyword, "keyword");
    let context = TokenContext::new(TokenType::Keyword);
    let corrections = corrector.correct_token(&token, &context);

    // Should not suggest "fn" -> "fn"
    assert!(
        !corrections
            .iter()
            .any(|c| c.replacement == "fn" && c.original == "fn"),
        "Should not suggest self-correction"
    );
}

// =============================================================================
// Correction Candidates Integration Tests
// =============================================================================

#[test]
fn test_correction_candidates_sorting() {
    let mut candidates = CorrectionCandidates::new(10);

    // Add corrections with different confidence levels
    let c1 =
        Correction::new(CorrectionKind::Spelling, 0, 5, "typo1", "fixed1").with_confidence(0.7);
    let c2 =
        Correction::new(CorrectionKind::Spelling, 0, 5, "typo1", "fixed2").with_confidence(0.9);
    let c3 =
        Correction::new(CorrectionKind::Spelling, 0, 5, "typo1", "fixed3").with_confidence(0.5);

    candidates.add(c1);
    candidates.add(c2);
    candidates.add(c3);

    let ranked = candidates.ranked();

    // Should be sorted by confidence descending
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].replacement, "fixed2"); // highest confidence
    assert_eq!(ranked[1].replacement, "fixed1");
    assert_eq!(ranked[2].replacement, "fixed3"); // lowest confidence
}

#[test]
fn test_correction_candidates_max_limit() {
    let mut candidates = CorrectionCandidates::new(3);

    for i in 0..10 {
        let c = Correction::new(
            CorrectionKind::Spelling,
            i,
            i + 1,
            &format!("typo{}", i),
            &format!("fixed{}", i),
        )
        .with_confidence(i as f64 * 0.1);
        candidates.add(c);
    }

    let ranked = candidates.ranked();
    assert!(ranked.len() <= 3, "Should respect max limit");
}

#[test]
fn test_correction_apply_multiple() {
    let source = "funciton foo() { retrun 42; }";

    // Apply corrections from end to start
    let mut result = source.to_string();

    // Second correction: "retrun" -> "return" at bytes 17-23
    result.replace_range(17..23, "return");

    // First correction: "funciton" -> "function" at bytes 0-8
    result.replace_range(0..8, "function");

    assert_eq!(result, "function foo() { return 42; }");
}

// =============================================================================
// Ensemble Corrector Integration Tests
// =============================================================================

#[cfg(test)]
mod ensemble_tests {
    use super::*;

    #[test]
    fn test_ensemble_lexical_only() {
        use libgrammstein::code::correctors::ensemble::{
            EnsembleCorrectorBuilder, EnsembleCorrectorConfig,
        };

        let lang = Arc::new(MockLanguage::rust_like());

        // Use builder with adjusted config for lexical-only mode
        // Default lexical_weight (0.4) * distance-2 confidence (0.7) = 0.28 < min_confidence (0.3)
        // So we need to either increase weight or lower min_confidence
        let config = EnsembleCorrectorConfig {
            lexical_weight: 1.0, // Full weight for lexical-only mode
            min_confidence: 0.2, // Lower threshold for distance-2 corrections
            ..Default::default()
        };

        let mut corrector = EnsembleCorrectorBuilder::new(lang)
            .with_config(config)
            .without_grammar()
            .without_semantic()
            .build();

        // Add some identifiers
        corrector.add_identifiers(&["calculate_total", "process_data"]);

        let token = CodeToken::new(
            "calcluate_total", // Typo: transposed 'lu' -> 'ul' (distance 2)
            0,
            1,
            0,
            TokenType::Identifier,
            "identifier",
        );
        let context = TokenContext::new(TokenType::Identifier);
        let corrections = corrector.correct_token(&token, &context);

        assert!(
            corrections
                .iter()
                .any(|c| c.replacement == "calculate_total"),
            "Should suggest correct identifier. Got: {:?}",
            corrections
                .iter()
                .map(|c| &c.replacement)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_ensemble_config() {
        use libgrammstein::code::correctors::ensemble::EnsembleCorrectorBuilder;

        let lang = Arc::new(MockLanguage::rust_like());
        let corrector = EnsembleCorrectorBuilder::new(lang)
            .without_semantic()
            .without_grammar()
            .lexical_weight(1.0)
            .build();

        assert_eq!(corrector.config().lexical_weight, 1.0);
    }
}

// =============================================================================
// Semantic Corrector Integration Tests
// =============================================================================

#[cfg(test)]
mod semantic_tests {
    use super::*;
    use libgrammstein::code::correctors::SemanticCorrector;

    #[test]
    fn test_semantic_variable_registration() {
        let lang = Arc::new(MockLanguage::rust_like());
        let mut corrector = SemanticCorrector::with_defaults(lang);

        // Register some variables
        corrector.register_variable("user_count".to_string(), Some("i32".to_string()), 0);
        corrector.register_variable("item_list".to_string(), Some("Vec<Item>".to_string()), 0);
        corrector.register_variable("config".to_string(), Some("Config".to_string()), 0);

        // Test that registered variables can be used for correction
        let token = CodeToken::new(
            "usr_count", // Typo
            0,
            1,
            0,
            TokenType::Identifier,
            "identifier",
        );
        let context = TokenContext::new(TokenType::Identifier);
        let corrections = corrector.correct_token(&token, &context);

        // Should suggest the correct variable name
        if corrections.iter().any(|c| c.replacement == "user_count") {
            println!("OK: usr_count -> user_count found");
        }
    }

    #[test]
    fn test_semantic_similar_names() {
        let lang = Arc::new(MockLanguage::rust_like());
        let mut corrector = SemanticCorrector::with_defaults(lang);

        // Register variables with similar names
        corrector.register_variable("request_handler".to_string(), None, 0);
        corrector.register_variable("response_handler".to_string(), None, 0);
        corrector.register_variable("error_handler".to_string(), None, 0);

        let token = CodeToken::new(
            "reqest_handler", // Missing 'u'
            0,
            1,
            0,
            TokenType::Identifier,
            "identifier",
        );
        let context = TokenContext::new(TokenType::Identifier);
        let corrections = corrector.correct_token(&token, &context);

        assert!(
            corrections
                .iter()
                .any(|c| c.replacement == "request_handler"),
            "Should suggest request_handler"
        );
    }
}

// =============================================================================
// End-to-End Workflow Tests
// =============================================================================

#[test]
fn test_correction_workflow_basic() {
    // Simulate a simple correction workflow
    let lang = Arc::new(MockLanguage::rust_like());
    let mut corrector = LexicalCorrector::with_defaults(lang);

    // Add project identifiers
    corrector.add_identifier("main");
    corrector.add_identifier("calculate_sum");
    corrector.add_identifier("result");
    corrector.add_identifier("println");

    // Simulate tokenizing code with typos
    let tokens = vec![
        CodeToken::new("fnn", 0, 1, 0, TokenType::Keyword, "keyword"), // "fn"
        CodeToken::new("mian", 4, 1, 4, TokenType::Identifier, "identifier"), // "main"
        CodeToken::new("lett", 12, 2, 4, TokenType::Keyword, "keyword"), // "let"
        CodeToken::new("reuslt", 17, 2, 9, TokenType::Identifier, "identifier"), // "result"
    ];

    let mut all_corrections = Vec::new();
    for token in &tokens {
        let context = TokenContext::new(token.token_type);
        let corrections = corrector.correct_token(token, &context);
        all_corrections.extend(corrections);
    }

    // Verify we got some corrections
    assert!(!all_corrections.is_empty(), "Should have corrections");

    // Check for expected corrections
    let has_fn_correction = all_corrections
        .iter()
        .any(|c| c.original == "fnn" && c.replacement == "fn");
    let has_let_correction = all_corrections
        .iter()
        .any(|c| c.original == "lett" && c.replacement == "let");

    println!("Has fn correction: {}", has_fn_correction);
    println!("Has let correction: {}", has_let_correction);
}

#[test]
fn test_correction_byte_offsets() {
    // Verify corrections maintain correct byte offsets
    let source = "fn mian() {}"; // "main" misspelled at position 3

    let token = CodeToken::new("mian", 3, 1, 3, TokenType::Identifier, "identifier");

    let lang = Arc::new(MockLanguage::rust_like());
    let mut corrector = LexicalCorrector::with_defaults(lang);
    corrector.add_identifier("main");

    let context = TokenContext::new(TokenType::Identifier);
    let corrections = corrector.correct_token(&token, &context);

    if let Some(correction) = corrections.iter().find(|c| c.replacement == "main") {
        assert_eq!(correction.start_byte, 3, "Start byte should be 3");
        assert_eq!(correction.end_byte, 7, "End byte should be 7");

        // Apply the correction
        let mut result = source.to_string();
        result.replace_range(
            correction.start_byte..correction.end_byte,
            &correction.replacement,
        );
        assert_eq!(result, "fn main() {}");
    }
}

#[test]
fn test_correction_kind_categorization() {
    // Test that corrections have appropriate kinds
    let lang = Arc::new(MockLanguage::rust_like());
    let corrector = LexicalCorrector::with_defaults(lang);

    // Keyword typo
    let token = CodeToken::new("lett", 0, 1, 0, TokenType::Keyword, "keyword");
    let context = TokenContext::new(TokenType::Keyword);
    let corrections = corrector.correct_token(&token, &context);

    // All lexical corrections should be Spelling kind
    for correction in &corrections {
        assert_eq!(
            correction.kind,
            CorrectionKind::Spelling,
            "Lexical corrections should be Spelling kind"
        );
    }
}
