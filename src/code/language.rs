//! Core traits and types for programming language support.
//!
//! This module defines the [`CodeLanguage`] trait that all supported programming
//! languages must implement, along with supporting types for token classification
//! and contextual information.

use std::collections::HashSet;
use tree_sitter::Language;

/// Classification of tokens in programming languages.
///
/// This enum categorizes tokens into semantic groups for type-aware correction.
/// Different token types may use different fuzzy matching strategies:
/// - Keywords: Exact dictionary with small Levenshtein distance
/// - Identifiers: Learned from project corpus with phonetic similarity
/// - Literals: Domain-specific handling based on type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// Language keywords (if, while, fn, let, etc.)
    Keyword,
    /// User-defined identifiers (variable names, function names, etc.)
    Identifier,
    /// Type names (int, String, Vec, etc.)
    TypeName,
    /// Operators (+, -, *, /, etc.)
    Operator,
    /// Punctuation (;, ,, (, ), etc.)
    Punctuation,
    /// String literals ("hello", 'c', etc.)
    StringLiteral,
    /// Numeric literals (42, 3.14, 0xFF, etc.)
    NumericLiteral,
    /// Boolean literals (true, false)
    BooleanLiteral,
    /// Comments (// comment, /* comment */)
    Comment,
    /// Whitespace (spaces, tabs, newlines)
    Whitespace,
    /// Special tokens specific to the language
    Special,
    /// Unknown or unclassified tokens
    Unknown,
}

impl TokenType {
    /// Returns whether this token type should be considered for correction.
    ///
    /// Comments and whitespace are typically not corrected as they don't
    /// affect program semantics.
    pub fn is_correctable(&self) -> bool {
        !matches!(self, TokenType::Comment | TokenType::Whitespace)
    }

    /// Returns whether this token type has a fixed vocabulary.
    ///
    /// Keywords, operators, and punctuation have fixed sets of valid values,
    /// while identifiers and literals can be arbitrary.
    pub fn has_fixed_vocabulary(&self) -> bool {
        matches!(
            self,
            TokenType::Keyword
                | TokenType::Operator
                | TokenType::Punctuation
                | TokenType::BooleanLiteral
        )
    }
}

/// Contextual information about a token's position in the AST.
///
/// This provides structural context for correction candidates,
/// enabling grammar-aware fuzzy matching.
#[derive(Debug, Clone)]
pub struct TokenContext {
    /// The token type classification
    pub token_type: TokenType,
    /// Parent node type in the AST (e.g., "function_definition", "if_statement")
    pub parent_node_type: Option<String>,
    /// Sibling node types for positional context
    pub sibling_types: Vec<String>,
    /// Depth in the AST (0 = root)
    pub depth: usize,
    /// Whether the token is inside an error node
    pub in_error_region: bool,
    /// Expected token types at this position (from grammar)
    pub expected_types: Vec<TokenType>,
}

impl TokenContext {
    /// Creates a new token context with minimal information.
    pub fn new(token_type: TokenType) -> Self {
        Self {
            token_type,
            parent_node_type: None,
            sibling_types: Vec::new(),
            depth: 0,
            in_error_region: false,
            expected_types: Vec::new(),
        }
    }

    /// Sets the parent node type.
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_node_type = Some(parent.into());
        self
    }

    /// Sets the AST depth.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Marks this context as being inside an error region.
    pub fn in_error(mut self) -> Self {
        self.in_error_region = true;
        self
    }
}

/// Trait for programming language support.
///
/// Implementations of this trait provide language-specific functionality:
/// - Tree-sitter grammar for parsing
/// - Token type classification
/// - Keyword and special token sets
/// - Semantic analysis hooks
///
/// # Example
///
/// ```ignore
/// use libgrammstein::code::{CodeLanguage, TokenType};
///
/// struct Python;
///
/// impl CodeLanguage for Python {
///     fn name(&self) -> &str { "python" }
///     fn tree_sitter_language(&self) -> tree_sitter::Language {
///         tree_sitter_python::LANGUAGE.into()
///     }
///     fn keywords(&self) -> &[&str] {
///         &["def", "class", "if", "else", "elif", "for", "while", ...]
///     }
///     // ... other methods
/// }
/// ```
pub trait CodeLanguage: Send + Sync {
    /// Returns the canonical name of the language (lowercase, e.g., "python", "rust").
    fn name(&self) -> &str;

    /// Returns the display name of the language (e.g., "Python", "Rust").
    fn display_name(&self) -> &str {
        self.name()
    }

    /// Returns the tree-sitter Language for parsing.
    fn tree_sitter_language(&self) -> Language;

    /// Returns the set of language keywords.
    ///
    /// Keywords are reserved words that have special meaning in the language.
    fn keywords(&self) -> &[&str];

    /// Returns language-specific special tokens.
    ///
    /// These are tokens with special meaning beyond standard operators,
    /// e.g., `@` for decorators in Python, `!` for macros in Rust.
    fn special_tokens(&self) -> &[&str] {
        &[]
    }

    /// Returns common file extensions for this language.
    fn file_extensions(&self) -> &[&str];

    /// Classifies a token string into a TokenType.
    ///
    /// This method uses both the token text and its AST node kind
    /// for accurate classification.
    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType;

    /// Returns whether a string is a valid identifier in this language.
    fn is_valid_identifier(&self, s: &str) -> bool;

    /// Returns built-in type names for this language.
    fn builtin_types(&self) -> &[&str] {
        &[]
    }

    /// Returns standard library functions/methods.
    fn stdlib_functions(&self) -> &[&str] {
        &[]
    }

    /// Returns the comment syntax for the language.
    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax::default()
    }

    /// Returns whether the language is whitespace-significant.
    ///
    /// Python and other indentation-based languages return true.
    fn is_whitespace_significant(&self) -> bool {
        false
    }

    /// Returns all keywords as a HashSet for efficient lookup.
    fn keyword_set(&self) -> HashSet<&str> {
        self.keywords().iter().copied().collect()
    }
}

/// Comment syntax configuration for a language.
#[derive(Debug, Clone)]
pub struct CommentSyntax {
    /// Single-line comment prefix (e.g., "//" or "#")
    pub line_comment: Option<&'static str>,
    /// Block comment delimiters (start, end)
    pub block_comment: Option<(&'static str, &'static str)>,
    /// Documentation comment prefix (e.g., "///" or "##")
    pub doc_comment: Option<&'static str>,
}

impl Default for CommentSyntax {
    fn default() -> Self {
        Self {
            line_comment: Some("//"),
            block_comment: Some(("/*", "*/")),
            doc_comment: Some("///"),
        }
    }
}

impl CommentSyntax {
    /// Creates comment syntax for C-style languages (C, C++, Java, JavaScript, Rust).
    pub fn c_style() -> Self {
        Self::default()
    }

    /// Creates comment syntax for Python-style languages.
    pub fn python_style() -> Self {
        Self {
            line_comment: Some("#"),
            block_comment: Some(("\"\"\"", "\"\"\"")),
            doc_comment: Some("#"),
        }
    }

    /// Creates comment syntax for shell-style languages.
    pub fn shell_style() -> Self {
        Self {
            line_comment: Some("#"),
            block_comment: None,
            doc_comment: None,
        }
    }

    /// Creates comment syntax for Lisp-style languages (including MeTTa).
    pub fn lisp_style() -> Self {
        Self {
            line_comment: Some(";"),
            block_comment: Some(("#|", "|#")),
            doc_comment: Some(";;"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_type_is_correctable() {
        assert!(TokenType::Keyword.is_correctable());
        assert!(TokenType::Identifier.is_correctable());
        assert!(!TokenType::Comment.is_correctable());
        assert!(!TokenType::Whitespace.is_correctable());
    }

    #[test]
    fn test_token_type_has_fixed_vocabulary() {
        assert!(TokenType::Keyword.has_fixed_vocabulary());
        assert!(TokenType::Operator.has_fixed_vocabulary());
        assert!(!TokenType::Identifier.has_fixed_vocabulary());
        assert!(!TokenType::StringLiteral.has_fixed_vocabulary());
    }

    #[test]
    fn test_token_context_builder() {
        let ctx = TokenContext::new(TokenType::Identifier)
            .with_parent("function_definition")
            .with_depth(3)
            .in_error();

        assert_eq!(ctx.token_type, TokenType::Identifier);
        assert_eq!(
            ctx.parent_node_type,
            Some("function_definition".to_string())
        );
        assert_eq!(ctx.depth, 3);
        assert!(ctx.in_error_region);
    }
}
