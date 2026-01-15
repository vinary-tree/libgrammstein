//! Rholang language support.
//!
//! Rholang is a reflective, concurrent programming language based on the
//! rho-calculus, designed for building scalable, secure, and composable
//! blockchain applications on the RChain platform.

use crate::code::language::{CodeLanguage, CommentSyntax, TokenType};
use tree_sitter::Language;

/// Rholang language implementation.
///
/// Rholang is a process algebra language where the primary abstractions are:
/// - **Channels** (names): Communication endpoints prefixed with `@`
/// - **Processes**: Concurrent computations composed with `|`
/// - **Contracts**: Persistent receive operations
/// - **Bundles**: Access control for channels
#[derive(Debug, Clone, Default)]
pub struct Rholang;

impl Rholang {
    /// Creates a new Rholang language handler.
    pub fn new() -> Self {
        Self
    }
}

impl CodeLanguage for Rholang {
    fn name(&self) -> &str {
        "rholang"
    }

    fn display_name(&self) -> &str {
        "Rholang"
    }

    fn tree_sitter_language(&self) -> Language {
        rholang_tree_sitter::LANGUAGE.into()
    }

    fn keywords(&self) -> &[&str] {
        &[
            // Control flow and declarations
            "new", "in", "if", "else", "let", "match", "select", "contract", "for",
            // Logical operators (keyword form)
            "or", "and", "matches", "not",
            // Bundle types
            "bundle", "bundle-", "bundle+", "bundle0",
            // Literals
            "true", "false", "Nil",
        ]
    }

    fn special_tokens(&self) -> &[&str] {
        &[
            // Channel operations
            "@",   // Quote (process -> name)
            "*",   // Eval/dereference (name -> process)
            // Send operations
            "!",   // Send single
            "!!",  // Send persistent
            "!?",  // Synchronous send-then-receive
            // Receive operations
            "<-",  // Linear receive
            "<=",  // Persistent receive
            "<<-", // Peek (non-consuming receive)
            "?!",  // Receive-then-send
            // Process algebra
            "|",   // Parallel composition
            "&",   // Concurrent binding
            ";",   // Sequential composition
            "=>",  // Pattern match arm
            // Set operations
            "++",  // Union/concatenation
            "--",  // Difference
            "/\\", // Conjunction
            "\\/", // Disjunction
            "~",   // Negation
            "%%",  // Interpolation
            // Variable reference kinds
            "=",   // Simple binding
            "=*",  // Binding with dereference
            // Remainder patterns
            "...", // Spread/rest
        ]
    }

    fn file_extensions(&self) -> &[&str] {
        &["rho"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        match node_kind {
            // Keywords
            "new" | "in" | "if" | "else" | "let" | "match" | "select" | "contract" | "for" => {
                TokenType::Keyword
            }
            "or" | "and" | "matches" | "not" => TokenType::Keyword,
            "bundle" | "bundle_read" | "bundle_write" | "bundle_equiv" | "bundle_read_write" => {
                TokenType::Keyword
            }
            "nil" | "Nil" => TokenType::Keyword,

            // Literals
            "true" | "false" | "bool_literal" => TokenType::BooleanLiteral,
            "long_literal" => TokenType::NumericLiteral,
            "string_literal" => TokenType::StringLiteral,
            "uri_literal" => TokenType::StringLiteral,

            // Types
            "simple_type" | "Bool" | "Int" | "String" | "Uri" | "ByteArray" => TokenType::TypeName,

            // Variables and names
            "var" => {
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else {
                    TokenType::Identifier
                }
            }
            "wildcard" => TokenType::Special,
            "name" | "quote" => TokenType::Identifier,

            // Channel operations
            "send" | "send_single" | "send_multiple" | "send_sync" => TokenType::Operator,
            "input" | "linear_bind" | "repeated_bind" | "peek_bind" => TokenType::Operator,

            // Operators
            "add" | "sub" | "mult" | "div" | "mod" | "neg" => TokenType::Operator,
            "eq" | "neq" | "lt" | "lte" | "gt" | "gte" => TokenType::Operator,
            "concat" | "diff" | "interpolation" => TokenType::Operator,
            "conjunction" | "disjunction" | "negation" => TokenType::Operator,
            "method" => TokenType::Operator,

            // Punctuation
            "(" | ")" | "[" | "]" | "{" | "}" | "{|" | "|}" => TokenType::Punctuation,
            "," | ":" | "." | ";" => TokenType::Punctuation,

            // Comments
            "line_comment" | "block_comment" => TokenType::Comment,

            // Process constructs
            "par" | "choice" | "branch" | "case" => TokenType::Special,
            "contract" | "new" | "ifElse" | "let" | "match" | "bundle" => TokenType::Special,

            // Collections
            "list" | "set" | "map" | "tuple" | "pathmap" | "collection" => TokenType::TypeName,

            // Operators by symbol
            "@" | "*" | "!" | "!!" | "!?" | "?!" | "|" | "&" | "=>" | "..." => TokenType::Operator,
            "<-" | "<=" | "<<-" | "++" | "--" | "/\\" | "\\/" | "~" | "%%" => TokenType::Operator,
            "+" | "-" | "/" | "%" | "==" | "!=" | "<" | ">" => TokenType::Operator,

            _ => {
                // Check if it's a known keyword
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.special_tokens().contains(&token) {
                    TokenType::Operator
                } else {
                    TokenType::Unknown
                }
            }
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        if s.is_empty() {
            return false;
        }

        // Rholang identifier pattern: [a-zA-Z]([a-zA-Z0-9_'])*|_([a-zA-Z0-9_'])+
        let mut chars = s.chars();
        let first = match chars.next() {
            Some(c) => c,
            None => return false,
        };

        if first == '_' {
            // Must have at least one more character after underscore
            let rest: String = chars.collect();
            if rest.is_empty() {
                return false; // "_" alone is a wildcard, not an identifier
            }
            rest.chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
        } else if first.is_alphabetic() {
            chars.all(|c| c.is_alphanumeric() || c == '_' || c == '\'')
        } else {
            false
        }
    }

    fn builtin_types(&self) -> &[&str] {
        &["Bool", "Int", "String", "Uri", "ByteArray", "Nil"]
    }

    fn stdlib_functions(&self) -> &[&str] {
        // Rholang doesn't have traditional stdlib functions,
        // but has built-in system contracts and methods
        &[
            // Common methods available on collections
            "nth",
            "length",
            "slice",
            "toByteArray",
            "hexToBytes",
            "toUtf8Bytes",
            "union",
            "diff",
            "add",
            "delete",
            "contains",
            "get",
            "getOrElse",
            "set",
            "keys",
            "size",
        ]
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax {
            line_comment: Some("//"),
            block_comment: Some(("/*", "*/")),
            doc_comment: Some("///"),
        }
    }

    fn is_whitespace_significant(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rholang_keywords() {
        let rholang = Rholang::new();
        assert!(rholang.keywords().contains(&"new"));
        assert!(rholang.keywords().contains(&"contract"));
        assert!(rholang.keywords().contains(&"for"));
        assert!(rholang.keywords().contains(&"match"));
    }

    #[test]
    fn test_rholang_identifier_validation() {
        let rholang = Rholang::new();

        // Valid identifiers
        assert!(rholang.is_valid_identifier("foo"));
        assert!(rholang.is_valid_identifier("bar123"));
        assert!(rholang.is_valid_identifier("_foo"));
        assert!(rholang.is_valid_identifier("foo'"));
        assert!(rholang.is_valid_identifier("x'y"));

        // Invalid identifiers
        assert!(!rholang.is_valid_identifier("")); // Empty
        assert!(!rholang.is_valid_identifier("_")); // Wildcard only
        assert!(!rholang.is_valid_identifier("123foo")); // Starts with digit
        assert!(!rholang.is_valid_identifier("@foo")); // Starts with @
    }

    #[test]
    fn test_rholang_special_tokens() {
        let rholang = Rholang::new();
        assert!(rholang.special_tokens().contains(&"@"));
        assert!(rholang.special_tokens().contains(&"!"));
        assert!(rholang.special_tokens().contains(&"|"));
        assert!(rholang.special_tokens().contains(&"<-"));
    }

    #[test]
    fn test_rholang_token_classification() {
        let rholang = Rholang::new();

        assert_eq!(
            rholang.classify_token("new", "new"),
            TokenType::Keyword
        );
        assert_eq!(
            rholang.classify_token("true", "bool_literal"),
            TokenType::BooleanLiteral
        );
        assert_eq!(
            rholang.classify_token("42", "long_literal"),
            TokenType::NumericLiteral
        );
        assert_eq!(
            rholang.classify_token("myVar", "var"),
            TokenType::Identifier
        );
        assert_eq!(
            rholang.classify_token("Int", "simple_type"),
            TokenType::TypeName
        );
    }

    #[test]
    fn test_rholang_file_extensions() {
        let rholang = Rholang::new();
        assert!(rholang.file_extensions().contains(&"rho"));
    }
}
