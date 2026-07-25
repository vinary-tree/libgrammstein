//! MeTTa language support.
//!
//! MeTTa (Meta Type Talk) is a functional meta-programming language designed
//! for knowledge representation, reasoning, and AI systems. It features
//! hypergraph-based data structures and powerful pattern matching capabilities.

use crate::code::language::{CodeLanguage, CommentSyntax, TokenType};
use tree_sitter::Language;

/// MeTTa language implementation.
///
/// MeTTa is a homoiconic language where:
/// - **Atoms**: Basic units (symbols, variables, expressions)
/// - **Expressions**: S-expression lists `(expr expr ...)`
/// - **Variables**: Pattern variables prefixed with `$`
/// - **Spaces**: Atomspace references prefixed with `&`
/// - **Types**: Gradual typing with `:` annotations
#[derive(Debug, Clone, Default)]
pub struct MeTTa;

impl MeTTa {
    /// Creates a new MeTTa language handler.
    pub fn new() -> Self {
        Self
    }
}

impl CodeLanguage for MeTTa {
    fn name(&self) -> &str {
        "metta"
    }

    fn display_name(&self) -> &str {
        "MeTTa"
    }

    fn tree_sitter_language(&self) -> Language {
        // The grammar is published as a version-agnostic `LanguageFn`; `.into()`
        // materializes it as the `tree_sitter::Language` of the linked
        // `tree-sitter` release.
        tree_sitter_metta::LANGUAGE.into()
    }

    fn keywords(&self) -> &[&str] {
        &[
            // Boolean literals
            "True",
            "False",
            // Core operations (commonly used as keywords in MeTTa programs)
            "match",
            "let",
            "let*",
            "if",
            "case",
            "function",
            "return",
            "empty",
            "Error",
            // Type system
            "Type",
            "Atom",
            "Symbol",
            "Variable",
            "Expression",
            "Grounded",
            "Unit",
            "Number",
            "String",
            "Bool",
            // Atomspace operations
            "new-space",
            "add-atom",
            "remove-atom",
            "get-atoms",
            "import!",
            "include",
            "bind!",
            "pragma!",
            // Control flow
            "sequential",
            "chain",
            "eval",
            "quote",
            "unquote",
        ]
    }

    fn special_tokens(&self) -> &[&str] {
        &[
            // Prefix operators
            "!", // Reduction/evaluation
            "?", // Query
            "'", // Quote
            // Variable prefix
            "$", // Pattern variable marker
            // Space reference prefix
            "&", // Atomspace reference (e.g., &self)
            // Type annotation
            ":", // Type annotation
            // Assignment/binding
            "=",  // Definition/equality
            ":=", // Rule definition
            // Arrows
            "->",  // Function type / transformation
            "<-",  // Reverse arrow
            "<<-", // Pattern binding
            // Comparison
            "==", // Equality test
            "!=", // Inequality test
            "<=", // Less than or equal
            ">=", // Greater than or equal
            "<",  // Less than
            ">",  // Greater than
            // Punctuation
            "|",   // Alternative/separator
            ",",   // Tuple/sequence separator
            "@",   // Apply/at
            "...", // Spread/variadic
            ".",   // Dot operator
            // Special
            "_", // Wildcard pattern
        ]
    }

    fn file_extensions(&self) -> &[&str] {
        &["metta", "mt"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        match node_kind {
            // Literals
            "boolean_literal" => TokenType::BooleanLiteral,
            "integer_literal" => TokenType::NumericLiteral,
            "float_literal" => TokenType::NumericLiteral,
            "string_literal" => TokenType::StringLiteral,

            // Variables and identifiers
            "variable" => TokenType::Identifier,
            "identifier" => {
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.builtin_types().contains(&token) {
                    TokenType::TypeName
                } else {
                    TokenType::Identifier
                }
            }
            "wildcard" => TokenType::Special,
            "space_reference" => TokenType::Special,
            "special_type_symbol" => TokenType::TypeName,

            // Operators
            "arrow_operator" => TokenType::Operator,
            "comparison_operator" => TokenType::Operator,
            "assignment_operator" => TokenType::Operator,
            "type_annotation_operator" => TokenType::Operator,
            "rule_definition_operator" => TokenType::Operator,
            "punctuation_operator" => TokenType::Punctuation,
            "arithmetic_operator" => TokenType::Operator,
            "logic_operator" => TokenType::Operator,
            "operator" => TokenType::Operator,

            // Prefix operators
            "exclaim_prefix" => TokenType::Operator,
            "question_prefix" => TokenType::Operator,
            "quote_prefix" => TokenType::Operator,

            // Structural elements
            "list" | "expression" => TokenType::Punctuation,
            "prefixed_expression" => TokenType::Special,
            "atom_expression" => TokenType::Unknown,

            // Punctuation
            "(" | ")" => TokenType::Punctuation,

            // Comments
            "line_comment" => TokenType::Comment,

            // Boolean literals by value
            "True" | "False" => TokenType::BooleanLiteral,

            _ => {
                // Check if it matches a keyword
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.special_tokens().contains(&token) {
                    if matches!(token, "(" | ")" | "|" | "," | ".") {
                        TokenType::Punctuation
                    } else {
                        TokenType::Operator
                    }
                } else if token.starts_with('$') {
                    TokenType::Identifier // Variable
                } else if token.starts_with('&') {
                    TokenType::Special // Space reference
                } else if token.starts_with('%') && token.ends_with('%') {
                    TokenType::TypeName // Special type symbol
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

        // MeTTa identifier pattern (from grammar):
        // Regular identifiers: [^\s()\[\]{}"$;!?'_&][^\s()\[\]{}"$;]*
        // Variables: $[^\s()";#]*
        // Space refs: &[^\s()";#]+

        // Check for variable ($...) or space reference (&...)
        if s.starts_with('$') || s.starts_with('&') {
            let rest = &s[1..];
            // Must have content after prefix for space refs
            if s.starts_with('&') && rest.is_empty() {
                return false;
            }
            // Check rest doesn't contain delimiters
            return !rest
                .chars()
                .any(|c| c.is_whitespace() || matches!(c, '(' | ')' | '"' | ';' | '#'));
        }

        // Wildcard
        if s == "_" {
            return true;
        }

        let mut chars = s.chars();
        let first = match chars.next() {
            Some(c) => c,
            None => return false,
        };

        // First character cannot be delimiter or special prefix
        if first.is_whitespace()
            || matches!(
                first,
                '(' | ')' | '[' | ']' | '{' | '}' | '"' | '$' | ';' | '!' | '?' | '\'' | '_' | '&'
            )
        {
            return false;
        }

        // Rest cannot contain these delimiters
        chars.all(|c| {
            !c.is_whitespace() && !matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '"' | '$' | ';')
        })
    }

    fn builtin_types(&self) -> &[&str] {
        &[
            // Core types
            "Type",
            "Atom",
            "Symbol",
            "Variable",
            "Expression",
            "Grounded",
            // Primitive types
            "Number",
            "String",
            "Bool",
            "Unit",
            // Collection types
            "List",
            "Tuple",
            // Function types
            "Function",
            "->",
            // Special types
            "%Undefined%",
            "%Irreducible%",
        ]
    }

    fn stdlib_functions(&self) -> &[&str] {
        &[
            // Core operations
            "match",
            "let",
            "let*",
            "if",
            "case",
            "function",
            "return",
            "eval",
            "quote",
            "unquote",
            // Arithmetic
            "+",
            "-",
            "*",
            "/",
            "%",
            // Comparison
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            // Boolean
            "and",
            "or",
            "not",
            // Atomspace operations
            "new-space",
            "add-atom",
            "remove-atom",
            "get-atoms",
            "bind!",
            "import!",
            "include",
            "pragma!",
            // List operations
            "cons-atom",
            "decons-atom",
            "car-atom",
            "cdr-atom",
            // Type operations
            "get-type",
            "get-metatype",
            // Utility
            "empty",
            "Error",
            "nop",
            "trace!",
            "println!",
            // Pattern matching
            "collapse",
            "superpose",
        ]
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax {
            line_comment: Some(";"),
            block_comment: None, // MeTTa only has line comments
            doc_comment: Some(";;"),
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
    fn test_metta_keywords() {
        let metta = MeTTa::new();
        assert!(metta.keywords().contains(&"True"));
        assert!(metta.keywords().contains(&"False"));
        assert!(metta.keywords().contains(&"match"));
        assert!(metta.keywords().contains(&"let"));
        assert!(metta.keywords().contains(&"Type"));
    }

    #[test]
    fn test_metta_identifier_validation() {
        let metta = MeTTa::new();

        // Valid identifiers
        assert!(metta.is_valid_identifier("foo"));
        assert!(metta.is_valid_identifier("my-function"));
        assert!(metta.is_valid_identifier("+"));
        assert!(metta.is_valid_identifier("->"));
        assert!(metta.is_valid_identifier("Type"));
        assert!(metta.is_valid_identifier("_")); // Wildcard

        // Variables
        assert!(metta.is_valid_identifier("$x"));
        assert!(metta.is_valid_identifier("$var"));
        assert!(metta.is_valid_identifier("$")); // Empty variable allowed

        // Space references
        assert!(metta.is_valid_identifier("&self"));
        assert!(metta.is_valid_identifier("&kb"));
        assert!(!metta.is_valid_identifier("&")); // Empty space ref not allowed

        // Invalid identifiers
        assert!(!metta.is_valid_identifier("")); // Empty
        assert!(!metta.is_valid_identifier("(foo")); // Starts with delimiter
        assert!(!metta.is_valid_identifier("foo(bar")); // Contains delimiter
    }

    #[test]
    fn test_metta_special_tokens() {
        let metta = MeTTa::new();
        assert!(metta.special_tokens().contains(&"!"));
        assert!(metta.special_tokens().contains(&"?"));
        assert!(metta.special_tokens().contains(&"$"));
        assert!(metta.special_tokens().contains(&"&"));
        assert!(metta.special_tokens().contains(&"->"));
        assert!(metta.special_tokens().contains(&":"));
    }

    #[test]
    fn test_metta_token_classification() {
        let metta = MeTTa::new();

        assert_eq!(
            metta.classify_token("True", "boolean_literal"),
            TokenType::BooleanLiteral
        );
        assert_eq!(
            metta.classify_token("42", "integer_literal"),
            TokenType::NumericLiteral
        );
        assert_eq!(
            metta.classify_token("3.14", "float_literal"),
            TokenType::NumericLiteral
        );
        assert_eq!(
            metta.classify_token("$x", "variable"),
            TokenType::Identifier
        );
        assert_eq!(
            metta.classify_token("match", "identifier"),
            TokenType::Keyword
        );
        assert_eq!(
            metta.classify_token("foo", "identifier"),
            TokenType::Identifier
        );
        assert_eq!(
            metta.classify_token("&self", "space_reference"),
            TokenType::Special
        );
    }

    #[test]
    fn test_metta_file_extensions() {
        let metta = MeTTa::new();
        assert!(metta.file_extensions().contains(&"metta"));
        assert!(metta.file_extensions().contains(&"mt"));
    }

    #[test]
    fn test_metta_comment_syntax() {
        let metta = MeTTa::new();
        let comment = metta.comment_syntax();
        assert_eq!(comment.line_comment, Some(";"));
        assert!(comment.block_comment.is_none());
    }
}
