//! JavaScript language support.

use crate::code::language::{CodeLanguage, CommentSyntax, TokenType};
use tree_sitter::Language;

/// JavaScript language implementation.
#[derive(Debug, Clone, Default)]
pub struct JavaScript;

impl JavaScript {
    /// Creates a new JavaScript language handler.
    pub fn new() -> Self {
        Self
    }
}

impl CodeLanguage for JavaScript {
    fn name(&self) -> &str {
        "javascript"
    }

    fn display_name(&self) -> &str {
        "JavaScript"
    }

    fn tree_sitter_language(&self) -> Language {
        tree_sitter_javascript::LANGUAGE.into()
    }

    fn keywords(&self) -> &[&str] {
        &[
            "async",
            "await",
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "else",
            "export",
            "extends",
            "false",
            "finally",
            "for",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "let",
            "new",
            "null",
            "return",
            "static",
            "super",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "typeof",
            "undefined",
            "var",
            "void",
            "while",
            "with",
            "yield",
        ]
    }

    fn special_tokens(&self) -> &[&str] {
        &["=>", "...", "?.", "??", "?."]
    }

    fn file_extensions(&self) -> &[&str] {
        &["js", "jsx", "mjs", "cjs"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        match node_kind {
            // Keywords
            "true" | "false" => TokenType::BooleanLiteral,
            "null" | "undefined" => TokenType::Keyword,
            k if self.keywords().contains(&k) => TokenType::Keyword,

            // Identifiers
            "identifier" | "property_identifier" | "shorthand_property_identifier" => {
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.builtin_types().contains(&token) {
                    TokenType::TypeName
                } else {
                    TokenType::Identifier
                }
            }

            // Literals
            "string" | "template_string" | "template_literal_type" => TokenType::StringLiteral,
            "number" => TokenType::NumericLiteral,
            "regex" => TokenType::StringLiteral,

            // Operators
            "+" | "-" | "*" | "/" | "%" | "**" | "=" | "==" | "===" | "!=" | "!==" | "<" | ">"
            | "<=" | ">=" | "&&" | "||" | "!" | "??" | "?." | "&" | "|" | "^" | "~" | "<<"
            | ">>" | ">>>" | "+=" | "-=" | "*=" | "/=" | "%=" | "**=" | "&=" | "|=" | "^="
            | "<<=" | ">>=" | ">>>=" | "=>" | "..." | "++" | "--" | "?" | ":" => {
                TokenType::Operator
            }

            // Punctuation
            "(" | ")" | "[" | "]" | "{" | "}" | "," | "." | ";" => TokenType::Punctuation,

            // Comments
            "comment" | "hash_bang_line" => TokenType::Comment,

            // JSX
            "jsx_opening_element" | "jsx_closing_element" => TokenType::Special,

            _ => TokenType::Unknown,
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        if s.is_empty() {
            return false;
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap();

        // First character must be letter, underscore, or $
        if !first.is_alphabetic() && first != '_' && first != '$' {
            return false;
        }

        // Rest must be alphanumeric, underscore, or $
        chars.all(|c| c.is_alphanumeric() || c == '_' || c == '$')
    }

    fn builtin_types(&self) -> &[&str] {
        &[
            // Primitive wrappers
            "Boolean",
            "Number",
            "String",
            "Symbol",
            "BigInt",
            // Objects
            "Object",
            "Array",
            "Function",
            "Date",
            "RegExp",
            "Error",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "Promise",
            "Proxy",
            "Reflect",
            // TypedArrays
            "ArrayBuffer",
            "SharedArrayBuffer",
            "DataView",
            "Int8Array",
            "Uint8Array",
            "Uint8ClampedArray",
            "Int16Array",
            "Uint16Array",
            "Int32Array",
            "Uint32Array",
            "Float32Array",
            "Float64Array",
            "BigInt64Array",
            "BigUint64Array",
            // Error types
            "TypeError",
            "RangeError",
            "ReferenceError",
            "SyntaxError",
            "EvalError",
            "URIError",
            "AggregateError",
            // Other
            "JSON",
            "Math",
            "Intl",
            "console",
        ]
    }

    fn stdlib_functions(&self) -> &[&str] {
        &[
            // Global functions
            "parseInt",
            "parseFloat",
            "isNaN",
            "isFinite",
            "encodeURI",
            "decodeURI",
            "encodeURIComponent",
            "decodeURIComponent",
            "eval",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            "fetch",
            "alert",
            "confirm",
            "prompt",
            // Array methods
            "push",
            "pop",
            "shift",
            "unshift",
            "slice",
            "splice",
            "map",
            "filter",
            "reduce",
            "forEach",
            "find",
            "findIndex",
            "some",
            "every",
            "includes",
            "indexOf",
            "join",
            "concat",
            "sort",
            "reverse",
            "flat",
            "flatMap",
            // Object methods
            "keys",
            "values",
            "entries",
            "assign",
            "freeze",
            "seal",
            "hasOwnProperty",
            "toString",
            "valueOf",
            // String methods
            "charAt",
            "charCodeAt",
            "split",
            "substring",
            "substr",
            "toLowerCase",
            "toUpperCase",
            "trim",
            "replace",
            "match",
            // Promise methods
            "then",
            "catch",
            "finally",
            "resolve",
            "reject",
            "all",
            "race",
            // Console methods
            "log",
            "warn",
            "error",
            "info",
            "debug",
            "trace",
        ]
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax::c_style()
    }

    fn is_whitespace_significant(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_javascript_keywords() {
        let js = JavaScript::new();
        assert!(js.keywords().contains(&"function"));
        assert!(js.keywords().contains(&"const"));
        assert!(js.keywords().contains(&"let"));
    }

    #[test]
    fn test_javascript_identifier_validation() {
        let js = JavaScript::new();
        assert!(js.is_valid_identifier("foo"));
        assert!(js.is_valid_identifier("_bar"));
        assert!(js.is_valid_identifier("$element")); // jQuery-style
        assert!(!js.is_valid_identifier("123foo"));
        assert!(!js.is_valid_identifier(""));
    }

    #[test]
    fn test_javascript_token_classification() {
        let js = JavaScript::new();

        assert_eq!(
            js.classify_token("function", "function"),
            TokenType::Keyword
        );
        assert_eq!(js.classify_token("true", "true"), TokenType::BooleanLiteral);
        assert_eq!(
            js.classify_token("foo", "identifier"),
            TokenType::Identifier
        );
        assert_eq!(js.classify_token("42", "number"), TokenType::NumericLiteral);
    }
}
