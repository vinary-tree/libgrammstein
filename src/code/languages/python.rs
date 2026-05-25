//! Python language support.

use crate::code::language::{CodeLanguage, CommentSyntax, TokenType};
use tree_sitter::Language;

/// Python language implementation.
#[derive(Debug, Clone, Default)]
pub struct Python;

impl Python {
    /// Creates a new Python language handler.
    pub fn new() -> Self {
        Self
    }
}

impl CodeLanguage for Python {
    fn name(&self) -> &str {
        "python"
    }

    fn display_name(&self) -> &str {
        "Python"
    }

    fn tree_sitter_language(&self) -> Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn keywords(&self) -> &[&str] {
        &[
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ]
    }

    fn special_tokens(&self) -> &[&str] {
        &["@", "->", ":", "**", "//", "...", "_"]
    }

    fn file_extensions(&self) -> &[&str] {
        &["py", "pyw", "pyi"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        match node_kind {
            // Keywords
            "True" | "False" => TokenType::BooleanLiteral,
            "None" => TokenType::Keyword,
            k if self.keywords().contains(&k) => TokenType::Keyword,

            // Identifiers
            "identifier" => {
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.builtin_types().contains(&token) {
                    TokenType::TypeName
                } else {
                    TokenType::Identifier
                }
            }

            // Literals
            "string" | "string_content" | "string_start" | "string_end" => TokenType::StringLiteral,
            "integer" | "float" | "imaginary" => TokenType::NumericLiteral,

            // Operators
            "+" | "-" | "*" | "/" | "//" | "%" | "**" | "==" | "!=" | "<" | ">" | "<=" | ">="
            | "is" | "in" | "and" | "or" | "not" | "&" | "|" | "^" | "~" | "<<" | ">>" | "="
            | "+=" | "-=" | "*=" | "/=" | "//=" | "%=" | "**=" | "&=" | "|=" | "^=" | "<<="
            | ">>=" | "->" => TokenType::Operator,

            // Punctuation (includes @ for decorators)
            "(" | ")" | "[" | "]" | "{" | "}" | ":" | "," | "." | ";" | "@" => {
                TokenType::Punctuation
            }

            // Comments
            "comment" => TokenType::Comment,

            // Type hints
            "type" => TokenType::TypeName,

            _ => TokenType::Unknown,
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        if s.is_empty() {
            return false;
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap();

        // First character must be letter or underscore
        if !first.is_alphabetic() && first != '_' {
            return false;
        }

        // Rest must be alphanumeric or underscore
        chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    fn builtin_types(&self) -> &[&str] {
        &[
            "int",
            "float",
            "complex",
            "str",
            "bytes",
            "bytearray",
            "list",
            "tuple",
            "set",
            "frozenset",
            "dict",
            "bool",
            "object",
            "type",
            "range",
            "slice",
            "memoryview",
            "property",
            "classmethod",
            "staticmethod",
            "Exception",
            "BaseException",
            "TypeError",
            "ValueError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "NameError",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Any",
            "Type",
            "Generic",
            "Protocol",
        ]
    }

    fn stdlib_functions(&self) -> &[&str] {
        &[
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "breakpoint",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
        ]
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax::python_style()
    }

    fn is_whitespace_significant(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_keywords() {
        let python = Python::new();
        assert!(python.keywords().contains(&"def"));
        assert!(python.keywords().contains(&"class"));
        assert!(python.keywords().contains(&"import"));
    }

    #[test]
    fn test_python_identifier_validation() {
        let python = Python::new();
        assert!(python.is_valid_identifier("foo"));
        assert!(python.is_valid_identifier("_bar"));
        assert!(python.is_valid_identifier("baz123"));
        assert!(!python.is_valid_identifier("123foo"));
        assert!(!python.is_valid_identifier(""));
    }

    #[test]
    fn test_python_token_classification() {
        let python = Python::new();

        assert_eq!(python.classify_token("def", "def"), TokenType::Keyword);
        assert_eq!(
            python.classify_token("True", "True"),
            TokenType::BooleanLiteral
        );
        assert_eq!(
            python.classify_token("foo", "identifier"),
            TokenType::Identifier
        );
        assert_eq!(
            python.classify_token("42", "integer"),
            TokenType::NumericLiteral
        );
    }

    #[test]
    fn test_python_whitespace_significant() {
        let python = Python::new();
        assert!(python.is_whitespace_significant());
    }
}
