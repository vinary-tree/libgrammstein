//! Rust language support.

use crate::code::language::{CodeLanguage, CommentSyntax, TokenType};
use tree_sitter::Language;

/// Rust language implementation.
#[derive(Debug, Clone, Default)]
pub struct Rust;

impl Rust {
    /// Creates a new Rust language handler.
    pub fn new() -> Self {
        Self
    }
}

impl CodeLanguage for Rust {
    fn name(&self) -> &str {
        "rust"
    }

    fn display_name(&self) -> &str {
        "Rust"
    }

    fn tree_sitter_language(&self) -> Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn keywords(&self) -> &[&str] {
        &[
            "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
            "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
            "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super",
            "trait", "true", "type", "unsafe", "use", "where", "while", "async", "await", "try",
        ]
    }

    fn special_tokens(&self) -> &[&str] {
        &[
            "#", "!", "?", "::", "=>", "->", "..", "..=", "@", "'", "&", "*", "$", "|",
        ]
    }

    fn file_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        match node_kind {
            // Keywords
            "true" | "false" => TokenType::BooleanLiteral,
            k if self.keywords().contains(&k) => TokenType::Keyword,

            // Identifiers
            "identifier" | "field_identifier" | "type_identifier" => {
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else if self.builtin_types().contains(&token) || node_kind == "type_identifier" {
                    TokenType::TypeName
                } else {
                    TokenType::Identifier
                }
            }

            // Literals
            "string_literal" | "raw_string_literal" | "char_literal" => TokenType::StringLiteral,
            "integer_literal" | "float_literal" => TokenType::NumericLiteral,

            // Operators
            "+" | "-" | "*" | "/" | "%" | "^" | "&" | "|" | "!" | "~" | "=" | "==" | "!=" | "<"
            | ">" | "<=" | ">=" | "&&" | "||" | "<<" | ">>" | "+=" | "-=" | "*=" | "/=" | "%="
            | "^=" | "&=" | "|=" | "<<=" | ">>=" | ".." | "..=" | "->" | "=>" | "::" | "?" => {
                TokenType::Operator
            }

            // Punctuation
            "(" | ")" | "[" | "]" | "{" | "}" | ":" | "," | "." | ";" | "@" | "#" => {
                TokenType::Punctuation
            }

            // Comments
            "line_comment" | "block_comment" => TokenType::Comment,

            // Types
            "primitive_type" => TokenType::TypeName,

            // Macros (special)
            "macro_invocation" => TokenType::Special,

            _ => TokenType::Unknown,
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        if s.is_empty() {
            return false;
        }

        // Rust identifiers can start with letter or underscore
        // Raw identifiers start with r#
        let s = s.strip_prefix("r#").unwrap_or(s);

        if s.is_empty() {
            return false;
        }

        let mut chars = s.chars();
        let first = chars.next().unwrap();

        if !first.is_alphabetic() && first != '_' {
            return false;
        }

        chars.all(|c| c.is_alphanumeric() || c == '_')
    }

    fn builtin_types(&self) -> &[&str] {
        &[
            // Primitive types
            "bool",
            "char",
            "str",
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "f32",
            "f64",
            // Common std types
            "String",
            "Vec",
            "Box",
            "Rc",
            "Arc",
            "Cell",
            "RefCell",
            "Option",
            "Result",
            "Ok",
            "Err",
            "Some",
            "None",
            "HashMap",
            "HashSet",
            "BTreeMap",
            "BTreeSet",
            "Path",
            "PathBuf",
            "OsStr",
            "OsString",
            "Cow",
            "Pin",
            "PhantomData",
            // Traits
            "Copy",
            "Clone",
            "Debug",
            "Display",
            "Default",
            "Send",
            "Sync",
            "Sized",
            "Unpin",
            "Eq",
            "PartialEq",
            "Ord",
            "PartialOrd",
            "Hash",
            "Iterator",
            "IntoIterator",
            "FromIterator",
            "From",
            "Into",
            "TryFrom",
            "TryInto",
            "AsRef",
            "AsMut",
            "Deref",
            "DerefMut",
            "Drop",
            "Fn",
            "FnMut",
            "FnOnce",
        ]
    }

    fn stdlib_functions(&self) -> &[&str] {
        &[
            // Common methods
            "new",
            "default",
            "clone",
            "to_string",
            "to_owned",
            "unwrap",
            "expect",
            "unwrap_or",
            "unwrap_or_else",
            "unwrap_or_default",
            "ok",
            "err",
            "is_ok",
            "is_err",
            "is_some",
            "is_none",
            "map",
            "map_err",
            "and_then",
            "or_else",
            "iter",
            "iter_mut",
            "into_iter",
            "collect",
            "fold",
            "filter",
            "map",
            "flat_map",
            "push",
            "pop",
            "insert",
            "remove",
            "get",
            "get_mut",
            "len",
            "is_empty",
            "clear",
            "contains",
            // Macros (commonly used)
            "println",
            "print",
            "eprintln",
            "eprint",
            "format",
            "vec",
            "panic",
            "assert",
            "assert_eq",
            "assert_ne",
            "dbg",
            "todo",
            "unimplemented",
            "unreachable",
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
    fn test_rust_keywords() {
        let rust = Rust::new();
        assert!(rust.keywords().contains(&"fn"));
        assert!(rust.keywords().contains(&"let"));
        assert!(rust.keywords().contains(&"impl"));
    }

    #[test]
    fn test_rust_identifier_validation() {
        let rust = Rust::new();
        assert!(rust.is_valid_identifier("foo"));
        assert!(rust.is_valid_identifier("_bar"));
        assert!(rust.is_valid_identifier("r#type")); // Raw identifier
        assert!(!rust.is_valid_identifier("123foo"));
        assert!(!rust.is_valid_identifier(""));
    }

    #[test]
    fn test_rust_token_classification() {
        let rust = Rust::new();

        assert_eq!(rust.classify_token("fn", "fn"), TokenType::Keyword);
        assert_eq!(
            rust.classify_token("true", "true"),
            TokenType::BooleanLiteral
        );
        assert_eq!(
            rust.classify_token("foo", "identifier"),
            TokenType::Identifier
        );
        assert_eq!(
            rust.classify_token("i32", "primitive_type"),
            TokenType::TypeName
        );
    }
}
