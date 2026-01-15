# Language Framework

The language module defines the core `CodeLanguage` trait and supporting types for programming language support in libgrammstein.

## What is the Language Framework?

The language framework provides:

- **`CodeLanguage` trait**: Interface that all language implementations must satisfy
- **`TokenType` enum**: Classification system for tokens
- **`TokenContext` struct**: Structural context for grammar-aware correction
- **`CommentSyntax`**: Language-specific comment configuration

This enables the code module to work with any programming language that has a tree-sitter grammar.

## TokenType

The `TokenType` enum classifies tokens into semantic groups for type-aware correction:

```rust
pub enum TokenType {
    Keyword,         // if, while, fn, let, etc.
    Identifier,      // variable names, function names
    TypeName,        // int, String, Vec, etc.
    Operator,        // +, -, *, /, etc.
    Punctuation,     // ;, ,, (, ), {, }, etc.
    StringLiteral,   // "hello", 'c'
    NumericLiteral,  // 42, 3.14, 0xFF
    BooleanLiteral,  // true, false
    Comment,         // // comment, /* comment */
    Whitespace,      // spaces, tabs, newlines
    Special,         // language-specific tokens
    Unknown,         // unclassified tokens
}
```

### Methods

```rust
impl TokenType {
    /// Returns true if this token type should be considered for correction.
    /// Comments and whitespace are not correctable.
    pub fn is_correctable(&self) -> bool;

    /// Returns true if this token type has a fixed vocabulary.
    /// Keywords, operators, punctuation, and booleans have fixed sets.
    pub fn has_fixed_vocabulary(&self) -> bool;
}
```

### Correction Strategies by Token Type

| Token Type | Correction Strategy |
|------------|---------------------|
| `Keyword` | Exact dictionary, Levenshtein distance 1-2 |
| `Identifier` | Project corpus + phonetic similarity |
| `TypeName` | Built-in types + imported types |
| `Operator` | Fixed set, transposition detection |
| `Punctuation` | Insertion/deletion suggestions |
| `StringLiteral` | Domain-specific (spell check content) |
| `NumericLiteral` | Format validation only |
| `BooleanLiteral` | Exact match (`true`/`false`) |

## TokenContext

The `TokenContext` struct provides structural information for grammar-aware correction:

```rust
pub struct TokenContext {
    /// The token type classification
    pub token_type: TokenType,
    /// Parent node type in the AST (e.g., "function_definition")
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
```

### Builder Pattern

```rust
let context = TokenContext::new(TokenType::Identifier)
    .with_parent("function_definition")
    .with_depth(3)
    .in_error();

assert_eq!(context.parent_node_type, Some("function_definition".into()));
assert!(context.in_error_region);
```

## CodeLanguage Trait

The `CodeLanguage` trait defines the interface for programming language support:

```rust
pub trait CodeLanguage: Send + Sync {
    /// Canonical name (lowercase): "python", "rust"
    fn name(&self) -> &str;

    /// Display name: "Python", "Rust"
    fn display_name(&self) -> &str { self.name() }

    /// Tree-sitter Language for parsing
    fn tree_sitter_language(&self) -> tree_sitter::Language;

    /// Reserved keywords: ["if", "while", "fn", "let", ...]
    fn keywords(&self) -> &[&str];

    /// Language-specific special tokens: ["@", "!", "#"]
    fn special_tokens(&self) -> &[&str] { &[] }

    /// File extensions: [".py", ".pyi"]
    fn file_extensions(&self) -> &[&str];

    /// Classify a token by text and AST node kind
    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType;

    /// Check if a string is a valid identifier
    fn is_valid_identifier(&self, s: &str) -> bool;

    /// Built-in type names: ["int", "str", "bool", ...]
    fn builtin_types(&self) -> &[&str] { &[] }

    /// Standard library functions: ["print", "len", "range", ...]
    fn stdlib_functions(&self) -> &[&str] { &[] }

    /// Comment syntax configuration
    fn comment_syntax(&self) -> CommentSyntax { CommentSyntax::default() }

    /// Whether whitespace is significant (Python: true)
    fn is_whitespace_significant(&self) -> bool { false }

    /// Keywords as HashSet for O(1) lookup
    fn keyword_set(&self) -> HashSet<&str> {
        self.keywords().iter().copied().collect()
    }
}
```

## CommentSyntax

Configure comment styles for different languages:

```rust
pub struct CommentSyntax {
    /// Single-line comment prefix: "//" or "#"
    pub line_comment: Option<&'static str>,
    /// Block comment delimiters: ("/*", "*/")
    pub block_comment: Option<(&'static str, &'static str)>,
    /// Documentation comment prefix: "///" or "##"
    pub doc_comment: Option<&'static str>,
}
```

### Preset Styles

```rust
// C-style: //, /* */, ///
let c_style = CommentSyntax::c_style();

// Python-style: #, """ """, #
let python_style = CommentSyntax::python_style();

// Shell-style: # only
let shell_style = CommentSyntax::shell_style();

// Lisp-style: ;, #| |#, ;;
let lisp_style = CommentSyntax::lisp_style();
```

## Implementing a Custom Language

To add support for a new programming language:

```rust
use libgrammstein::code::{CodeLanguage, TokenType, CommentSyntax};

#[derive(Debug, Clone)]
pub struct MyLanguage;

impl CodeLanguage for MyLanguage {
    fn name(&self) -> &str {
        "mylang"
    }

    fn display_name(&self) -> &str {
        "MyLanguage"
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        // Return your tree-sitter grammar
        tree_sitter_mylang::language()
    }

    fn keywords(&self) -> &[&str] {
        &["if", "else", "while", "for", "fn", "let", "return"]
    }

    fn special_tokens(&self) -> &[&str] {
        &["@", "!"]  // Language-specific operators
    }

    fn file_extensions(&self) -> &[&str] {
        &["ml", "myl"]
    }

    fn classify_token(&self, token: &str, node_kind: &str) -> TokenType {
        // Use node_kind from tree-sitter for accurate classification
        match node_kind {
            "keyword" => TokenType::Keyword,
            "identifier" => TokenType::Identifier,
            "type_identifier" => TokenType::TypeName,
            "integer" | "float" => TokenType::NumericLiteral,
            "string" => TokenType::StringLiteral,
            "comment" => TokenType::Comment,
            _ => {
                // Fallback to text-based classification
                if self.keywords().contains(&token) {
                    TokenType::Keyword
                } else {
                    TokenType::Unknown
                }
            }
        }
    }

    fn is_valid_identifier(&self, s: &str) -> bool {
        let mut chars = s.chars();
        match chars.next() {
            Some(c) if c.is_alphabetic() || c == '_' => {
                chars.all(|c| c.is_alphanumeric() || c == '_')
            }
            _ => false,
        }
    }

    fn builtin_types(&self) -> &[&str] {
        &["int", "float", "string", "bool", "void"]
    }

    fn stdlib_functions(&self) -> &[&str] {
        &["print", "input", "len", "range"]
    }

    fn comment_syntax(&self) -> CommentSyntax {
        CommentSyntax::c_style()
    }

    fn is_whitespace_significant(&self) -> bool {
        false  // Set to true for Python-like languages
    }
}
```

## Thread Safety

All `CodeLanguage` implementations must be `Send + Sync`:

```rust
use std::sync::Arc;

// Languages can be shared across threads
let language: Arc<dyn CodeLanguage> = Arc::new(Python::new());

// Use in multiple threads
let lang1 = Arc::clone(&language);
let lang2 = Arc::clone(&language);
```

## Best Practices

1. **Node Kind Classification**: Use tree-sitter node kinds in `classify_token` for accuracy
2. **Keyword Completeness**: Include all reserved words in `keywords()`
3. **Extension Coverage**: Include all valid file extensions
4. **Identifier Validation**: Handle Unicode identifiers if supported
5. **Comment Syntax**: Configure accurately for syntax highlighting compatibility

## See Also

- [Languages](languages.md) - Built-in language implementations
- [AST](ast.md) - Tree-sitter integration
- [Tokenizer](tokenizer.md) - Token extraction with context
- [Correction](correction.md) - How token types affect correction
