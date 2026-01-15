# Tokenizer Module

The tokenizer module provides code-aware tokenization that preserves semantic information from tree-sitter parsing, enabling type-aware correction.

## Overview

Unlike simple lexical tokenizers, the code tokenizer:

- **Preserves AST context**: Parent nodes, siblings, depth
- **Classifies tokens semantically**: Keywords, identifiers, types, literals
- **Tracks error regions**: Marks tokens inside parse errors
- **Filters selectively**: Include/exclude whitespace and comments

## Key Types

| Type | Description |
|------|-------------|
| `CodeToken` | Token with text, position, type, and context |
| `CodeTokenizer<L>` | Configurable tokenizer for a language |
| `TokenIterator<L>` | Iterator over tokens in source code |

## CodeToken

The `CodeToken` struct contains comprehensive token information:

```rust
pub struct CodeToken {
    /// The token text
    pub text: String,
    /// Byte offset in the source
    pub byte_offset: usize,
    /// Line number (0-indexed)
    pub line: usize,
    /// Column number (0-indexed)
    pub column: usize,
    /// Token type classification
    pub token_type: TokenType,
    /// Tree-sitter node kind
    pub node_kind: String,
    /// Contextual information
    pub context: TokenContext,
}
```

### Methods

```rust
impl CodeToken {
    /// Creates a new code token.
    pub fn new(
        text: impl Into<String>,
        byte_offset: usize,
        line: usize,
        column: usize,
        token_type: TokenType,
        node_kind: impl Into<String>,
    ) -> Self;

    /// Returns whether this token is inside an error region.
    pub fn is_in_error(&self) -> bool;

    /// Returns whether this token should be considered for correction.
    pub fn is_correctable(&self) -> bool;
}
```

### Example: Creating Tokens

```rust
use libgrammstein::code::{CodeToken, TokenType};

let token = CodeToken::new(
    "calculate_total",
    0,        // byte offset
    1,        // line (0-indexed)
    5,        // column (0-indexed)
    TokenType::Identifier,
    "identifier"
);

assert_eq!(token.text, "calculate_total");
assert_eq!(token.token_type, TokenType::Identifier);
assert!(token.is_correctable());  // Identifiers are correctable
assert!(!token.is_in_error());    // Not in error region
```

## TokenType

Tokens are classified into semantic categories:

```rust
pub enum TokenType {
    Keyword,         // if, while, fn, let
    Identifier,      // variable names, function names
    TypeName,        // int, String, Vec
    Operator,        // +, -, *, /
    Punctuation,     // ;, ,, (, )
    StringLiteral,   // "hello"
    NumericLiteral,  // 42, 3.14
    BooleanLiteral,  // true, false
    Comment,         // // comment
    Whitespace,      // spaces, tabs
    Special,         // language-specific
    Unknown,         // unclassified
}
```

### Correctability

Not all token types are correctable:

| Token Type | Correctable | Reason |
|------------|-------------|--------|
| `Keyword` | Yes | Can be misspelled (`retrun` → `return`) |
| `Identifier` | Yes | Can be misspelled or wrong |
| `TypeName` | Yes | Can be misspelled |
| `Operator` | Yes | Can be transposed (`=+` → `+=`) |
| `Punctuation` | Yes | Can be missing or extra |
| `StringLiteral` | Yes | Content can be spell-checked |
| `NumericLiteral` | No | Format validation only |
| `BooleanLiteral` | Yes | Limited vocabulary |
| `Comment` | No | Natural language, not code |
| `Whitespace` | No | Formatting only |

## TokenContext

The `TokenContext` struct provides structural information:

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

### Context Example

```rust
// For token "result" in: def calculate(x): result = x + 1
let context = token.context;

assert_eq!(context.parent_node_type, Some("assignment".into()));
assert!(context.sibling_types.contains(&"=".into()));
assert!(context.sibling_types.contains(&"binary_operator".into()));
assert_eq!(context.depth, 3);  // module → function → block → assignment
assert!(!context.in_error_region);
```

## CodeTokenizer

The `CodeTokenizer<L>` extracts tokens from parsed code:

```rust
pub struct CodeTokenizer<'a, L: CodeLanguage> {
    language: &'a L,
    include_whitespace: bool,
    include_comments: bool,
}
```

### Builder Pattern

```rust
use libgrammstein::code::{CodeTokenizer, Python};

let python = Python::new();

// Default: no whitespace, no comments
let tokenizer = CodeTokenizer::new(&python);

// Include whitespace (for indentation-sensitive languages)
let tokenizer = CodeTokenizer::new(&python)
    .with_whitespace(true);

// Include comments (for documentation analysis)
let tokenizer = CodeTokenizer::new(&python)
    .with_comments(true);

// Include both
let tokenizer = CodeTokenizer::new(&python)
    .with_whitespace(true)
    .with_comments(true);
```

### Tokenization Methods

```rust
impl<'a, L: CodeLanguage> CodeTokenizer<'a, L> {
    /// Extracts all tokens from a parsed tree.
    pub fn tokenize(&self, tree: &Tree, source: &str) -> Vec<CodeToken>;

    /// Extracts tokens only from error regions.
    pub fn tokenize_errors(&self, tree: &Tree, source: &str) -> Vec<CodeToken>;
}
```

### Example: Basic Tokenization

```rust
use libgrammstein::code::{CodeParser, CodeTokenizer, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python.clone())?;

let source = r#"
def greet(name):
    print(f"Hello, {name}!")
"#;

let parsed = parser.parse(source)?;
let tokenizer = CodeTokenizer::new(python.as_ref());
let tokens = tokenizer.tokenize(&parsed.tree, source);

for token in &tokens {
    println!(
        "{:15} {:15} line {} col {}",
        token.text,
        format!("{:?}", token.token_type),
        token.line + 1,
        token.column
    );
}
```

Output:
```
def             Keyword         line 2 col 0
greet           Identifier      line 2 col 4
(               Punctuation     line 2 col 9
name            Identifier      line 2 col 10
)               Punctuation     line 2 col 14
:               Punctuation     line 2 col 15
print           Identifier      line 3 col 4
...
```

### Example: Error-Focused Tokenization

```rust
let source = "def foo(\n    retrun 42";  // Missing ) and misspelled return
let parsed = parser.parse(source)?;

let tokenizer = CodeTokenizer::new(python.as_ref());
let error_tokens = tokenizer.tokenize_errors(&parsed.tree, source);

println!("Tokens in error regions:");
for token in error_tokens {
    println!(
        "  '{}' ({:?}) at line {}",
        token.text,
        token.token_type,
        token.line + 1
    );
}
```

## Token Filtering

The tokenizer automatically filters based on configuration:

### Default (No Whitespace/Comments)

```rust
let tokenizer = CodeTokenizer::new(&python);
let tokens = tokenizer.tokenize(&tree, source);

// Tokens: def, foo, (, x, ), :, return, x, +, 1
// No whitespace or comments
```

### With Whitespace (Python)

For Python, whitespace is significant for indentation:

```rust
let tokenizer = CodeTokenizer::new(&python)
    .with_whitespace(true);

let tokens = tokenizer.tokenize(&tree, source);
// Now includes indentation tokens
```

### With Comments

For documentation extraction or comment spell-checking:

```rust
let tokenizer = CodeTokenizer::new(&python)
    .with_comments(true);

let tokens = tokenizer.tokenize(&tree, source);
// Now includes comment tokens
```

## Using TokenContext

The token context enables grammar-aware correction:

### Example: Context-Aware Correction

```rust
for token in tokens {
    if token.is_in_error() {
        println!("Error token: '{}'", token.text);

        // Use context for better correction
        if let Some(parent) = &token.context.parent_node_type {
            match parent.as_str() {
                "function_definition" => {
                    // Likely a keyword or parameter
                    suggest_from(&["def", "async", "return"]);
                }
                "assignment" => {
                    // Likely an identifier
                    suggest_from_corpus();
                }
                _ => {}
            }
        }
    }
}
```

### Example: Identifying Declaration Contexts

```rust
fn is_function_parameter(token: &CodeToken) -> bool {
    token.context.parent_node_type.as_deref() == Some("parameters")
}

fn is_type_annotation(token: &CodeToken) -> bool {
    token.context.parent_node_type.as_deref() == Some("type")
}

fn is_import_statement(token: &CodeToken) -> bool {
    token.context.parent_node_type.as_deref() == Some("import_statement")
        || token.context.parent_node_type.as_deref() == Some("import_from_statement")
}
```

## TokenIterator

For streaming tokenization:

```rust
pub struct TokenIterator<'a, L: CodeLanguage> {
    tokenizer: CodeTokenizer<'a, L>,
    tree: Tree,
    source: String,
    tokens: Vec<CodeToken>,
    position: usize,
}

impl<L: CodeLanguage> Iterator for TokenIterator<'_, L> {
    type Item = CodeToken;
    fn next(&mut self) -> Option<Self::Item>;
}
```

### Example: Iterator Usage

```rust
let iterator = TokenIterator::new(tokenizer, tree, source);

// Find all identifiers
let identifiers: Vec<_> = iterator
    .filter(|t| t.token_type == TokenType::Identifier)
    .collect();

// Find correctable tokens in errors
let correctable_errors: Vec<_> = iterator
    .filter(|t| t.is_in_error() && t.is_correctable())
    .collect();
```

## Integration with Correctors

The tokenizer output feeds directly into correctors:

```rust
use libgrammstein::code::{
    CodeParser, CodeTokenizer, LexicalCorrector, Python
};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python.clone())?;
let tokenizer = CodeTokenizer::new(python.as_ref());

// Parse code with errors
let source = "def calcluate(x):\n    retrun x";
let parsed = parser.parse(source)?;

// Extract tokens from error regions
let error_tokens = tokenizer.tokenize_errors(&parsed.tree, source);

// Create corrector
let mut corrector = LexicalCorrector::with_defaults(python);

// Correct each error token
for token in error_tokens {
    let corrections = corrector.correct_token(&token, &token.context);
    for correction in corrections.top(3) {
        println!("{} -> {} ({:.2})",
            token.text,
            correction.replacement,
            correction.confidence
        );
    }
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Full tokenization | O(n) | n = leaf nodes in AST |
| Error tokenization | O(e) | e = nodes in error regions |
| Token creation | O(1) | Constant per token |
| Context extraction | O(s) | s = number of siblings |

### Optimization Tips

1. **Use `tokenize_errors()`** for correction - don't tokenize entire file
2. **Disable whitespace/comments** unless needed
3. **Cache tokenizer** - create once per language
4. **Batch processing** - tokenize multiple error regions together

## Thread Safety

`CodeTokenizer` is `Send` but not `Sync` (holds language reference):

```rust
use std::thread;

let python = Python::new();

// Move tokenizer to thread
thread::spawn(move || {
    let tokenizer = CodeTokenizer::new(&python);
    // Use tokenizer in this thread
});
```

For parallel tokenization, create separate tokenizers per thread:

```rust
use rayon::prelude::*;

let sources: Vec<&str> = vec![...];

let results: Vec<_> = sources.par_iter()
    .map(|source| {
        let python = Python::new();
        let tokenizer = CodeTokenizer::new(&python);
        let mut parser = CodeParser::new(Arc::new(python.clone())).unwrap();
        let parsed = parser.parse(source).unwrap();
        tokenizer.tokenize(&parsed.tree, source)
    })
    .collect();
```

## See Also

- [Language Framework](language.md) - `TokenType` and `TokenContext`
- [AST](ast.md) - Tree-sitter parsing
- [Correction](correction.md) - Using tokens for correction
- [Correctors](correctors/overview.md) - Token-level correction
