# AST Module

The AST module provides tree-sitter integration for incremental parsing with error recovery, enabling correction of partially valid code.

## Overview

This module wraps tree-sitter's parsing capabilities to provide:

- **Incremental parsing**: Re-parse only changed portions of code
- **Error recovery**: Continue parsing past syntax errors
- **Error location**: Precise byte and line/column positions
- **AST traversal**: Depth-first iteration over nodes

## Key Types

| Type | Description |
|------|-------------|
| `ParsedCode` | Parse result with tree, source, and error information |
| `AstNode` | Simplified representation of a tree-sitter node |
| `CodeParser<L>` | Parser configured for a specific language |
| `ErrorRange` | Location and content of a syntax error |
| `EditInfo` | Description of a source code edit |
| `AstError` | Error types for parsing operations |

## ParsedCode

The `ParsedCode` struct contains the parse tree and metadata:

```rust
pub struct ParsedCode {
    /// The tree-sitter parse tree
    pub tree: Tree,
    /// The original source code
    pub source: String,
    /// Language used for parsing
    pub language_name: String,
    /// Whether the parse tree contains any errors
    pub has_errors: bool,
    /// Error nodes in the tree
    pub error_ranges: Vec<ErrorRange>,
}
```

### Methods

```rust
impl ParsedCode {
    /// Returns the root node of the AST.
    pub fn root(&self) -> Node;

    /// Returns an iterator over all error ranges.
    pub fn errors(&self) -> impl Iterator<Item = &ErrorRange>;

    /// Returns the number of syntax errors.
    pub fn error_count(&self) -> usize;

    /// Checks if a byte offset is within an error region.
    pub fn is_in_error(&self, byte_offset: usize) -> bool;
}
```

### Example: Checking for Errors

```rust
use libgrammstein::code::{CodeParser, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python)?;

// Parse code with a syntax error
let source = "def foo(\n    return 42";
let parsed = parser.parse(source)?;

if parsed.has_errors {
    println!("Found {} errors:", parsed.error_count());
    for error in parsed.errors() {
        println!(
            "  Line {}, Column {}: {:?} - '{}'",
            error.start_position.0 + 1,
            error.start_position.1,
            error.kind,
            error.text
        );
    }
}
```

## ErrorRange

The `ErrorRange` struct provides detailed error location information:

```rust
pub struct ErrorRange {
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Start position (line, column) - 0-indexed
    pub start_position: (usize, usize),
    /// End position (line, column) - 0-indexed
    pub end_position: (usize, usize),
    /// The erroneous text
    pub text: String,
    /// The node kind (usually "ERROR" or "MISSING")
    pub kind: String,
}
```

### Error Kinds

| Kind | Description |
|------|-------------|
| `ERROR` | Unexpected token or malformed syntax |
| `MISSING` | Expected token not present (e.g., missing `)`) |

## AstNode

The `AstNode` struct provides a simplified, owned representation of tree-sitter nodes:

```rust
pub struct AstNode {
    /// The node kind (e.g., "function_definition", "identifier")
    pub kind: String,
    /// Start byte offset
    pub start_byte: usize,
    /// End byte offset
    pub end_byte: usize,
    /// Start position (line, column)
    pub start_position: (usize, usize),
    /// End position (line, column)
    pub end_position: (usize, usize),
    /// Whether this is a named node
    pub is_named: bool,
    /// Whether this node is an error node
    pub is_error: bool,
    /// Whether this node is missing (expected but not present)
    pub is_missing: bool,
    /// Child nodes
    pub children: Vec<AstNode>,
    /// The text content (for leaf nodes)
    pub text: Option<String>,
}
```

### Creating from Tree-sitter

```rust
use libgrammstein::code::AstNode;

// Convert a tree-sitter node to AstNode
let ast_node = AstNode::from_ts_node(parsed.root(), &source);
```

### Traversal Methods

```rust
impl AstNode {
    /// Returns an iterator over all descendant nodes (depth-first).
    pub fn descendants(&self) -> impl Iterator<Item = &AstNode>;

    /// Finds nodes by kind. Returns an iterator to avoid allocating a Vec.
    pub fn find_by_kind<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a AstNode>;

    /// Finds all error nodes. Returns an iterator to avoid allocating a Vec.
    pub fn find_errors(&self) -> impl Iterator<Item = &AstNode>;
}
```

### Example: Finding All Functions

```rust
let ast = AstNode::from_ts_node(parsed.root(), &source);

// Find all function definitions
let functions = ast.find_by_kind("function_definition");
for func in functions {
    println!("Function at line {}", func.start_position.0 + 1);

    // Find the function name
    if let Some(name_node) = func.children.iter()
        .find(|c| c.kind == "identifier")
    {
        if let Some(name) = &name_node.text {
            println!("  Name: {}", name);
        }
    }
}
```

### Example: Finding Errors

```rust
let ast = AstNode::from_ts_node(parsed.root(), &source);

// Find all error and missing nodes
let errors = ast.find_errors();
for error in errors {
    if error.is_missing {
        println!("Missing token at {:?}", error.start_position);
    } else {
        println!("Error: '{}' at {:?}",
            error.text.as_deref().unwrap_or(""),
            error.start_position
        );
    }
}
```

## CodeParser

The `CodeParser<L>` struct provides parsing with caching support:

```rust
pub struct CodeParser<L: CodeLanguage> {
    language: Arc<L>,
    parser: Parser,
    tree_cache: HashMap<u64, Tree>,
}
```

### Creating a Parser

```rust
use libgrammstein::code::{CodeParser, Python, Rust, JavaScript};
use std::sync::Arc;

// Create parsers for different languages
let python_parser = CodeParser::new(Arc::new(Python::new()))?;
let rust_parser = CodeParser::new(Arc::new(Rust::new()))?;
let js_parser = CodeParser::new(Arc::new(JavaScript::new()))?;
```

### Basic Parsing

```rust
let mut parser = CodeParser::new(Arc::new(Python::new()))?;

let source = r#"
def greet(name):
    print(f"Hello, {name}!")

greet("World")
"#;

let parsed = parser.parse(source)?;

println!("Language: {}", parsed.language_name);
println!("Has errors: {}", parsed.has_errors);
println!("Root node kind: {}", parsed.root().kind());
```

### Incremental Parsing

For editor integration, incremental parsing re-parses only changed portions:

```rust
use libgrammstein::code::EditInfo;

let mut parser = CodeParser::new(Arc::new(Python::new()))?;

// Initial parse
let source = "def foo():\n    pass";
let mut parsed = parser.parse(source)?;
let mut tree = parsed.tree;

// User types " + 1" after "pass"
let edit = EditInfo::insertion(
    19,      // byte position
    1,       // row
    8,       // column
    " + 1"   // inserted text
);

let new_source = "def foo():\n    pass + 1";
let new_parsed = parser.parse_incremental(new_source, &mut tree, &edit)?;
```

## EditInfo

The `EditInfo` struct describes source code modifications:

```rust
pub struct EditInfo {
    /// Start byte of the edit
    pub start_byte: usize,
    /// Old end byte (before edit)
    pub old_end_byte: usize,
    /// New end byte (after edit)
    pub new_end_byte: usize,
    /// Start position (row, column)
    pub start_position: (usize, usize),
    /// Old end position
    pub old_end_position: (usize, usize),
    /// New end position
    pub new_end_position: (usize, usize),
}
```

### Factory Methods

```rust
impl EditInfo {
    /// Creates an EditInfo for an insertion at a position.
    pub fn insertion(
        position: usize,
        row: usize,
        column: usize,
        inserted_text: &str
    ) -> Self;

    /// Creates an EditInfo for a deletion.
    pub fn deletion(
        start_byte: usize,
        end_byte: usize,
        start_pos: (usize, usize),
        end_pos: (usize, usize),
    ) -> Self;
}
```

### Example: Handling Edits

```rust
// User inserts "x" at position (0, 5)
let insert_edit = EditInfo::insertion(5, 0, 5, "x");

// User deletes characters from (0, 10) to (0, 15)
let delete_edit = EditInfo::deletion(10, 15, (0, 10), (0, 15));

// User replaces text (delete then insert)
let replace_edit = EditInfo {
    start_byte: 10,
    old_end_byte: 15,
    new_end_byte: 13,
    start_position: (0, 10),
    old_end_position: (0, 15),
    new_end_position: (0, 13),
};
```

## AstError

Error types for AST operations:

```rust
pub enum AstError {
    /// Parser initialization failed
    ParserInit(String),
    /// Parsing failed completely (no tree produced)
    ParseFailed,
    /// Language mismatch
    LanguageMismatch { expected: String, got: String },
}
```

### Error Handling

```rust
use libgrammstein::code::{CodeParser, Python, AstError};
use std::sync::Arc;

fn parse_python(source: &str) -> Result<(), AstError> {
    let mut parser = CodeParser::new(Arc::new(Python::new()))?;
    let parsed = parser.parse(source)?;

    if parsed.has_errors {
        // Note: parsing still succeeds with errors due to error recovery
        println!("Parsed with {} errors", parsed.error_count());
    }

    Ok(())
}

match parse_python("def broken(") {
    Ok(()) => println!("Parsed successfully"),
    Err(AstError::ParserInit(msg)) => {
        eprintln!("Failed to initialize parser: {}", msg);
    }
    Err(AstError::ParseFailed) => {
        eprintln!("Parsing failed completely");
    }
    Err(AstError::LanguageMismatch { expected, got }) => {
        eprintln!("Wrong language: expected {}, got {}", expected, got);
    }
}
```

## Tree-sitter Node Kinds

Common tree-sitter node kinds by language:

### Python

| Kind | Description |
|------|-------------|
| `module` | Root node |
| `function_definition` | `def` function |
| `class_definition` | `class` definition |
| `if_statement` | `if` block |
| `for_statement` | `for` loop |
| `identifier` | Variable/function name |
| `string` | String literal |
| `integer` | Integer literal |
| `ERROR` | Parse error |

### Rust

| Kind | Description |
|------|-------------|
| `source_file` | Root node |
| `function_item` | `fn` function |
| `impl_item` | `impl` block |
| `struct_item` | `struct` definition |
| `identifier` | Name |
| `type_identifier` | Type name |
| `string_literal` | String |
| `integer_literal` | Number |

### JavaScript

| Kind | Description |
|------|-------------|
| `program` | Root node |
| `function_declaration` | `function` declaration |
| `arrow_function` | Arrow function |
| `class_declaration` | `class` |
| `identifier` | Name |
| `string` | String literal |
| `number` | Number literal |

## Performance Considerations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Full parse | O(n) | Linear in source length |
| Incremental parse | O(k) | k = size of changed region |
| Error collection | O(e) | e = number of errors |
| AST traversal | O(n) | n = number of nodes |

### Best Practices

1. **Use incremental parsing** for editor integration
2. **Reuse parsers** across parses (they cache state)
3. **Check `has_errors`** before expensive operations
4. **Use `is_in_error()`** to scope correction efforts

## Thread Safety

`CodeParser<L>` is not `Sync` due to tree-sitter's internal state, but `ParsedCode` can be shared:

```rust
use std::sync::Arc;
use std::thread;

// Parse in main thread
let mut parser = CodeParser::new(Arc::new(Python::new()))?;
let parsed = Arc::new(parser.parse(source)?);

// Share parsed result across threads
let parsed1 = Arc::clone(&parsed);
let parsed2 = Arc::clone(&parsed);

let handles = vec![
    thread::spawn(move || parsed1.error_count()),
    thread::spawn(move || parsed2.root().kind().to_string()),
];
```

## See Also

- [Language Framework](language.md) - `CodeLanguage` trait
- [Tokenizer](tokenizer.md) - Token extraction from AST
- [CPG](cpg.md) - Code Property Graphs built from AST
- [Pipeline](pipeline.md) - End-to-end correction using AST
