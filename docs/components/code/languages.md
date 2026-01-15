# Language Implementations

This module provides built-in implementations of the `CodeLanguage` trait for various programming languages.

## Overview

libgrammstein supports both mainstream programming languages and domain-specific languages (DSLs). Each implementation provides:

- Tree-sitter grammar integration
- Keyword and operator definitions
- Token classification logic
- Built-in types and standard library functions
- Comment syntax configuration

## Supported Languages

| Language | Feature Flag | Category | Tree-sitter Grammar |
|----------|--------------|----------|---------------------|
| Python | `code-python` | Mainstream | `tree-sitter-python` |
| Rust | `code-rust` | Mainstream | `tree-sitter-rust` |
| JavaScript | `code-javascript` | Mainstream | `tree-sitter-javascript` |
| Rholang | `code-rholang` | DSL (Blockchain) | `rholang-tree-sitter` |
| MeTTa | `code-metta` | DSL (AI/Reasoning) | `tree-sitter-metta` |

## Feature Flags

Enable language support in `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["code", "code-python", "code-rust"] }
```

Convenience feature groups:

| Feature | Includes |
|---------|----------|
| `code-mainstream` | Python, Rust, JavaScript |
| `code-dsl` | Rholang, MeTTa |
| `code-full` | All languages + neural features |

---

## Python

Python support with type hints and indentation awareness.

### Usage

```rust
use libgrammstein::code::Python;

let python = Python::new();

// Parse Python code
let source = r#"
def greet(name: str) -> str:
    return f"Hello, {name}!"
"#;

// Access language information
assert_eq!(python.name(), "python");
assert!(python.is_whitespace_significant());
```

### Characteristics

| Property | Value |
|----------|-------|
| File Extensions | `.py`, `.pyw`, `.pyi` |
| Whitespace Significant | Yes |
| Comment Syntax | `#` (line), `"""` (block) |
| Unicode Identifiers | Yes |

### Keywords

```
False   None    True    and     as      assert  async   await
break   class   continue  def   del     elif    else    except
finally for     from    global  if      import  in      is
lambda  nonlocal not     or      pass    raise   return  try
while   with    yield
```

### Built-in Types

```rust
// Primitive types
["int", "float", "complex", "str", "bytes", "bytearray",
 "list", "tuple", "set", "frozenset", "dict", "bool", "object", "type"]

// Type hints (typing module)
["Optional", "Union", "List", "Dict", "Set", "Tuple",
 "Callable", "Any", "Type", "Generic", "Protocol"]

// Exception types
["Exception", "BaseException", "TypeError", "ValueError",
 "KeyError", "IndexError", "AttributeError", "NameError"]
```

### Token Classification

```rust
let python = Python::new();

// Keywords use their text as node_kind
assert_eq!(python.classify_token("def", "def"), TokenType::Keyword);
assert_eq!(python.classify_token("None", "None"), TokenType::Keyword);

// Boolean literals
assert_eq!(python.classify_token("True", "True"), TokenType::BooleanLiteral);

// Identifiers and types
assert_eq!(python.classify_token("foo", "identifier"), TokenType::Identifier);
assert_eq!(python.classify_token("int", "identifier"), TokenType::TypeName);

// Literals
assert_eq!(python.classify_token("42", "integer"), TokenType::NumericLiteral);
assert_eq!(python.classify_token("3.14", "float"), TokenType::NumericLiteral);
```

### Identifier Validation

```rust
let python = Python::new();

// Valid Python identifiers
assert!(python.is_valid_identifier("foo"));
assert!(python.is_valid_identifier("_private"));
assert!(python.is_valid_identifier("CamelCase"));
assert!(python.is_valid_identifier("snake_case_123"));

// Invalid identifiers
assert!(!python.is_valid_identifier("123abc"));  // Starts with digit
assert!(!python.is_valid_identifier(""));        // Empty
assert!(!python.is_valid_identifier("my-var"));  // Contains hyphen
```

---

## Rust

Rust support with macro awareness and raw identifier handling.

### Usage

```rust
use libgrammstein::code::Rust;

let rust = Rust::new();

// Access language information
assert_eq!(rust.name(), "rust");
assert!(!rust.is_whitespace_significant());
```

### Characteristics

| Property | Value |
|----------|-------|
| File Extensions | `.rs` |
| Whitespace Significant | No |
| Comment Syntax | `//` (line), `/* */` (block), `///` (doc) |
| Raw Identifiers | `r#keyword` syntax |

### Keywords

```
as      async   await   break   const   continue  crate   dyn
else    enum    extern  false   fn      for       if      impl
in      let     loop    match   mod     move      mut     pub
ref     return  self    Self    static  struct    super   trait
true    type    unsafe  use     where   while     try
```

### Built-in Types

```rust
// Primitive types
["bool", "char", "str",
 "i8", "i16", "i32", "i64", "i128", "isize",
 "u8", "u16", "u32", "u64", "u128", "usize",
 "f32", "f64"]

// Standard library types
["String", "Vec", "Box", "Rc", "Arc", "Cell", "RefCell",
 "Option", "Result", "Ok", "Err", "Some", "None",
 "HashMap", "HashSet", "BTreeMap", "BTreeSet",
 "Path", "PathBuf", "Cow", "Pin", "PhantomData"]

// Common traits
["Copy", "Clone", "Debug", "Display", "Default",
 "Send", "Sync", "Sized", "Eq", "PartialEq", "Ord", "PartialOrd",
 "Iterator", "IntoIterator", "From", "Into", "Drop", "Fn", "FnMut", "FnOnce"]
```

### Token Classification

```rust
let rust = Rust::new();

// Keywords
assert_eq!(rust.classify_token("fn", "fn"), TokenType::Keyword);
assert_eq!(rust.classify_token("let", "let"), TokenType::Keyword);

// Boolean literals
assert_eq!(rust.classify_token("true", "true"), TokenType::BooleanLiteral);

// Primitive types
assert_eq!(rust.classify_token("i32", "primitive_type"), TokenType::TypeName);

// Macro invocations are special
assert_eq!(rust.classify_token("println!", "macro_invocation"), TokenType::Special);
```

### Raw Identifier Support

```rust
let rust = Rust::new();

// Raw identifiers allow keywords as names
assert!(rust.is_valid_identifier("r#type"));   // Valid: raw identifier
assert!(rust.is_valid_identifier("r#match"));  // Valid: raw identifier
assert!(rust.is_valid_identifier("r#loop"));   // Valid: raw identifier

// Regular identifiers
assert!(rust.is_valid_identifier("foo_bar"));
assert!(rust.is_valid_identifier("_hidden"));
```

---

## JavaScript

JavaScript ES6+ support with JSX awareness.

### Usage

```rust
use libgrammstein::code::JavaScript;

let js = JavaScript::new();

// Access language information
assert_eq!(js.name(), "javascript");
assert_eq!(js.display_name(), "JavaScript");
```

### Characteristics

| Property | Value |
|----------|-------|
| File Extensions | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Whitespace Significant | No |
| Comment Syntax | `//` (line), `/* */` (block), `///` (doc) |
| Dollar Identifiers | `$variable` syntax |

### Keywords

```
async     await     break     case      catch     class     const
continue  debugger  default   delete    do        else      export
extends   false     finally   for       function  if        import
in        instanceof  let     new       null      return    static
super     switch    this      throw     true      try       typeof
undefined var       void      while     with      yield
```

### Built-in Types

```rust
// Primitive wrappers
["Boolean", "Number", "String", "Symbol", "BigInt"]

// Objects
["Object", "Array", "Function", "Date", "RegExp", "Error",
 "Map", "Set", "WeakMap", "WeakSet", "Promise", "Proxy", "Reflect"]

// TypedArrays
["ArrayBuffer", "DataView", "Int8Array", "Uint8Array",
 "Int16Array", "Uint16Array", "Int32Array", "Uint32Array",
 "Float32Array", "Float64Array", "BigInt64Array", "BigUint64Array"]

// Error types
["TypeError", "RangeError", "ReferenceError", "SyntaxError"]
```

### Token Classification

```rust
let js = JavaScript::new();

// Keywords
assert_eq!(js.classify_token("function", "function"), TokenType::Keyword);
assert_eq!(js.classify_token("const", "const"), TokenType::Keyword);

// Null-like keywords
assert_eq!(js.classify_token("null", "null"), TokenType::Keyword);
assert_eq!(js.classify_token("undefined", "undefined"), TokenType::Keyword);

// Boolean literals
assert_eq!(js.classify_token("true", "true"), TokenType::BooleanLiteral);

// JSX elements are special
assert_eq!(js.classify_token("<div>", "jsx_opening_element"), TokenType::Special);
```

### Identifier Validation

```rust
let js = JavaScript::new();

// Valid JavaScript identifiers
assert!(js.is_valid_identifier("foo"));
assert!(js.is_valid_identifier("_private"));
assert!(js.is_valid_identifier("$element"));    // jQuery-style
assert!(js.is_valid_identifier("$$internal"));  // Angular-style

// Invalid identifiers
assert!(!js.is_valid_identifier("123abc"));  // Starts with digit
assert!(!js.is_valid_identifier("my-var"));  // Contains hyphen
```

---

## Rholang

Rholang is a reflective, concurrent programming language based on the rho-calculus, designed for building scalable, secure blockchain applications on the RChain platform.

### Core Concepts

- **Channels (names)**: Communication endpoints prefixed with `@`
- **Processes**: Concurrent computations composed with `|`
- **Contracts**: Persistent receive operations
- **Bundles**: Access control for channels

### Usage

```rust
use libgrammstein::code::Rholang;

let rholang = Rholang::new();

// Access language information
assert_eq!(rholang.name(), "rholang");
assert!(!rholang.is_whitespace_significant());
```

### Characteristics

| Property | Value |
|----------|-------|
| File Extensions | `.rho` |
| Whitespace Significant | No |
| Comment Syntax | `//` (line), `/* */` (block), `///` (doc) |
| Paradigm | Concurrent, process algebra |

### Keywords

```
new     in      if      else    let     match   select  contract  for
or      and     matches not
bundle  bundle- bundle+ bundle0
true    false   Nil
```

### Special Tokens (Operators)

Rholang has a rich set of channel and process operators:

```rust
// Channel operations
"@"   // Quote (process -> name)
"*"   // Eval/dereference (name -> process)

// Send operations
"!"   // Send single
"!!"  // Send persistent
"!?"  // Synchronous send-then-receive

// Receive operations
"<-"  // Linear receive
"<="  // Persistent receive
"<<-" // Peek (non-consuming receive)
"?!"  // Receive-then-send

// Process algebra
"|"   // Parallel composition
"&"   // Concurrent binding
";"   // Sequential composition
"=>"  // Pattern match arm

// Set operations
"++"  // Union/concatenation
"--"  // Difference
"/\\" // Conjunction
"\\/" // Disjunction
"~"   // Negation
```

### Built-in Types

```rust
["Bool", "Int", "String", "Uri", "ByteArray", "Nil"]
```

### Token Classification

```rust
let rholang = Rholang::new();

// Keywords
assert_eq!(rholang.classify_token("new", "new"), TokenType::Keyword);
assert_eq!(rholang.classify_token("contract", "contract"), TokenType::Keyword);

// Boolean literals
assert_eq!(rholang.classify_token("true", "bool_literal"), TokenType::BooleanLiteral);

// Types
assert_eq!(rholang.classify_token("Int", "simple_type"), TokenType::TypeName);

// Variables
assert_eq!(rholang.classify_token("myVar", "var"), TokenType::Identifier);
```

### Identifier Validation

Rholang identifiers can include apostrophes (for mathematical notation):

```rust
let rholang = Rholang::new();

// Valid Rholang identifiers
assert!(rholang.is_valid_identifier("foo"));
assert!(rholang.is_valid_identifier("bar123"));
assert!(rholang.is_valid_identifier("_foo"));
assert!(rholang.is_valid_identifier("x'"));      // With apostrophe
assert!(rholang.is_valid_identifier("foo'bar")); // Apostrophe in middle

// Invalid identifiers
assert!(!rholang.is_valid_identifier("_"));      // Wildcard only
assert!(!rholang.is_valid_identifier("123foo")); // Starts with digit
assert!(!rholang.is_valid_identifier("@foo"));   // Starts with @
```

### Example Rholang Code

```rholang
// A simple contract that echoes messages
new echo, stdout(`rho:io:stdout`) in {
  contract echo(@msg, return) = {
    return!(msg) |
    stdout!(["Echo:", msg])
  } |
  new ack in {
    echo!("Hello", *ack) |
    for (@response <- ack) {
      stdout!(["Response:", response])
    }
  }
}
```

---

## MeTTa

MeTTa (Meta Type Talk) is a functional meta-programming language designed for knowledge representation, reasoning, and AI systems. It features hypergraph-based data structures and powerful pattern matching.

### Core Concepts

- **Atoms**: Basic units (symbols, variables, expressions)
- **Expressions**: S-expression lists `(expr expr ...)`
- **Variables**: Pattern variables prefixed with `$`
- **Spaces**: Atomspace references prefixed with `&`
- **Types**: Gradual typing with `:` annotations

### Usage

```rust
use libgrammstein::code::MeTTa;

let metta = MeTTa::new();

// Access language information
assert_eq!(metta.name(), "metta");
assert_eq!(metta.display_name(), "MeTTa");
```

### Characteristics

| Property | Value |
|----------|-------|
| File Extensions | `.metta`, `.mt` |
| Whitespace Significant | No |
| Comment Syntax | `;` (line), `;;` (doc) |
| Paradigm | Functional, homoiconic |

### Keywords

```
True    False   match   let     let*    if      case    function  return
empty   Error   Type    Atom    Symbol  Variable  Expression  Grounded
Unit    Number  String  Bool
new-space  add-atom  remove-atom  get-atoms  import!  include  bind!  pragma!
sequential  chain  eval  quote  unquote
```

### Special Tokens

```rust
// Prefix operators
"!"   // Reduction/evaluation
"?"   // Query
"'"   // Quote

// Variable prefix
"$"   // Pattern variable marker

// Space reference prefix
"&"   // Atomspace reference (e.g., &self)

// Type annotation
":"   // Type annotation

// Assignment/binding
"="   // Definition/equality
":="  // Rule definition

// Arrows
"->"  // Function type / transformation
```

### Built-in Types

```rust
// Core types
["Type", "Atom", "Symbol", "Variable", "Expression", "Grounded"]

// Primitive types
["Number", "String", "Bool", "Unit"]

// Collection types
["List", "Tuple"]

// Function types
["Function", "->"]

// Special types
["%Undefined%", "%Irreducible%"]
```

### Token Classification

```rust
let metta = MeTTa::new();

// Boolean literals
assert_eq!(metta.classify_token("True", "boolean_literal"), TokenType::BooleanLiteral);

// Numeric literals
assert_eq!(metta.classify_token("42", "integer_literal"), TokenType::NumericLiteral);
assert_eq!(metta.classify_token("3.14", "float_literal"), TokenType::NumericLiteral);

// Variables
assert_eq!(metta.classify_token("$x", "variable"), TokenType::Identifier);

// Keywords
assert_eq!(metta.classify_token("match", "identifier"), TokenType::Keyword);

// Space references
assert_eq!(metta.classify_token("&self", "space_reference"), TokenType::Special);
```

### Identifier Validation

MeTTa has a permissive identifier syntax:

```rust
let metta = MeTTa::new();

// Valid MeTTa identifiers (symbols)
assert!(metta.is_valid_identifier("foo"));
assert!(metta.is_valid_identifier("my-function"));  // Hyphens allowed
assert!(metta.is_valid_identifier("+"));            // Operators as symbols
assert!(metta.is_valid_identifier("->"));           // Arrow as symbol
assert!(metta.is_valid_identifier("_"));            // Wildcard

// Variables
assert!(metta.is_valid_identifier("$x"));
assert!(metta.is_valid_identifier("$var"));
assert!(metta.is_valid_identifier("$"));            // Empty variable

// Space references
assert!(metta.is_valid_identifier("&self"));
assert!(metta.is_valid_identifier("&kb"));
assert!(!metta.is_valid_identifier("&"));           // Empty not allowed

// Invalid identifiers
assert!(!metta.is_valid_identifier(""));            // Empty
assert!(!metta.is_valid_identifier("(foo"));        // Starts with delimiter
assert!(!metta.is_valid_identifier("foo(bar"));     // Contains delimiter
```

### Example MeTTa Code

```metta
; Define a type
(: add-numbers (-> Number Number Number))

; Define a function
(= (add-numbers $x $y)
   (+ $x $y))

; Pattern matching example
(= (factorial $n)
   (if (== $n 0)
       1
       (* $n (factorial (- $n 1)))))

; Using atomspaces
!(bind! &kb
  (new-space))

!(add-atom &kb (Person "Alice"))
!(add-atom &kb (knows (Person "Alice") (Person "Bob")))

; Query the space
!(match &kb
  (knows (Person "Alice") $who)
  $who)
```

---

## Comparison Table

| Feature | Python | Rust | JavaScript | Rholang | MeTTa |
|---------|--------|------|------------|---------|-------|
| Paradigm | OOP/Functional | Systems | Multi-paradigm | Concurrent | Functional |
| Whitespace | Significant | No | No | No | No |
| Line Comment | `#` | `//` | `//` | `//` | `;` |
| Block Comment | `"""` | `/* */` | `/* */` | `/* */` | None |
| Doc Comment | `#` | `///` | `///` | `///` | `;;` |
| Variable Prefix | None | None | `$` optional | None | `$` |
| Type Annotation | `:` | `:` | TypeScript | `:` | `:` |
| Unicode Identifiers | Yes | Yes | Yes | Yes | Yes |
| Raw Identifiers | No | `r#` | No | No | No |

## Thread Safety

All language implementations are `Send + Sync` and can be safely shared across threads:

```rust
use std::sync::Arc;
use std::thread;

let python = Arc::new(Python::new());
let rust = Arc::new(Rust::new());

let handles: Vec<_> = vec![
    {
        let lang = Arc::clone(&python);
        thread::spawn(move || lang.keywords().len())
    },
    {
        let lang = Arc::clone(&rust);
        thread::spawn(move || lang.keywords().len())
    },
];

for handle in handles {
    let count = handle.join().unwrap();
    println!("Keywords: {}", count);
}
```

## See Also

- [Language Framework](language.md) - `CodeLanguage` trait and `TokenType` system
- [AST](ast.md) - Tree-sitter integration
- [Tokenizer](tokenizer.md) - Token extraction from source code
- [Correction](correction.md) - How token types affect correction strategies
