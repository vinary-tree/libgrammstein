# Semantic Corrector

The semantic corrector analyzes code semantics using Code Property Graphs (CPG) and Graph Neural Networks (GNN) to detect contextual issues like variable misuse, unused bindings, and type errors.

## Overview

The `SemanticCorrector` provides:

- **Variable misuse detection**: Wrong variable names in context
- **Unused binding detection**: Variables defined but never used
- **Type error detection**: Type mismatches (when type info available)
- **GNN-based scoring**: Neural network analysis of code patterns

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    SemanticCorrector                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  GnnSemanticScorer                          │ │
│  │                                                             │ │
│  │  • Feature extraction from CPG nodes                        │ │
│  │  • Message passing for pattern detection                    │ │
│  │  • Issue scoring and ranking                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │               CodePropertyGraph (CPG)                       │ │
│  │                                                             │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │ │
│  │  │   AST   │    │   CFG   │    │   DFG   │                 │ │
│  │  │         │    │         │    │         │                 │ │
│  │  │ Parent  │    │ Next    │    │ Read    │                 │ │
│  │  │ Child   │    │ Branch  │    │ Write   │                 │ │
│  │  │ Sibling │    │ Back    │    │ Flow    │                 │ │
│  │  └─────────┘    └─────────┘    └─────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────┴────────────────────────────────┐ │
│  │              Known Variables / Functions                    │ │
│  │                                                             │ │
│  │  Variables: userCount(int), userName(string), ...           │ │
│  │  Functions: calculateTotal(2), processData(1), ...          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## SemanticCorrectorConfig

Configuration options for the semantic corrector:

```rust
pub struct SemanticCorrectorConfig {
    /// Minimum confidence threshold for reporting (default: 0.5)
    pub min_confidence: f64,
    /// Maximum candidates per issue (default: 5)
    pub max_candidates: usize,
    /// Whether to check for variable misuse (default: true)
    pub check_variable_misuse: bool,
    /// Whether to check for unused bindings (default: true)
    pub check_unused_bindings: bool,
    /// Whether to check for type errors (default: true)
    pub check_type_errors: bool,
    /// GNN configuration
    pub gnn_config: GnnConfig,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_confidence` | 0.5 | Threshold for reporting issues |
| `max_candidates` | 5 | Maximum suggestions per issue |
| `check_variable_misuse` | true | Enable variable misuse detection |
| `check_unused_bindings` | true | Enable unused variable detection |
| `check_type_errors` | true | Enable type mismatch detection |

## Creating a Semantic Corrector

### Basic Creation

```rust
use libgrammstein::code::{SemanticCorrector, SemanticCorrectorConfig, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());

// With default configuration
let corrector = SemanticCorrector::with_defaults(python.clone());

// With custom configuration
let config = SemanticCorrectorConfig {
    min_confidence: 0.6,
    max_candidates: 10,
    check_variable_misuse: true,
    check_unused_bindings: true,
    check_type_errors: false,  // Disable type checking
    gnn_config: GnnConfig::default(),
};
let corrector = SemanticCorrector::new(python, config);
```

### Registering Project Context

Provide the corrector with knowledge of your project's symbols:

```rust
let mut corrector = SemanticCorrector::with_defaults(python);

// Register known variables with optional type information
corrector.register_variable(
    "userCount".to_string(),
    Some("int".to_string()),
    0,  // scope level
);
corrector.register_variable(
    "userName".to_string(),
    Some("string".to_string()),
    0,
);

// Register known functions with arity and return type
corrector.register_function(
    "calculateTotal".to_string(),
    2,  // arity (number of parameters)
    Some("float".to_string()),  // return type
);
```

## VariableInfo and FunctionInfo

### VariableInfo

Information about a known variable:

```rust
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Inferred or declared type (if known)
    pub type_name: Option<String>,
    /// Scope level where defined
    pub scope_level: usize,
    /// Number of times used
    pub use_count: usize,
}
```

### FunctionInfo

Information about a known function:

```rust
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    /// Parameter types (if known)
    pub param_types: Vec<Option<String>>,
    /// Return type (if known)
    pub return_type: Option<String>,
    /// Number of parameters
    pub arity: usize,
}
```

## Analyzing Code Property Graphs

### Basic CPG Analysis

```rust
use libgrammstein::code::{CodeParser, CodePropertyGraph, Python};

let python = Arc::new(Python::new());
let mut parser = CodeParser::new(python.clone()).unwrap();
let corrector = SemanticCorrector::with_defaults(python);

let source = r#"
def calculate(x, y):
    result = x + y
    return resutl  # Typo: should be 'result'
"#;

let parsed = parser.parse(source).unwrap();
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

// Analyze CPG for semantic issues
let issues = corrector.analyze_cpg(&cpg);

for issue in &issues {
    println!("Issue at node {}: {:?} (confidence: {:.2})",
        issue.node_idx, issue.issue_type, issue.confidence);
}
```

### Full Analysis with Corrections

```rust
// Get corrections from parsed code and CPG
let corrections = corrector.analyze_parsed(&parsed, &cpg);

for correction in &corrections {
    println!("{} → {} ({:?}, confidence: {:.2})",
        correction.original,
        correction.replacement,
        correction.kind,
        correction.confidence
    );
}
```

## Issue Types

The semantic corrector detects several issue types:

```rust
pub enum IssueType {
    VariableMisuse,        // Wrong variable in context
    UnusedBinding,         // Variable defined but never used
    TypeError,             // Type mismatch
    MissingErrorHandling,  // Unhandled error case
    // ... other types
}
```

### Variable Misuse

Detects when a variable name is likely wrong:

```rust
// Source: "return resutl" (should be "result")
// Issue: VariableMisuse at "resutl" node
// Suggestion: "result" based on data flow and name similarity
```

### Unused Binding

Detects variables that are defined but never read:

```rust
// Source:
// def foo():
//     unused = 42  # Never read
//     return 0

// Issue: UnusedBinding at "unused" definition
// Suggestion: Remove or use the variable
```

### Type Error

Detects type mismatches when type information is available:

```rust
// Source (with type annotations):
// def add(x: int, y: int) -> int:
//     return x + "hello"  # Type error

// Issue: TypeError at string literal
// Suggestion: Expected int
```

## Finding Variable Misuse

Find candidates for variable replacement:

```rust
// Find alternatives for a potentially misused variable
let candidates = corrector.find_variable_misuse(&cpg, "resutl", node_idx);

for (name, score) in &candidates {
    println!("  {} (score: {:.2})", name, score);
}
// Output:
//   result (score: 0.85)
//   results (score: 0.65)
```

## Name Similarity

The corrector uses Levenshtein distance for name similarity:

```rust
// Similarity calculation
fn name_similarity(a: &str, b: &str) -> f64 {
    if a == b { return 1.0; }
    let distance = levenshtein_distance(a, b);
    let max_len = a.len().max(b.len());
    1.0 - (distance as f64 / max_len as f64)
}

// Examples:
// "result" vs "resutl" → similarity ~0.83
// "count" vs "counter" → similarity ~0.71
// "foo" vs "bar" → similarity ~0.0
```

## Token-Level Correction

The semantic corrector can also correct individual tokens:

```rust
use libgrammstein::code::{CodeToken, TokenContext, TokenType, CodeCorrector};

let mut corrector = SemanticCorrector::with_defaults(python);

// Register known identifiers
corrector.register_variable("calculateTotal".to_string(), None, 0);
corrector.register_variable("calculateAverage".to_string(), None, 0);

// Correct an unknown identifier
let token = CodeToken::new(
    "calulateTotal",  // Misspelled
    0, 1, 0,
    TokenType::Identifier,
    "identifier",
);

let context = TokenContext::new(TokenType::Identifier);
let corrections = corrector.correct_token(&token, &context);

// Suggests: "calculateTotal" (high similarity)
```

## Correction Sources

Semantic corrections are tagged with their analysis source:

```rust
pub enum CorrectionSource {
    Neural,         // From GNN analysis
    TypeInference,  // From type checking
    ControlFlow,    // From CFG analysis
    DataFlow,       // From DFG analysis
    // ...
}
```

Usage:

```rust
for correction in corrections {
    match correction.source {
        CorrectionSource::Neural => {
            println!("GNN detected: {}", correction.context.as_deref().unwrap_or(""));
        }
        CorrectionSource::DataFlow => {
            println!("Data flow issue: {}", correction.context.as_deref().unwrap_or(""));
        }
        CorrectionSource::TypeInference => {
            println!("Type error: {}", correction.context.as_deref().unwrap_or(""));
        }
        _ => {}
    }
}
```

## Integration Example

Complete example using the semantic corrector:

```rust
use libgrammstein::code::{
    CodeParser, CodeTokenizer, CodePropertyGraph,
    SemanticCorrector, Python, CorrectionKind
};
use std::sync::Arc;

fn analyze_semantics(source: &str) -> Vec<String> {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();
    let mut corrector = SemanticCorrector::with_defaults(python.clone());

    // Parse and build CPG
    let parsed = parser.parse(source).unwrap();
    let cpg = CodePropertyGraph::from_parsed_code(&parsed);

    // Extract and register known variables from the code
    let tokenizer = CodeTokenizer::new(python.as_ref());
    let tokens = tokenizer.tokenize(&parsed.tree, source);

    for token in &tokens {
        if token.token_type == TokenType::Identifier {
            // Check if this is a definition (simplified check)
            if let Some(parent) = &token.context.parent_node_type {
                if parent.contains("assignment") || parent.contains("parameter") {
                    corrector.register_variable(token.text.clone(), None, 0);
                }
            }
        }
    }

    // Analyze for semantic issues
    let corrections = corrector.analyze_parsed(&parsed, &cpg);

    let mut messages = Vec::new();
    for c in &corrections {
        let msg = match c.kind {
            CorrectionKind::VariableMisuse => {
                format!("Variable '{}' might be '{}' (line {})",
                    c.original, c.replacement,
                    // Would need line mapping
                    0
                )
            }
            CorrectionKind::Deletion => {
                format!("'{}' appears to be unused", c.original)
            }
            CorrectionKind::TypeError => {
                format!("Type error at '{}': {}", c.original,
                    c.context.as_deref().unwrap_or(""))
            }
            _ => format!("Issue at '{}': {:?}", c.original, c.kind),
        };
        messages.push(msg);
    }

    messages
}

let source = r#"
def process_data(items):
    total = 0
    for item in items:
        totla += item.value  # Typo: should be 'total'
    return total
"#;

let issues = analyze_semantics(source);
for issue in issues {
    println!("  {}", issue);
}
```

## GNN Integration

The semantic corrector uses `GnnSemanticScorer` for pattern detection:

```rust
// Access the GNN scorer
let scorer = corrector.gnn_scorer();

// The scorer extracts features from CPG nodes
// and uses message passing to detect semantic patterns
```

### GnnConfig

Configure the GNN behavior:

```rust
pub struct GnnConfig {
    /// Number of message passing layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_dim: usize,
    /// Dropout rate
    pub dropout: f64,
    // ...
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| CPG analysis | O(n + e) | n = nodes, e = edges |
| Variable misuse | O(v) | v = known variables |
| Name similarity | O(len²) | Levenshtein distance |
| GNN inference | O(L × n) | L = layers, n = nodes |

### Optimization Tips

1. **Register variables early**: Populate known variables at project load
2. **Use confidence threshold**: Set `min_confidence` appropriately
3. **Disable unused checks**: Turn off `check_unused_bindings` if not needed
4. **Cache CPG**: Build CPG once and reuse for multiple analyses

## Thread Safety

`SemanticCorrector` is `Send + Sync` when its language type is:

```rust
use std::sync::Arc;

let corrector = Arc::new(SemanticCorrector::with_defaults(python));

// Share across threads for read-only analysis
let results: Vec<_> = cpgs.par_iter()
    .map(|cpg| corrector.analyze_cpg(cpg))
    .collect();
```

Note: Modifying the corrector (registering variables) requires mutable access.

## Limitations

1. **Requires full AST**: Token-level correction is limited
2. **Type inference**: Depends on available type annotations
3. **Scope tracking**: Simplified scope model
4. **Cross-file analysis**: Limited to single-file context

## See Also

- [Correctors Overview](overview.md) - Architecture and comparison
- [Lexical Corrector](lexical.md) - Fuzzy matching
- [Grammar Corrector](grammar.md) - Syntax-based correction
- [Ensemble Corrector](ensemble.md) - Multi-source aggregation
- [CPG](../cpg.md) - Code Property Graphs
- [GNN](../gnn.md) - Graph Neural Networks
