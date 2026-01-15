# Lexical Corrector

The lexical corrector handles token-level spelling errors by maintaining dictionaries of valid tokens and using Levenshtein distance for fuzzy matching.

## Overview

The `LexicalCorrector` provides:

- **Fuzzy matching**: Find similar tokens within edit distance
- **Type-aware dictionaries**: Separate dictionaries for keywords, identifiers, types
- **Configurable thresholds**: Control edit distance and confidence scoring
- **Project adaptation**: Add project-specific identifiers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LexicalCorrector                         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Keywords   │  │   Types     │  │  Stdlib     │         │
│  │ Dictionary  │  │ Dictionary  │  │ Dictionary  │         │
│  │             │  │             │  │             │         │
│  │ if, while,  │  │ int, str,   │  │ print,      │         │
│  │ for, def... │  │ bool, Vec.. │  │ len, open.. │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │ Levenshtein     │                         │
│                 │ Distance Calc   │                         │
│                 └─────────────────┘                         │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────┤
│  │  Project Identifiers (dynamically added)                │
│  │  calculateTotal, processData, handleError, ...          │
│  └─────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
```

## LexicalCorrectorConfig

Configuration options for the lexical corrector:

```rust
pub struct LexicalCorrectorConfig {
    /// Maximum edit distance for corrections (default: 2)
    pub max_edit_distance: usize,
    /// Minimum token length to consider (default: 2)
    pub min_token_length: usize,
    /// Maximum candidates to return per token (default: 5)
    pub max_candidates: usize,
    /// Confidence penalty per edit operation (default: 0.15)
    pub edit_penalty: f64,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_edit_distance` | 2 | Maximum number of edits to consider |
| `min_token_length` | 2 | Skip tokens shorter than this |
| `max_candidates` | 5 | Maximum suggestions per token |
| `edit_penalty` | 0.15 | Confidence reduction per edit |

### Edit Distance Impact on Confidence

The confidence score is calculated as:

```
confidence = 1.0 - (edit_distance × edit_penalty)
```

| Edit Distance | Confidence (default penalty) |
|---------------|------------------------------|
| 1 | 0.85 |
| 2 | 0.70 |
| 3 | 0.55 (if allowed) |

## Creating a Lexical Corrector

### Basic Creation

```rust
use libgrammstein::code::{LexicalCorrector, LexicalCorrectorConfig, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());

// With default configuration
let corrector = LexicalCorrector::with_defaults(python.clone());

// With custom configuration
let config = LexicalCorrectorConfig {
    max_edit_distance: 3,
    min_token_length: 3,
    max_candidates: 10,
    edit_penalty: 0.1,
};
let corrector = LexicalCorrector::new(python, config);
```

### Adding Project Identifiers

The corrector automatically populates dictionaries from the language's keywords, types, and stdlib functions. You can add project-specific identifiers:

```rust
let mut corrector = LexicalCorrector::with_defaults(python);

// Add individual identifiers
corrector.add_identifier("calculateTotal");
corrector.add_identifier("processUserData");
corrector.add_identifier("handleNetworkError");

// Add from source code
let source = r#"
def calculateTotal(items):
    return sum(item.price for item in items)

def processUserData(user):
    return user.name.upper()
"#;
corrector.add_identifiers_from_source(source);

// Add from parsed tokens
corrector.add_identifiers_from_tokens(&tokens);
```

## Fuzzy Matching

The corrector uses Levenshtein distance to find similar tokens:

### Levenshtein Distance

| Original | Target | Distance | Operations |
|----------|--------|----------|------------|
| `pritn` | `print` | 1 | Swap i↔r |
| `retrun` | `return` | 1 | Swap r↔u |
| `funtion` | `function` | 1 | Insert c |
| `calulate` | `calculate` | 1 | Insert c |
| `fro` | `for` | 1 | Swap r↔o |

### Dictionary Selection

The corrector searches the appropriate dictionary based on token type:

```rust
match token_type {
    TokenType::Keyword   => search keywords dictionary,
    TokenType::TypeName  => search types dictionary,
    TokenType::Identifier => search identifiers + stdlib,
    _                    => search all dictionaries,
}
```

## Correcting Tokens

### Basic Token Correction

```rust
use libgrammstein::code::{CodeToken, TokenContext, TokenType, CodeCorrector};

let token = CodeToken::new(
    "funtion",           // Misspelled
    0,                   // byte offset
    1,                   // line
    0,                   // column
    TokenType::Keyword,  // Token type
    "keyword",           // AST node kind
);

let context = TokenContext::new(TokenType::Keyword);
let corrections = corrector.correct_token(&token, &context);

for c in &corrections {
    println!("{} → {} (confidence: {:.2})",
        c.original, c.replacement, c.confidence);
}
// Output: funtion → function (confidence: 0.85)
```

### Correcting a Range

```rust
let source = "def funtion(x): return x";
let corrections = corrector.correct_range(source, 4, 11);

// Corrects "funtion" at bytes 4-11
```

## Dictionary Management

### Inspecting Dictionaries

```rust
// Check dictionary sizes
println!("Keywords: {}", corrector.keyword_count());
println!("Identifiers: {}", corrector.identifier_count());
```

### Dictionary Sources

| Dictionary | Source | Examples |
|------------|--------|----------|
| Keywords | `language.keywords()` | `if`, `while`, `def`, `return` |
| Types | `language.builtin_types()` | `int`, `str`, `bool`, `list` |
| Stdlib | `language.stdlib_functions()` | `print`, `len`, `range`, `open` |
| Identifiers | User-added | Project-specific names |

## Confidence Scoring

The lexical corrector calculates confidence based on:

1. **Edit Distance**: Lower distance = higher confidence
2. **Token Type Match**: Keywords in keyword dictionary score higher

### Scoring Example

```rust
// Token: "retrun" (misspelled "return")
// Edit distance: 1 (swap r↔u)
// Base confidence: 1.0 - (1 × 0.15) = 0.85

let correction = Correction::new(
    CorrectionKind::Spelling,
    0, 6,
    "retrun", "return",
)
.with_confidence(0.85)
.with_source(CorrectionSource::Lexical)
.with_context("Edit distance: 1");
```

## Integration Example

Complete example using the lexical corrector:

```rust
use libgrammstein::code::{
    CodeParser, CodeTokenizer, LexicalCorrector,
    Python, CodeCorrector, TokenType
};
use std::sync::Arc;

fn correct_misspellings(source: &str) -> Vec<(String, String, f64)> {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();
    let tokenizer = CodeTokenizer::new(python.as_ref());
    let corrector = LexicalCorrector::with_defaults(python);

    // Parse and tokenize
    let parsed = parser.parse(source).unwrap();
    let tokens = tokenizer.tokenize(&parsed.tree, source);

    let mut results = Vec::new();

    // Check each token
    for token in tokens {
        // Only check keywords and identifiers
        if token.token_type != TokenType::Keyword
            && token.token_type != TokenType::Identifier {
            continue;
        }

        let corrections = corrector.correct_token(&token, &token.context);

        // Take best correction if confident
        if let Some(best) = corrections.first() {
            if best.confidence >= 0.7 {
                results.push((
                    token.text.clone(),
                    best.replacement.clone(),
                    best.confidence,
                ));
            }
        }
    }

    results
}

let source = "def calcluate(x):\n    retrun x + 1";
let fixes = correct_misspellings(source);

for (original, replacement, confidence) in fixes {
    println!("{} → {} ({:.0}%)", original, replacement, confidence * 100.0);
}
// Output:
// calcluate → calculate (85%)
// retrun → return (85%)
```

## Performance Optimization

### Length-Based Filtering

The corrector skips terms where length difference exceeds max edit distance:

```rust
// If max_edit_distance = 2
// Query: "fn" (length 2)
// "function" (length 8) is skipped - length diff of 6 > 2
```

### Dictionary Size Impact

| Dictionary Size | Lookup Time |
|-----------------|-------------|
| < 100 | ~1ms |
| 100-1000 | ~5ms |
| 1000-10000 | ~50ms |
| > 10000 | Consider caching |

### Best Practices

1. **Use appropriate max_edit_distance**: 2 is usually sufficient
2. **Skip short tokens**: `min_token_length: 2` avoids noise
3. **Limit candidates**: `max_candidates: 5` keeps results focused
4. **Pre-populate identifiers**: Add project identifiers at startup

## Thread Safety

`LexicalCorrector` is `Send + Sync` when its language type is:

```rust
use std::sync::Arc;
use std::thread;

let corrector = Arc::new(LexicalCorrector::with_defaults(python));

let handles: Vec<_> = tokens.into_iter().map(|token| {
    let corrector = Arc::clone(&corrector);
    thread::spawn(move || {
        corrector.correct_token(&token, &token.context)
    })
}).collect();

let results: Vec<_> = handles.into_iter()
    .map(|h| h.join().unwrap())
    .collect();
```

## Error Handling

The lexical corrector handles edge cases gracefully:

```rust
// Empty token - returns empty
let token = CodeToken::new("", 0, 0, 0, TokenType::Unknown, "");
assert!(corrector.correct_token(&token, &context).is_empty());

// Token too short - returns empty (if min_token_length > 1)
let token = CodeToken::new("x", 0, 0, 0, TokenType::Identifier, "");
assert!(corrector.correct_token(&token, &context).is_empty());

// Exact match - no suggestions (distance must be > 0)
let token = CodeToken::new("return", 0, 0, 0, TokenType::Keyword, "");
assert!(corrector.correct_token(&token, &context)
    .iter()
    .all(|c| c.replacement != "return"));
```

## See Also

- [Correctors Overview](overview.md) - Architecture and comparison
- [Grammar Corrector](grammar.md) - Syntax-based correction
- [Ensemble Corrector](ensemble.md) - Multi-source aggregation
- [Correction Framework](../correction.md) - Correction types
- [Language Framework](../language.md) - Token types
