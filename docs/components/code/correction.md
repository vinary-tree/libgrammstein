# Correction Framework

The correction module defines types and traits for representing and applying corrections to source code, integrating with liblevenshtein for fuzzy matching.

## Overview

The correction framework provides:

- **`Correction`**: A suggested fix with confidence score
- **`CorrectionKind`**: Classification of correction types
- **`CorrectionSource`**: Origin of the correction
- **`CodeCorrector`**: Trait for correction providers
- **`CorrectionCandidates`**: Ranked collection of suggestions

## Correction

The `Correction` struct represents a suggested fix:

```rust
pub struct Correction {
    /// The kind of correction
    pub kind: CorrectionKind,
    /// Start byte offset in the source
    pub start_byte: usize,
    /// End byte offset in the source
    pub end_byte: usize,
    /// The original text
    pub original: String,
    /// The suggested replacement text
    pub replacement: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Source of the correction (which component suggested it)
    pub source: CorrectionSource,
    /// Additional context about the correction
    pub context: Option<String>,
}
```

### Creating Corrections

```rust
use libgrammstein::code::{Correction, CorrectionKind, CorrectionSource};

// Basic correction
let correction = Correction::new(
    CorrectionKind::Spelling,
    0,       // start_byte
    5,       // end_byte
    "pritn", // original
    "print", // replacement
);

// With builder pattern
let correction = Correction::new(
    CorrectionKind::Spelling,
    0,
    5,
    "pritn",
    "print",
)
.with_confidence(0.95)
.with_source(CorrectionSource::Lexical)
.with_context("Likely typo in function name");
```

### Applying Corrections

```rust
let source = "pritn(\"hello\")";
let correction = Correction::new(
    CorrectionKind::Spelling,
    0, 5, "pritn", "print"
);

let fixed = correction.apply(source);
assert_eq!(fixed, "print(\"hello\")");
```

### Correction Properties

```rust
let correction = Correction::new(
    CorrectionKind::Spelling,
    0, 5, "pritn", "print"
);

// Check if no-op
if correction.is_noop() {
    println!("No change needed");
}

// Estimate edit distance
let distance = correction.edit_distance();
println!("Edit distance: {}", distance);
```

## CorrectionKind

Classification of correction types:

```rust
pub enum CorrectionKind {
    Spelling,       // Typo correction (retrun -> return)
    Insertion,      // Missing token (add missing ;)
    Deletion,       // Extra token (remove duplicate)
    Replacement,    // Wrong token (semantic)
    VariableMisuse, // Wrong variable name
    TypeError,      // Type mismatch
    MissingImport,  // Import needed
    SyntaxError,    // General syntax fix
    Formatting,     // Whitespace/formatting
    Other,          // Uncategorized
}
```

### Kind Properties

```rust
let kind = CorrectionKind::VariableMisuse;

// Human-readable description
println!("{}", kind.description()); // "Wrong variable name"

// Check if semantic (vs. syntactic)
if kind.is_semantic() {
    println!("Semantic correction (needs type/data flow analysis)");
} else {
    println!("Syntactic correction (lexical/grammar level)");
}
```

### Semantic vs Syntactic

| Kind | Category | Analysis Required |
|------|----------|-------------------|
| `Spelling` | Syntactic | Lexical fuzzy matching |
| `Insertion` | Syntactic | Grammar parsing |
| `Deletion` | Syntactic | Grammar parsing |
| `SyntaxError` | Syntactic | Grammar parsing |
| `Formatting` | Syntactic | Token analysis |
| `VariableMisuse` | Semantic | Data flow analysis |
| `TypeError` | Semantic | Type inference |
| `MissingImport` | Semantic | Symbol resolution |
| `Replacement` | Mixed | Context-dependent |

## CorrectionSource

Origin of a correction suggestion:

```rust
pub enum CorrectionSource {
    Lexical,       // liblevenshtein fuzzy matching
    Grammar,       // PCFG/Earley parsing
    Neural,        // UniXcoder/GNN models
    TypeInference, // Type system analysis
    ControlFlow,   // CFG analysis
    DataFlow,      // DFG analysis
    Combined,      // Ensemble/aggregated
    Unknown,       // Not specified
}
```

### Using Source for Debugging

```rust
for correction in candidates.ranked() {
    match correction.source {
        CorrectionSource::Lexical => {
            println!("From fuzzy matching");
        }
        CorrectionSource::Grammar => {
            println!("From grammar constraints");
        }
        CorrectionSource::Neural => {
            println!("From neural model");
        }
        _ => {}
    }
}
```

## CodeCorrector Trait

The `CodeCorrector` trait defines the interface for correction providers:

```rust
pub trait CodeCorrector: Send + Sync {
    /// Suggests corrections for a token.
    fn correct_token(
        &self,
        token: &CodeToken,
        context: &TokenContext
    ) -> Vec<Correction>;

    /// Suggests corrections for a range of source code.
    fn correct_range(
        &self,
        source: &str,
        start_byte: usize,
        end_byte: usize,
    ) -> Vec<Correction>;

    /// Returns the maximum edit distance considered.
    fn max_edit_distance(&self) -> usize { 2 }

    /// Returns the name of this corrector.
    fn name(&self) -> &str;
}
```

### Implementing a Custom Corrector

```rust
use libgrammstein::code::{
    CodeCorrector, Correction, CorrectionKind, CorrectionSource,
    CodeToken, TokenContext
};

struct MyCustomCorrector {
    dictionary: Vec<String>,
}

impl CodeCorrector for MyCustomCorrector {
    fn correct_token(
        &self,
        token: &CodeToken,
        context: &TokenContext
    ) -> Vec<Correction> {
        let mut corrections = Vec::new();

        // Simple example: check if token is in dictionary
        if !self.dictionary.contains(&token.text) {
            // Find closest match
            for word in &self.dictionary {
                if similar(word, &token.text) {
                    corrections.push(
                        Correction::new(
                            CorrectionKind::Spelling,
                            token.byte_offset,
                            token.byte_offset + token.text.len(),
                            &token.text,
                            word,
                        )
                        .with_source(CorrectionSource::Lexical)
                        .with_confidence(0.8)
                    );
                }
            }
        }

        corrections
    }

    fn correct_range(
        &self,
        source: &str,
        start_byte: usize,
        end_byte: usize,
    ) -> Vec<Correction> {
        // Delegate to token-based correction
        Vec::new()
    }

    fn name(&self) -> &str {
        "MyCustomCorrector"
    }
}
```

## CorrectionCandidates

A ranked collection of correction suggestions:

```rust
pub struct CorrectionCandidates {
    corrections: Vec<Correction>,
    max_candidates: usize,
}
```

### Creating and Using Candidates

```rust
use libgrammstein::code::{
    CorrectionCandidates, Correction, CorrectionKind
};

// Create with maximum capacity
let mut candidates = CorrectionCandidates::new(10);

// Add corrections
candidates.add(
    Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "print")
        .with_confidence(0.95)
);
candidates.add(
    Correction::new(CorrectionKind::Spelling, 0, 5, "pritn", "pint")
        .with_confidence(0.7)
);

// Get best suggestion
if let Some(best) = candidates.best() {
    println!("Best: {} (confidence: {:.2})", best.replacement, best.confidence);
}

// Get all ranked
for (i, correction) in candidates.ranked().iter().enumerate() {
    println!("{}. {} ({:.2})", i + 1, correction.replacement, correction.confidence);
}
```

### Adding Multiple Candidates

```rust
let lexical_corrections = lexical_corrector.correct_token(&token, &context);
let grammar_corrections = grammar_corrector.correct_token(&token, &context);

let mut candidates = CorrectionCandidates::new(10);
candidates.add_all(lexical_corrections);
candidates.add_all(grammar_corrections);

// Automatically sorted by confidence
```

### Filtering Candidates

```rust
let mut candidates = CorrectionCandidates::new(10);
// ... add corrections ...

// Filter by minimum confidence
candidates.filter_by_confidence(0.5);

// Filter by kind
candidates.filter_by_kind(CorrectionKind::Spelling);

// Filter by source
candidates.filter_by_source(CorrectionSource::Lexical);
```

### Iterating Candidates

```rust
// By reference
for correction in &candidates {
    println!("{} -> {}", correction.original, correction.replacement);
}

// By value (consuming)
for correction in candidates {
    let fixed = correction.apply(source);
}
```

## Confidence Scoring

Confidence scores range from 0.0 to 1.0:

| Score Range | Meaning |
|-------------|---------|
| 0.9 - 1.0 | Very high confidence (exact match, distance 1) |
| 0.7 - 0.9 | High confidence (distance 2, strong context) |
| 0.5 - 0.7 | Medium confidence (multiple sources agree) |
| 0.3 - 0.5 | Low confidence (single source, no context) |
| 0.0 - 0.3 | Very low confidence (weak suggestion) |

### Confidence Calculation Example

```rust
fn calculate_confidence(
    edit_distance: usize,
    token_type: TokenType,
    in_error: bool,
) -> f64 {
    let base = match edit_distance {
        0 => 1.0,
        1 => 0.9,
        2 => 0.7,
        _ => 0.5,
    };

    // Boost for keywords (fixed vocabulary)
    let type_boost = if token_type == TokenType::Keyword {
        1.1
    } else {
        1.0
    };

    // Boost if in error region
    let error_boost = if in_error { 1.1 } else { 1.0 };

    (base * type_boost * error_boost).min(1.0)
}
```

## Applying Multiple Corrections

When applying multiple corrections, apply from end to start:

```rust
fn apply_corrections(source: &str, corrections: &[Correction]) -> String {
    // Sort by start position descending
    let mut sorted: Vec<_> = corrections.iter().collect();
    sorted.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

    let mut result = source.to_string();
    for correction in sorted {
        result = format!(
            "{}{}{}",
            &result[..correction.start_byte],
            correction.replacement,
            &result[correction.end_byte..]
        );
    }

    result
}
```

## Integration Example

Complete example using the correction framework:

```rust
use libgrammstein::code::{
    CodeParser, CodeTokenizer, LexicalCorrector,
    CorrectionCandidates, Python
};
use std::sync::Arc;

fn correct_code(source: &str) -> String {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();

    // Parse
    let parsed = parser.parse(source).unwrap();
    if !parsed.has_errors {
        return source.to_string();
    }

    // Tokenize error regions
    let tokenizer = CodeTokenizer::new(python.as_ref());
    let error_tokens = tokenizer.tokenize_errors(&parsed.tree, source);

    // Create corrector
    let corrector = LexicalCorrector::with_defaults(python);

    // Collect all corrections
    let mut all_corrections = Vec::new();
    for token in error_tokens {
        let mut candidates = CorrectionCandidates::new(5);
        candidates.add_all(corrector.correct_token(&token, &token.context));

        if let Some(best) = candidates.best() {
            if best.confidence >= 0.7 {
                all_corrections.push(best.clone());
            }
        }
    }

    // Apply corrections (from end to start)
    apply_corrections(source, &all_corrections)
}
```

## Thread Safety

All correction types are `Send + Sync`:

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
```

## See Also

- [Correctors Overview](correctors/overview.md) - Corrector implementations
- [Lexical Corrector](correctors/lexical.md) - Fuzzy matching
- [Grammar Corrector](correctors/grammar.md) - PCFG-based
- [Semantic Corrector](correctors/semantic.md) - CPG-based
- [Ensemble Corrector](correctors/ensemble.md) - Combining correctors
- [Pipeline](pipeline.md) - End-to-end workflow
