# Ensemble Corrector

The ensemble corrector combines suggestions from lexical, grammar, and semantic correctors using configurable weighting, deduplication, and agreement boosting to produce a unified ranked list of corrections.

## Overview

The `EnsembleCorrector` provides:

- **Multi-source aggregation**: Combine corrections from all corrector types
- **Weighted scoring**: Configurable importance per source
- **Deduplication**: Merge identical suggestions
- **Agreement boosting**: Increase confidence when sources agree

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      EnsembleCorrector                             │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Lexical    │  │   Grammar    │  │   Semantic   │             │
│  │  Corrector   │  │  Corrector   │  │  Corrector   │             │
│  │              │  │              │  │              │             │
│  │ weight: 0.40 │  │ weight: 0.35 │  │ weight: 0.25 │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                      │
│         └────────────┬────┴────────────────┘                       │
│                      ▼                                             │
│         ┌────────────────────────────┐                             │
│         │    Correction Collection    │                            │
│         └────────────────────────────┘                             │
│                      │                                             │
│                      ▼                                             │
│         ┌────────────────────────────┐                             │
│         │    Deduplication & Merge   │                             │
│         │                            │                             │
│         │  • Group by (replacement,  │                             │
│         │    start_byte, end_byte)   │                             │
│         │  • Merge identical         │                             │
│         │  • Agreement boost         │                             │
│         └────────────────────────────┘                             │
│                      │                                             │
│                      ▼                                             │
│         ┌────────────────────────────┐                             │
│         │    Ranking & Filtering     │                             │
│         │                            │                             │
│         │  • Filter by min_confidence│                             │
│         │  • Sort by confidence      │                             │
│         │  • Truncate to max_cands   │                             │
│         └────────────────────────────┘                             │
│                      │                                             │
│                      ▼                                             │
│              Ranked Corrections                                    │
└────────────────────────────────────────────────────────────────────┘
```

## EnsembleCorrectorConfig

Configuration options for the ensemble corrector:

```rust
pub struct EnsembleCorrectorConfig {
    /// Weight for lexical corrections (default: 0.4)
    pub lexical_weight: f64,
    /// Weight for grammar corrections (default: 0.35)
    pub grammar_weight: f64,
    /// Weight for semantic corrections (default: 0.25)
    pub semantic_weight: f64,
    /// Minimum confidence to include (default: 0.3)
    pub min_confidence: f64,
    /// Maximum total candidates (default: 10)
    pub max_candidates: usize,
    /// Whether to deduplicate (default: true)
    pub deduplicate: bool,
    /// Similarity threshold for deduplication (default: 0.9)
    pub dedup_threshold: f64,
    /// Whether to boost on agreement (default: true)
    pub agreement_boost: bool,
    /// Boost factor when sources agree (default: 1.3)
    pub agreement_boost_factor: f64,
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lexical_weight` | 0.4 | Weight for spelling corrections |
| `grammar_weight` | 0.35 | Weight for syntax corrections |
| `semantic_weight` | 0.25 | Weight for semantic corrections |
| `min_confidence` | 0.3 | Minimum confidence threshold |
| `max_candidates` | 10 | Maximum results to return |
| `deduplicate` | true | Merge identical suggestions |
| `dedup_threshold` | 0.9 | Similarity for merging |
| `agreement_boost` | true | Boost when sources agree |
| `agreement_boost_factor` | 1.3 | Multiplier for agreement |

## Creating an Ensemble Corrector

### Basic Creation

```rust
use libgrammstein::code::{EnsembleCorrector, Python, WeightedCFG};
use std::sync::Arc;

let python = Arc::new(Python::new());

// Without grammar (lexical + semantic only)
let corrector = EnsembleCorrector::with_defaults(python.clone(), None);

// With grammar
let grammar = build_python_grammar();  // Your PCFG
let corrector = EnsembleCorrector::with_defaults(python, Some(grammar));
```

### Lexical-Only Mode

For fast, lightweight correction:

```rust
let corrector = EnsembleCorrector::lexical_only(python);
// Only lexical corrections, no grammar or semantic
```

### Custom Configuration

```rust
use libgrammstein::code::EnsembleCorrectorConfig;

let config = EnsembleCorrectorConfig {
    lexical_weight: 0.5,     // Prioritize spelling
    grammar_weight: 0.3,
    semantic_weight: 0.2,
    min_confidence: 0.4,     // Higher threshold
    max_candidates: 5,       // Fewer results
    deduplicate: true,
    dedup_threshold: 0.9,
    agreement_boost: true,
    agreement_boost_factor: 1.5,  // Strong agreement boost
};

let corrector = EnsembleCorrector::new(python, Some(grammar), config);
```

## EnsembleCorrectorBuilder

Use the builder pattern for flexible configuration:

```rust
use libgrammstein::code::EnsembleCorrectorBuilder;

let corrector = EnsembleCorrectorBuilder::new(python)
    // Set weights
    .lexical_weight(0.5)
    .grammar_weight(0.3)
    .semantic_weight(0.2)
    // Provide grammar
    .with_grammar(grammar)
    // Disable components
    .without_semantic()  // Skip semantic analysis
    // Custom config
    .with_config(EnsembleCorrectorConfig {
        min_confidence: 0.5,
        ..Default::default()
    })
    .build();
```

### Builder Methods

| Method | Description |
|--------|-------------|
| `with_grammar(grammar)` | Enable grammar correction |
| `with_config(config)` | Set custom configuration |
| `without_lexical()` | Disable lexical correction |
| `without_grammar()` | Disable grammar correction |
| `without_semantic()` | Disable semantic correction |
| `lexical_weight(w)` | Set lexical weight |
| `grammar_weight(w)` | Set grammar weight |
| `semantic_weight(w)` | Set semantic weight |

## Adding Project Context

### Adding Identifiers

```rust
let mut corrector = EnsembleCorrector::with_defaults(python, None);

// Add identifiers to lexical corrector
corrector.add_identifiers(&[
    "calculateTotal",
    "processUserData",
    "handleNetworkError",
]);
```

### Registering Variables

```rust
// Register variables with semantic corrector
corrector.register_variables(&[
    ("userCount".to_string(), Some("int".to_string())),
    ("userName".to_string(), Some("string".to_string())),
    ("isActive".to_string(), Some("bool".to_string())),
]);
```

### Accessing Sub-Correctors

```rust
// Modify lexical corrector directly
if let Some(lexical) = corrector.lexical_mut() {
    lexical.add_identifier("customFunction");
}

// Modify semantic corrector directly
if let Some(semantic) = corrector.semantic_mut() {
    semantic.register_function("customFunc".to_string(), 2, Some("void".to_string()));
}
```

## Correcting Tokens

### Basic Token Correction

```rust
use libgrammstein::code::{CodeToken, TokenContext, TokenType, CodeCorrector};

let token = CodeToken::new(
    "funtion",           // Misspelled "function"
    0,
    1,
    0,
    TokenType::Keyword,
    "keyword",
);

let context = TokenContext::new(TokenType::Keyword);
let corrections = corrector.correct_token(&token, &context);

for c in &corrections {
    println!("{} → {} (source: {:?}, confidence: {:.2})",
        c.original, c.replacement, c.source, c.confidence);
}
```

### Correcting a Range

```rust
let source = "def funtion(x): retrun x";
let corrections = corrector.correct_range(source, 4, 11);
// Corrects "funtion" at bytes 4-11
```

## Full Analysis with CPG

For comprehensive analysis including semantic issues:

```rust
use libgrammstein::code::{CodeParser, CodePropertyGraph};

let mut parser = CodeParser::new(python.clone()).unwrap();
let parsed = parser.parse(source).unwrap();
let cpg = CodePropertyGraph::from_parsed_code(&parsed);

// Full analysis with CPG
let corrections = corrector.analyze_full(&parsed, &cpg);
```

## Weighting and Scoring

### Weight Application

Each correction's confidence is multiplied by its source weight:

```rust
// Original confidence from lexical: 0.85
// Lexical weight: 0.4
// Weighted confidence: 0.85 × 0.4 = 0.34

// Original confidence from grammar: 0.75
// Grammar weight: 0.35
// Weighted confidence: 0.75 × 0.35 = 0.26
```

### Agreement Boosting

When multiple sources suggest the same correction:

```rust
// Lexical suggests: "function" (confidence: 0.34)
// Grammar suggests: "function" (confidence: 0.26)

// Combined average: (0.34 + 0.26) / 2 = 0.30
// Agreement boost: 0.30 × 1.3 = 0.39

// Final correction:
// - replacement: "function"
// - source: Combined
// - confidence: 0.39
// - context: "Suggested by 2 sources"
```

## Merging Corrections

### Deduplication Process

Corrections are grouped by `(replacement, start_byte, end_byte)`:

```rust
// Before merging:
// 1. Lexical: "function" @ 0-7, confidence 0.34
// 2. Grammar: "function" @ 0-7, confidence 0.26
// 3. Lexical: "functor" @ 0-7, confidence 0.20

// After merging:
// 1. Combined: "function" @ 0-7, confidence 0.39 (boosted)
// 2. Lexical: "functor" @ 0-7, confidence 0.20
```

### Merge Logic

```rust
fn merge_corrections(corrections: Vec<(Correction, f64)>) -> Vec<Correction> {
    // Group by (replacement, start_byte, end_byte)
    let mut groups = HashMap::new();
    for (c, weight) in corrections {
        let key = (c.replacement.clone(), c.start_byte, c.end_byte);
        groups.entry(key).or_default().push((c, weight));
    }

    // Process each group
    for (key, group) in groups {
        if group.len() == 1 {
            // Single source: apply weight only
            result.push(apply_weight(group[0]));
        } else {
            // Multiple sources: merge and boost
            let merged = merge_group(group);
            result.push(merged);
        }
    }
}
```

## Correction Flow

Complete correction pipeline:

```
Token + Context
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
   Lexical            Grammar           Semantic
   Corrector          Corrector         Corrector
       │                  │                  │
       │ [c1, c2]        │ [c3]            │ [c4, c5]
       │                  │                  │
       └──────────────────┴──────────────────┘
                          │
                          ▼
              Apply Weights: [(c1,0.4), (c2,0.4),
                             (c3,0.35), (c4,0.25), (c5,0.25)]
                          │
                          ▼
              Group by (replacement, position)
                          │
                          ▼
              Merge duplicates + Agreement boost
                          │
                          ▼
              Filter by min_confidence (0.3)
                          │
                          ▼
              Sort by confidence descending
                          │
                          ▼
              Truncate to max_candidates (10)
                          │
                          ▼
              Final Ranked Corrections
```

## Integration Example

Complete example using the ensemble corrector:

```rust
use libgrammstein::code::{
    CodeParser, CodeTokenizer, CodePropertyGraph,
    EnsembleCorrectorBuilder, Python, CorrectionSource
};
use std::sync::Arc;

fn correct_code(source: &str) -> String {
    let python = Arc::new(Python::new());
    let mut parser = CodeParser::new(python.clone()).unwrap();

    // Build ensemble corrector
    let mut corrector = EnsembleCorrectorBuilder::new(python.clone())
        .lexical_weight(0.5)
        .grammar_weight(0.3)
        .semantic_weight(0.2)
        .build();

    // Parse source
    let parsed = parser.parse(source).unwrap();

    if !parsed.has_errors {
        return source.to_string();
    }

    // Build CPG for semantic analysis
    let cpg = CodePropertyGraph::from_parsed_code(&parsed);

    // Extract tokens from error regions
    let tokenizer = CodeTokenizer::new(python.as_ref());
    let error_tokens = tokenizer.tokenize_errors(&parsed.tree, source);

    // Populate identifiers from all tokens
    let all_tokens = tokenizer.tokenize(&parsed.tree, source);
    for token in &all_tokens {
        if token.token_type == TokenType::Identifier {
            corrector.add_identifiers(&[&token.text]);
        }
    }

    // Collect high-confidence corrections
    let mut corrections_to_apply = Vec::new();

    for token in &error_tokens {
        let corrections = corrector.correct_token(token, &token.context);

        if let Some(best) = corrections.first() {
            if best.confidence >= 0.5 {
                corrections_to_apply.push(best.clone());
            }
        }
    }

    // Apply corrections from end to start
    corrections_to_apply.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

    let mut result = source.to_string();
    for correction in corrections_to_apply {
        result = format!(
            "{}{}{}",
            &result[..correction.start_byte],
            correction.replacement,
            &result[correction.end_byte..]
        );
    }

    result
}

let source = "def funtion(x):\n    retrun x + 1";
let fixed = correct_code(source);
println!("Fixed: {}", fixed);
// Output: def function(x):\n    return x + 1
```

## Debugging and Inspection

### Checking Configuration

```rust
let config = corrector.config();
println!("Lexical weight: {}", config.lexical_weight);
println!("Grammar weight: {}", config.grammar_weight);
println!("Semantic weight: {}", config.semantic_weight);
println!("Min confidence: {}", config.min_confidence);
```

### Tracing Correction Sources

```rust
for correction in corrections {
    match correction.source {
        CorrectionSource::Lexical => {
            println!("[Lexical] {} → {}", correction.original, correction.replacement);
        }
        CorrectionSource::Grammar => {
            println!("[Grammar] {} → {}", correction.original, correction.replacement);
        }
        CorrectionSource::Neural => {
            println!("[Semantic] {} → {}", correction.original, correction.replacement);
        }
        CorrectionSource::Combined => {
            println!("[Multi-source] {} → {} ({})",
                correction.original,
                correction.replacement,
                correction.context.as_deref().unwrap_or("")
            );
        }
        _ => {}
    }
}
```

## Performance

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Token correction | O(L + G + S) | Sum of sub-corrector times |
| Merge corrections | O(n log n) | Grouping and sorting |
| Full analysis | O(n + e) | n = nodes, e = CPG edges |

Where:
- L = lexical correction time
- G = grammar correction time
- S = semantic correction time

### Optimization Tips

1. **Use lexical-only for speed**: `EnsembleCorrector::lexical_only()`
2. **Disable unused correctors**: `.without_semantic()` if not needed
3. **Adjust weights**: Higher weight = more influence on ranking
4. **Set appropriate thresholds**: Higher `min_confidence` = fewer results

## Thread Safety

`EnsembleCorrector` is `Send + Sync` when its language type is:

```rust
use std::sync::Arc;
use rayon::prelude::*;

let corrector = Arc::new(EnsembleCorrector::with_defaults(python, None));

let results: Vec<_> = tokens.par_iter()
    .map(|token| {
        let corrector = Arc::clone(&corrector);
        corrector.correct_token(token, &token.context)
    })
    .collect();
```

Note: Adding identifiers or registering variables requires mutable access.

## See Also

- [Correctors Overview](overview.md) - Architecture and comparison
- [Lexical Corrector](lexical.md) - Fuzzy matching
- [Grammar Corrector](grammar.md) - PCFG-based correction
- [Semantic Corrector](semantic.md) - CPG/GNN analysis
- [Pipeline](../pipeline.md) - End-to-end workflow
- [Correction Framework](../correction.md) - Base types
