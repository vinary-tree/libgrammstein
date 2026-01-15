# Correctors Overview

The correctors module provides concrete implementations of the `CodeCorrector` trait, each specializing in different aspects of code correction.

## Architecture

The correction system uses a three-layer architecture:

```
                    ┌─────────────────────────────────────┐
                    │       EnsembleCorrector             │
                    │  (Aggregation & Ranking)            │
                    └─────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ LexicalCorrector │     │ GrammarCorrector │     │ SemanticCorrector│
│                  │     │                  │     │                  │
│  • Fuzzy match   │     │  • PCFG rules    │     │  • CPG analysis  │
│  • Edit distance │     │  • Earley parse  │     │  • GNN scoring   │
│  • Dictionaries  │     │  • Completions   │     │  • Data flow     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
    Spelling              Syntax errors            Semantic issues
    corrections           insertions/              variable misuse
                          deletions                type errors
```

## Corrector Types

| Corrector | Focus | Analysis Method | Source |
|-----------|-------|-----------------|--------|
| `LexicalCorrector` | Spelling | Levenshtein distance | `Lexical` |
| `GrammarCorrector` | Syntax | PCFG + Earley parsing | `Grammar` |
| `SemanticCorrector` | Semantics | CPG + GNN | `Neural`, `DataFlow` |
| `EnsembleCorrector` | Combined | Weighted aggregation | `Combined` |

## Layer Responsibilities

### Layer 1: Lexical

The lexical layer handles token-level spelling errors:

- **Input**: Individual tokens
- **Method**: Fuzzy matching against dictionaries
- **Output**: Alternative spellings within edit distance

```rust
// Example: "retrun" → "return" (edit distance 1)
let corrections = lexical_corrector.correct_token(&token, &context);
```

### Layer 2: Grammar

The grammar layer ensures syntactic validity:

- **Input**: Token sequences
- **Method**: PCFG validation and Earley parsing
- **Output**: Insertions, deletions, replacements

```rust
// Example: Missing ";" → insert ";"
let syntax_errors = grammar_corrector.find_syntax_errors(&tokens);
```

### Layer 3: Semantic

The semantic layer detects contextual issues:

- **Input**: Full AST/CPG
- **Method**: Data flow analysis, GNN scoring
- **Output**: Variable misuse, type errors, unused bindings

```rust
// Example: "count" used where "counter" was intended
let issues = semantic_corrector.analyze_cpg(&cpg);
```

## Correction Flow

The typical correction flow processes errors through each layer:

```
Source Code with Error
         │
         ▼
    ┌─────────┐
    │  Parse  │ ──► Tree-sitter AST
    └─────────┘
         │
         ▼
   ┌──────────┐
   │ Tokenize │ ──► Error tokens extracted
   └──────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
 Lexical  Grammar    Semantic
    │         │            │
    └────┬────┴────────────┘
         │
         ▼
   ┌──────────┐
   │ Ensemble │ ──► Merge, dedupe, rank
   └──────────┘
         │
         ▼
  Ranked Corrections
```

## Using Correctors

### Single Corrector

Use individual correctors for focused correction:

```rust
use libgrammstein::code::{LexicalCorrector, Python, CodeToken, TokenContext, TokenType};
use std::sync::Arc;

let python = Arc::new(Python::new());
let corrector = LexicalCorrector::with_defaults(python);

let token = CodeToken::new("pritn", 0, 1, 0, TokenType::Identifier, "identifier");
let context = TokenContext::new(TokenType::Identifier);

let corrections = corrector.correct_token(&token, &context);
for c in &corrections {
    println!("{} → {} ({:.2})", c.original, c.replacement, c.confidence);
}
```

### Ensemble Corrector

Use the ensemble for comprehensive correction:

```rust
use libgrammstein::code::{EnsembleCorrector, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());
let mut corrector = EnsembleCorrector::with_defaults(python, None);

// Add project-specific identifiers
corrector.add_identifiers(&["calculateTotal", "processData", "handleError"]);

// Register known variables for semantic analysis
corrector.register_variables(&[
    ("userCount".to_string(), Some("int".to_string())),
    ("userName".to_string(), Some("string".to_string())),
]);

let corrections = corrector.correct_token(&token, &context);
```

### Builder Pattern

Configure ensemble behavior precisely:

```rust
use libgrammstein::code::{EnsembleCorrectorBuilder, Python};
use std::sync::Arc;

let python = Arc::new(Python::new());

let corrector = EnsembleCorrectorBuilder::new(python)
    .lexical_weight(0.5)      // Prioritize spelling
    .grammar_weight(0.3)      // Balance syntax
    .semantic_weight(0.2)     // Lower semantic weight
    .without_grammar()        // Disable grammar (no PCFG)
    .build();
```

## Configuration Options

Each corrector has specific configuration:

### LexicalCorrectorConfig

| Option | Default | Description |
|--------|---------|-------------|
| `max_edit_distance` | 2 | Maximum Levenshtein distance |
| `min_token_length` | 2 | Skip tokens shorter than this |
| `max_candidates` | 5 | Maximum suggestions per token |
| `edit_penalty` | 0.15 | Confidence reduction per edit |

### GrammarCorrectorConfig

| Option | Default | Description |
|--------|---------|-------------|
| `max_candidates` | 5 | Maximum suggestions per error |
| `min_rule_probability` | 0.01 | Minimum rule probability |
| `suggest_insertions` | true | Suggest missing tokens |
| `suggest_deletions` | true | Suggest removing extra tokens |
| `max_lookahead` | 3 | Lookahead for completions |
| `base_confidence` | 0.8 | Base confidence score |

### SemanticCorrectorConfig

| Option | Default | Description |
|--------|---------|-------------|
| `min_confidence` | 0.5 | Threshold for reporting |
| `max_candidates` | 5 | Maximum suggestions per issue |
| `check_variable_misuse` | true | Detect wrong variables |
| `check_unused_bindings` | true | Detect unused variables |
| `check_type_errors` | true | Detect type mismatches |

### EnsembleCorrectorConfig

| Option | Default | Description |
|--------|---------|-------------|
| `lexical_weight` | 0.4 | Weight for lexical corrections |
| `grammar_weight` | 0.35 | Weight for grammar corrections |
| `semantic_weight` | 0.25 | Weight for semantic corrections |
| `min_confidence` | 0.3 | Minimum confidence to include |
| `max_candidates` | 10 | Maximum total results |
| `deduplicate` | true | Merge identical suggestions |
| `agreement_boost` | true | Boost when sources agree |
| `agreement_boost_factor` | 1.3 | Boost multiplier |

## Correction Sources

Each correction is tagged with its source:

```rust
pub enum CorrectionSource {
    Lexical,       // From fuzzy matching
    Grammar,       // From PCFG/Earley
    Neural,        // From GNN/embeddings
    TypeInference, // From type analysis
    ControlFlow,   // From CFG analysis
    DataFlow,      // From DFG analysis
    Combined,      // From ensemble agreement
    Unknown,       // Unspecified
}
```

Use the source to filter or debug corrections:

```rust
for correction in corrections {
    match correction.source {
        CorrectionSource::Lexical => println!("Spelling: {}", correction.replacement),
        CorrectionSource::Grammar => println!("Syntax: {}", correction.replacement),
        CorrectionSource::Neural => println!("Semantic: {}", correction.replacement),
        CorrectionSource::Combined => println!("Multi-source: {}", correction.replacement),
        _ => {}
    }
}
```

## When to Use Each Corrector

| Scenario | Recommended Corrector |
|----------|----------------------|
| Typos in keywords | `LexicalCorrector` |
| Missing semicolons/brackets | `GrammarCorrector` |
| Wrong variable names | `SemanticCorrector` |
| General code correction | `EnsembleCorrector` |
| IDE integration | `EnsembleCorrector` |
| Batch processing | `EnsembleCorrector` |
| Performance-critical | `LexicalCorrector` only |

## Thread Safety

All correctors implement `Send + Sync` when their language type does:

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

## Performance

| Corrector | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| `LexicalCorrector` | O(d × n) | O(v) |
| `GrammarCorrector` | O(n³) worst case | O(n²) |
| `SemanticCorrector` | O(n + e) | O(n) |
| `EnsembleCorrector` | Sum of above | Sum of above |

Where:
- d = max edit distance
- n = number of tokens/nodes
- e = number of edges in CPG
- v = vocabulary size

## See Also

- [Lexical Corrector](lexical.md) - Fuzzy matching details
- [Grammar Corrector](grammar.md) - PCFG-based correction
- [Semantic Corrector](semantic.md) - GNN/CPG analysis
- [Ensemble Corrector](ensemble.md) - Multi-source aggregation
- [Correction Framework](../correction.md) - Base types
- [Pipeline](../pipeline.md) - End-to-end workflow
