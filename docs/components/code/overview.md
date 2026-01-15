# Code Module Overview

The code module provides a comprehensive framework for programming language modeling, syntactic analysis, and intelligent code correction in libgrammstein.

## What is the Code Module?

The code module enables error detection and correction in source code through a layered architecture combining:

- **Lexical correction**: Token-level fuzzy matching using liblevenshtein
- **Grammar correction**: PCFG-based structural validation with Earley parsing
- **Semantic correction**: GNN-powered analysis using Code Property Graphs

It supports multiple programming languages (Python, Rust, JavaScript, Rholang, MeTTa) with a pluggable language interface.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Code Module                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Source Code Input                           │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Tree-sitter (Incremental Parsing)                   │    │
│  │           ParsedCode with AST + ERROR nodes                      │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│         ┌───────────────────────┼───────────────────────┐               │
│         │                       │                       │               │
│         ▼                       ▼                       ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │    Lexical      │  │    Grammar      │  │      Semantic           │  │
│  │   Corrector     │  │   Corrector     │  │      Corrector          │  │
│  │                 │  │                 │  │                         │  │
│  │ • fuzzy match   │  │ • PCFG rules    │  │ • CPG analysis          │  │
│  │ • edit distance │  │ • Earley parse  │  │ • GNN scoring           │  │
│  │ • dictionaries  │  │ • completions   │  │ • embeddings            │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
│           │                    │                        │               │
│           └────────────────────┼────────────────────────┘               │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Ensemble Corrector                            │    │
│  │  • weighted combination    • deduplication    • agreement boost  │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Correction Pipeline                           │    │
│  │            Parse → Tokenize → Analyze → Correct → Rank          │    │
│  └──────────────────────────────┬──────────────────────────────────┘    │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   Ranked Corrections                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Description |
|-----------|-------------|
| `CodeLanguage` | Trait defining language-specific behavior (keywords, syntax, parsing) |
| `ParsedCode` | Tree-sitter parse result with error recovery |
| `CodePropertyGraph` | Unified AST + CFG + DFG representation |
| `WeightedCFG` | Probabilistic context-free grammar for structure |
| `Correction` | Single correction suggestion with confidence score |
| `CorrectionPipeline` | End-to-end orchestration of correction phases |

## Quick Start

```rust
use std::sync::Arc;
use libgrammstein::code::{
    CorrectionPipeline, PipelineConfig, Python,
    CodeCorrector, Correction,
};

// Create a Python language handler
let python = Arc::new(Python::new());

// Create a correction pipeline
let config = PipelineConfig::default();
let pipeline = CorrectionPipeline::new(python, config);

// Analyze code with errors
let source = r#"
def calcluate_total(items):
    retrun sum(items)
"#;

let result = pipeline.analyze(source)?;

// Print corrections
for correction in result.corrections {
    println!(
        "Line {}: {} -> {} (confidence: {:.2})",
        correction.start_byte,
        correction.original,
        correction.replacement,
        correction.confidence
    );
}
```

## Correction Layers

### Layer 1: Lexical Correction

Token-level spelling correction using liblevenshtein:

- **Keywords**: Correct `retrun` → `return`, `whlie` → `while`
- **Identifiers**: Suggest similar names from project corpus
- **Types**: Fix `stirng` → `string`, `boolen` → `boolean`

```rust
use libgrammstein::code::correctors::LexicalCorrector;

let mut corrector = LexicalCorrector::with_defaults(python.clone());
corrector.add_identifier("calculate_total");  // Learn from codebase

let corrections = corrector.correct_token(&token, &context);
```

### Layer 2: Grammar Correction

PCFG-based structural validation:

- Detect missing tokens (`;`, `)`, `}`)
- Suggest valid completions based on grammar rules
- Use Earley parsing for incremental validation

```rust
use libgrammstein::code::correctors::GrammarCorrector;
use libgrammstein::code::pcfg::WeightedCFG;

let grammar = WeightedCFG::from_corpus(&corpus)?;
let corrector = GrammarCorrector::with_defaults(python.clone(), grammar);

let corrections = corrector.correct_token(&token, &context);
```

### Layer 3: Semantic Correction

CPG and GNN-based semantic analysis:

- **Variable misuse**: Detect undefined or shadowed variables
- **Type errors**: Identify type mismatches
- **API misuse**: Flag incorrect API usage patterns

```rust
use libgrammstein::code::correctors::SemanticCorrector;

let corrector = SemanticCorrector::with_defaults(python.clone());

// Register known variables
corrector.register_variable("user_count".into(), Some("int".into()), 0);

let corrections = corrector.correct_token(&token, &context);
```

### Ensemble Combination

Combine all layers with configurable weights:

```rust
use libgrammstein::code::correctors::{EnsembleCorrector, EnsembleCorrectorConfig};

let config = EnsembleCorrectorConfig {
    lexical_weight: 0.4,
    grammar_weight: 0.35,
    semantic_weight: 0.25,
    min_confidence: 0.3,
    agreement_boost: true,
    ..Default::default()
};

let corrector = EnsembleCorrector::new(python.clone(), Some(grammar), config);
```

## Supported Languages

| Language | Feature Flag | Tree-sitter Grammar |
|----------|--------------|---------------------|
| Python | `code-python` | `tree-sitter-python` |
| Rust | `code-rust` | `tree-sitter-rust` |
| JavaScript | `code-javascript` | `tree-sitter-javascript` |
| Rholang | `code-rholang` | `rholang-tree-sitter` |
| MeTTa | `code-metta` | `tree-sitter-metta` |

## Feature Flags

Enable the code module with feature flags in `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["code", "code-python"] }
```

| Feature | Description |
|---------|-------------|
| `code` | Core code module (tree-sitter, petgraph) |
| `code-python` | Python language support |
| `code-rust` | Rust language support |
| `code-javascript` | JavaScript language support |
| `code-rholang` | Rholang (blockchain) support |
| `code-metta` | MeTTa (reasoning) support |
| `code-neural` | Neural embeddings (UniXcoder, GraphCodeBERT) |
| `code-mainstream` | All mainstream languages |
| `code-dsl` | All domain-specific languages |
| `code-full` | All languages + neural features |

## Integration with lling-llang

Export grammars to WFSTs for composition with lling-llang pipelines:

```rust
#[cfg(feature = "lling-llang-integration")]
use libgrammstein::code::{PcfgWfstConfig, PcfgWfstExport};

let config = PcfgWfstConfig {
    max_depth: 5,
    min_probability: 1e-10,
    ..Default::default()
};

let (wfst, vocabulary) = grammar.to_wfst::<TropicalWeight>(config);
```

## Thread Safety

All code module components support concurrent access:

- `CodeLanguage` implementations are `Send + Sync`
- Correctors use `&self` (immutable) API for thread-safe sharing
- `CorrectionPipeline` can be wrapped in `Arc` for multi-threaded use

```rust
use std::sync::Arc;
use std::thread;

let pipeline = Arc::new(CorrectionPipeline::new(python, config));

let handles: Vec<_> = sources.iter().map(|source| {
    let pipeline = Arc::clone(&pipeline);
    let source = source.clone();
    thread::spawn(move || pipeline.analyze(&source))
}).collect();
```

## Performance Considerations

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Parsing | O(n) | Incremental with tree-sitter |
| Lexical correction | O(k * d) | k = dictionary size, d = max edit distance |
| Grammar validation | O(n³) | Earley parser worst case |
| CPG construction | O(n + e) | n = nodes, e = edges |
| GNN scoring | O(L * n²) | L = layers, n = nodes |

For large codebases, consider:
- Incremental parsing for real-time analysis
- Caching embeddings for repeated queries
- Limiting correction scope to error regions

## See Also

- [Language](language.md) - CodeLanguage trait and TokenType system
- [Languages](languages.md) - Language implementations
- [AST](ast.md) - Tree-sitter integration
- [CPG](cpg.md) - Code Property Graphs
- [Correction](correction.md) - Correction types and framework
- [Correctors](correctors/overview.md) - Corrector implementations
- [Pipeline](pipeline.md) - End-to-end correction workflow
- [PCFG](pcfg.md) - Probabilistic context-free grammars
- [GNN](gnn.md) - Graph neural networks for code
