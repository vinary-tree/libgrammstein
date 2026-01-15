# LaTeX Scoring Module Overview

The LaTeX module provides statistical and neural scoring for LaTeX document correction. It integrates with the latex-corrector pipeline to provide language model-based ranking of correction candidates.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   LaTeX Scoring Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Candidate Correction Paths                              │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ LaTeX Tokenizer │ ← Mode-aware tokenization                  │
│  └────────┬────────┘                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │  Mode Detector  │ ← Text / Math / Command modes              │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────┴────────────────────────────┐                        │
│  │                                     │                        │
│  ▼                                     ▼                        │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ N-gram Scoring  │     │ Neural Scoring  │                    │
│  │ (per mode)      │     │ (ModernBERT)    │                    │
│  └────────┬────────┘     └────────┬────────┘                    │
│           │                       │                             │
│           └───────────┬───────────┘                             │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │ Score Combiner  │ ← Weighted combination         │
│              └────────┬────────┘                                │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │ Optional: RAG   │ ← Equation retrieval           │
│              │ Enhancement     │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│                       ▼                                         │
│              Scored Candidates                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. LaTeX Tokenizer

Mode-aware tokenization that recognizes:
- **Command mode**: `\alpha`, `\frac`, `\begin`
- **Math mode**: Content within `$...$`, `\[...\]`
- **Text mode**: Regular text content

```rust
use libgrammstein::latex::{LaTeXTokenizer, TokenMode};

let tokenizer = LaTeXTokenizer::new();
let tokens = tokenizer.tokenize(r"\begin{equation} \alpha + \beta \end{equation}");

for token in tokens {
    println!("{}: {:?}", token.text, token.mode);
}
// Output:
// \begin: Command
// {equation}: Delimiter
// \alpha: Math
// +: Math
// \beta: Math
// \end: Command
// {equation}: Delimiter
```

### 2. Mode-Aware N-gram Model

Separate n-gram models for different LaTeX contexts:

```rust
use libgrammstein::latex::{LaTeXNgramModel, NgramConfig};

let config = NgramConfig {
    command_model_path: "models/latex_commands.ngram",
    math_model_path: "models/latex_math.ngram",
    text_model_path: "models/latex_text.ngram",
    order: 5,
};

let model = LaTeXNgramModel::load(&config)?;
let score = model.score(&tokens)?;
```

### 3. Neural Rescoring

ModernBERT-based rescoring for context-aware correction:

```rust
use libgrammstein::neural::{ModernBertRescorer, RescoringConfig};

let config = RescoringConfig {
    ngram_weight: 0.7,
    neural_weight: 0.3,
    top_k: 100,
    ..Default::default()
};

let rescorer = ModernBertRescorer::new(config)?;
let scored_paths = rescorer.rescore_paths(candidates)?;
```

### 4. Equation RAG

Retrieve similar equations for reference and validation:

```rust
use libgrammstein::rag::{RagIndex, ExactCosineBackend, Retriever};

let index = RagIndex::<ExactCosineBackend>::load("equation_index.bin")?;
let retriever = Retriever::new(index, embedder, RetrievalConfig::default());

let similar = retriever.query(r"\sum_{i=1}^n x_i")?;
```

## Integration with latex-corrector

The LaTeX scoring module integrates as Layer 4 in the correction pipeline:

```rust
use latex_corrector::{Corrector, CorrectorConfig, LayerConfig};
use libgrammstein::latex::LaTeXLanguageModelLayer;

// Configure with statistical scoring
let config = CorrectorConfig {
    layers: LayerConfig {
        statistical: true,
        statistical_weight: 0.8,
        ..Default::default()
    },
    ..Default::default()
};

let mut corrector = Corrector::with_config(config)?;
```

## Scoring Formula

The final score for a candidate path combines multiple sources:

```
final_score = α × lexical_score + β × syntax_score + γ × semantic_score
            + δ × ngram_score + ε × neural_score
```

Where:
- `lexical_score`: Edit distance from liblevenshtein
- `syntax_score`: CFG parse validity from lling-llang
- `semantic_score`: Type checking score from lling-llang
- `ngram_score`: N-gram probability (mode-weighted)
- `neural_score`: ModernBERT pseudo-perplexity

Default weights:
| Component | Weight (δ, ε) |
|-----------|--------------|
| N-gram | 0.8 |
| Neural | 0.3 |

## Training Data

The LaTeX models are trained on arXiv corpus:

| Model | Training Data | Size |
|-------|--------------|------|
| Command n-grams | 1M+ papers | ~500MB |
| Math n-grams | 1M+ papers | ~800MB |
| Text n-grams | 1M+ papers | ~400MB |
| Embeddings | 100K papers | ~2GB |

## Feature Flags

Enable LaTeX features in `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["latex", "neural"] }
```

Available features:
- `latex`: Basic LaTeX tokenization and n-gram scoring
- `neural`: ModernBERT rescoring
- `rag`: Equation retrieval
- `rag-hnsw`: Approximate nearest neighbor (large indices)

## Performance

| Operation | Time (avg) | Notes |
|-----------|-----------|-------|
| Tokenization | 0.1ms | Per document |
| N-gram scoring | 0.5ms | 5-gram model |
| Neural scoring | 10ms | Per candidate |
| RAG query | 5ms | 1000 equations |

## Related Documentation

- [Tokenizer](./tokenizer.md): LaTeX-aware tokenization
- [N-gram Models](./ngram.md): Mode-aware scoring
- [Embeddings](./embedding.md): Command and equation embeddings
- [Neural Rescorer](./rescorer.md): ModernBERT integration
- [RAG](./rag.md): Equation retrieval
- [Combined Scorer](./scorer.md): Score combination strategies
