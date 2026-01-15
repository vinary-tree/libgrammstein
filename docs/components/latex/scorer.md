# Combined LaTeX Scorer

The combined scorer integrates n-gram models, neural rescoring, and RAG validation into a unified scoring interface for the LaTeX correction pipeline.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Combined LaTeX Scorer                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Correction candidates                               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │           Parallel Scoring              │               │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │               │
│  │  │ N-gram  │ │ Neural  │ │   RAG   │   │               │
│  │  │ Score   │ │ Score   │ │ Boost   │   │               │
│  │  └────┬────┘ └────┬────┘ └────┬────┘   │               │
│  │       │           │           │         │               │
│  └───────┼───────────┼───────────┼─────────┘               │
│          │           │           │                          │
│          └───────────┼───────────┘                          │
│                      │                                      │
│                      ▼                                      │
│             ┌─────────────────┐                             │
│             │ Weight Combiner │                             │
│             └────────┬────────┘                             │
│                      │                                      │
│                      ▼                                      │
│              Final Ranked List                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

```rust
pub struct LaTeXScorerConfig {
    /// N-gram model configuration
    pub ngram: NgramScoringConfig,
    /// Neural rescoring configuration
    pub neural: Option<NeuralScoringConfig>,
    /// RAG validation configuration
    pub rag: Option<RagValidationConfig>,
    /// Score combination weights
    pub weights: ScoringWeights,
}

pub struct ScoringWeights {
    /// N-gram score weight
    pub ngram: f64,
    /// Neural score weight
    pub neural: f64,
    /// RAG validation boost
    pub rag_boost: f64,
}
```

## Basic Usage

```rust
use libgrammstein::latex::{LaTeXScorer, LaTeXScorerConfig, ScoringWeights};

let config = LaTeXScorerConfig {
    ngram: NgramScoringConfig {
        model_path: "models/latex.5gram".into(),
        order: 5,
    },
    neural: Some(NeuralScoringConfig {
        model_path: "models/modernbert".into(),
        weight: 0.3,
    }),
    rag: Some(RagValidationConfig {
        index_path: "indices/equations.bin".into(),
        boost: 0.1,
    }),
    weights: ScoringWeights {
        ngram: 0.6,
        neural: 0.3,
        rag_boost: 0.1,
    },
};

let scorer = LaTeXScorer::new(config)?;
```

## Scoring Candidates

```rust
let candidates = vec![
    r"\begin{equation}",
    r"\bgegin{equation}",
    r"\begin{equaton}",
];

let scores = scorer.score_candidates(&candidates)?;

// Sort by score
let mut ranked: Vec<_> = candidates.iter().zip(scores.iter()).collect();
ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

for (candidate, score) in ranked {
    println!("{:.4}: {}", score, candidate);
}
```

## Score Components

```rust
pub struct ScoreBreakdown {
    /// N-gram log probability
    pub ngram_score: f64,
    /// Neural pseudo-perplexity (inverted)
    pub neural_score: Option<f64>,
    /// RAG similarity boost
    pub rag_boost: Option<f64>,
    /// Final combined score
    pub final_score: f64,
}

let breakdown = scorer.score_with_breakdown(candidate)?;
println!("N-gram: {:.4}", breakdown.ngram_score);
println!("Neural: {:.4}", breakdown.neural_score.unwrap_or(0.0));
println!("RAG boost: {:.4}", breakdown.rag_boost.unwrap_or(0.0));
println!("Final: {:.4}", breakdown.final_score);
```

## Score Combination

The final score is computed as:

```
final_score = w_n × ngram_score + w_r × neural_score + boost × rag_match
```

Where:
- `w_n` = ngram weight (default 0.6)
- `w_r` = neural weight (default 0.3)
- `boost` = RAG boost if similar equation found (default 0.1)
- `rag_match` = 1 if RAG finds similar equation with score > threshold

## N-gram Scoring

```rust
pub struct NgramScoringConfig {
    /// Path to n-gram model
    pub model_path: PathBuf,
    /// N-gram order
    pub order: usize,
    /// Mode-specific models
    pub mode_models: Option<ModeModels>,
}

pub struct ModeModels {
    pub text_model: PathBuf,
    pub math_model: PathBuf,
    pub command_model: PathBuf,
}
```

## Neural Scoring

```rust
pub struct NeuralScoringConfig {
    /// Path to ModernBERT model
    pub model_path: PathBuf,
    /// Device (CPU/GPU)
    pub device: Device,
    /// Score weight
    pub weight: f64,
    /// Use pseudo-perplexity
    pub use_pseudo_ppl: bool,
}
```

## RAG Validation

```rust
pub struct RagValidationConfig {
    /// Path to equation index
    pub index_path: PathBuf,
    /// Minimum similarity for boost
    pub min_similarity: f32,
    /// Boost amount when match found
    pub boost: f64,
}
```

## Batch Scoring

Score multiple candidates efficiently:

```rust
let candidates: Vec<&str> = vec![/* many candidates */];

// Batch scoring (more efficient)
let scores = scorer.score_batch(&candidates)?;

// With parallel processing
let scores = scorer.score_batch_parallel(&candidates, num_threads)?;
```

## Caching

Enable caching for repeated scoring:

```rust
let config = LaTeXScorerConfig {
    cache: Some(CacheConfig {
        ngram_cache_size: 10000,
        neural_cache_size: 1000,
        rag_cache_size: 5000,
    }),
    ..default_config
};
```

## Streaming Scoring

For large candidate sets:

```rust
let candidates = candidate_generator.iter();

for (candidate, score) in scorer.score_streaming(candidates) {
    if score > threshold {
        accepted.push(candidate);
    }
}
```

## Weight Optimization

Optimize weights on a validation set:

```rust
use libgrammstein::latex::WeightOptimizer;

let optimizer = WeightOptimizer::new(scorer);

let validation_data: Vec<(String, String)> = vec![
    (r"\bgegin{equation}".into(), r"\begin{equation}".into()),
    // more (error, correction) pairs
];

let optimal_weights = optimizer.optimize(&validation_data)?;
scorer.set_weights(optimal_weights);
```

## Integration with Pipeline

```rust
use latex_corrector::{Corrector, CorrectorConfig};

// Create scorer
let latex_scorer = LaTeXScorer::new(scorer_config)?;

// Configure corrector with scorer
let corrector_config = CorrectorConfig {
    layers: LayerConfig {
        statistical: true,
        statistical_weight: 0.8,
        ..Default::default()
    },
    ..Default::default()
};

let mut corrector = Corrector::with_scorer(corrector_config, latex_scorer)?;
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| N-gram only | 0.5ms | Per candidate |
| N-gram + Neural | 15ms | Per candidate |
| Full (with RAG) | 20ms | Per candidate |
| Batch (100) | 200ms | Amortized |

## Memory Usage

| Component | Memory |
|-----------|--------|
| N-gram model | 500MB |
| Neural model | 1.5GB |
| RAG index | 1GB |
| Total | ~3GB |

## Error Handling

```rust
match scorer.score_candidates(&candidates) {
    Ok(scores) => {
        // Process scores
    }
    Err(ScoringError::NgramError(e)) => {
        // Fall back to neural only
        let scores = scorer.score_neural_only(&candidates)?;
    }
    Err(ScoringError::NeuralError(e)) => {
        // Fall back to n-gram only
        let scores = scorer.score_ngram_only(&candidates)?;
    }
    Err(e) => return Err(e.into()),
}
```

## Fallback Strategies

```rust
pub enum FallbackStrategy {
    /// Fail if any component fails
    Strict,
    /// Use available components
    Graceful,
    /// Fall back to n-gram only
    NgramOnly,
}

let config = LaTeXScorerConfig {
    fallback: FallbackStrategy::Graceful,
    ..default_config
};
```

## Score Normalization

Normalize scores for comparison:

```rust
impl LaTeXScorer {
    /// Normalize scores to [0, 1] range
    pub fn normalize_scores(&self, scores: &[f64]) -> Vec<f64> {
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range == 0.0 {
            vec![1.0; scores.len()]
        } else {
            scores.iter().map(|s| (s - min) / range).collect()
        }
    }
}
```

## Related

- [N-gram Models](./ngram.md): N-gram scoring details
- [Neural Rescorer](./rescorer.md): Neural scoring details
- [RAG](./rag.md): Equation retrieval
- [Overview](./overview.md): Module architecture
