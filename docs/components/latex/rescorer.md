# Neural Rescorer

The neural rescorer uses ModernBERT to rerank correction candidates from the n-gram beam search, providing context-aware scoring that captures long-range dependencies.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Rescoring Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Top-K candidates from n-gram search                 │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Batch Encoding  │ ← Tokenize all candidates              │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ ModernBERT      │ ← Forward pass                         │
│  │ Inference       │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Pseudo-PPL      │ ← MLM-based scoring                    │
│  │ Scoring         │                                        │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Score Combine   │ ← α×ngram + β×neural                   │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  Reranked Candidates                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Basic Usage

```rust
use libgrammstein::neural::{ModernBertRescorer, RescoringConfig, ScoredPath};

let config = RescoringConfig {
    ngram_weight: 0.7,
    neural_weight: 0.3,
    top_k: 100,
    batch_size: 32,
    use_pseudo_perplexity: true,
    ..Default::default()
};

let rescorer = ModernBertRescorer::new(config)?;
```

## Rescoring Candidates

```rust
// Create scored paths from beam search
let candidates: Vec<ScoredPath<f64>> = vec![
    ScoredPath::new(
        vec!["\\begin".to_string(), "{equation}".to_string()],
        -2.5  // n-gram log probability
    ),
    ScoredPath::new(
        vec!["\\bgegin".to_string(), "{equation}".to_string()],
        -4.2
    ),
];

// Rescore with neural model
let rescored = rescorer.rescore_paths(candidates)?;

// Best candidate is now first
println!("Best: {}", rescored[0].text());
println!("Score: {:.4}", rescored[0].final_score);
```

## Scoring Methods

### Pseudo-Perplexity (MLM)

Mask each token and compute prediction probability:

```rust
let config = RescoringConfig {
    use_pseudo_perplexity: true,
    ..Default::default()
};

let rescorer = ModernBertRescorer::new(config)?;
let score = rescorer.score_sentence(r"\begin{equation} x = 1 \end{equation}")?;
// Lower score = more probable
```

### Embedding Coherence

Alternative scoring based on embedding similarity:

```rust
let config = RescoringConfig {
    use_pseudo_perplexity: false,  // Use coherence instead
    ..Default::default()
};

let rescorer = ModernBertRescorer::new(config)?;
```

## Configuration

```rust
pub struct RescoringConfig {
    /// Model configuration
    pub model_config: ModernBertConfig,
    /// Weight for n-gram scores (alpha)
    pub ngram_weight: f64,
    /// Weight for neural scores (beta)
    pub neural_weight: f64,
    /// Number of top paths to rescore
    pub top_k: usize,
    /// Batch size for parallel rescoring
    pub batch_size: usize,
    /// Use MLM pseudo-perplexity
    pub use_pseudo_perplexity: bool,
}
```

### Weight Tuning

| Use Case | ngram_weight | neural_weight |
|----------|--------------|---------------|
| Speed priority | 0.9 | 0.1 |
| Balanced | 0.7 | 0.3 |
| Accuracy priority | 0.5 | 0.5 |
| Neural heavy | 0.3 | 0.7 |

## ScoredPath Structure

```rust
pub struct ScoredPath<W> {
    /// The path (sequence of tokens)
    pub tokens: Vec<String>,
    /// Original n-gram score
    pub ngram_score: W,
    /// Neural score (after rescoring)
    pub neural_score: Option<f64>,
    /// Combined final score
    pub final_score: f64,
}

impl<W: Clone + Into<f64>> ScoredPath<W> {
    /// Get text representation
    pub fn text(&self) -> String {
        self.tokens.join(" ")
    }
}
```

## Batch Scoring

Score multiple sentences efficiently:

```rust
let sentences = vec![
    r"\begin{equation} x = 1 \end{equation}",
    r"\bgegin{equation} x = 1 \end{equation}",
    r"\begin{equaton} x = 1 \end{equation}",
];

let scores = rescorer.score_batch(&sentences)?;

for (sentence, score) in sentences.iter().zip(scores.iter()) {
    println!("{:.4}: {}", score, sentence);
}
```

## Score Combination

Final score combines n-gram and neural:

```
final_score = α × ngram_score + β × neural_normalized

Where:
  α = ngram_weight (default 0.7)
  β = neural_weight (default 0.3)
  neural_normalized = 1 / (1 + perplexity)
```

## Dynamic Weight Adjustment

```rust
let mut rescorer = ModernBertRescorer::new(config)?;

// Adjust weights based on context
if document_is_math_heavy {
    rescorer.set_weights(0.6, 0.4);  // More neural for math
} else {
    rescorer.set_weights(0.8, 0.2);  // More n-gram for text
}
```

## Integration with Beam Search

```rust
use libgrammstein::ngram::BeamSearch;

// Run beam search with n-gram model
let beam = BeamSearch::new(&ngram_model, BeamConfig {
    beam_width: 100,
    max_length: 50,
});

let candidates = beam.search(&prefix)?;

// Convert to ScoredPath
let scored_paths: Vec<ScoredPath<f64>> = candidates
    .into_iter()
    .map(|(tokens, score)| ScoredPath::new(tokens, score))
    .collect();

// Rescore with neural model
let final_candidates = rescorer.rescore_paths(scored_paths)?;
```

## Model Sharing

Share model between embedder and rescorer:

```rust
use std::sync::Arc;
use libgrammstein::neural::{ModernBertModel, ModernBertEmbedder, ModernBertRescorer};

// Load model once
let model = Arc::new(ModernBertModel::load(config.clone())?);

// Share between components
let embedder = ModernBertEmbedder::from_model(Arc::clone(&model), emb_config);
let rescorer = ModernBertRescorer::from_model(Arc::clone(&model), rescore_config);
```

## Detailed Results

Get detailed scoring information:

```rust
pub struct RescoringResult {
    /// Best path after rescoring
    pub best_path: String,
    /// Top-k paths with scores
    pub top_paths: Vec<RankedPath>,
    /// Total paths considered
    pub total_paths: usize,
}

pub struct RankedPath {
    pub text: String,
    pub rank: usize,
    pub ngram_score: f64,
    pub neural_score: f64,
    pub final_score: f64,
}
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single sentence | 10ms | GPU |
| Batch (32) | 50ms | GPU |
| Top-100 rescore | 200ms | GPU |

### GPU Memory

| Batch Size | Memory |
|------------|--------|
| 1 | 1.5GB |
| 8 | 2GB |
| 32 | 4GB |
| 64 | 7GB |

## Error Handling

```rust
match rescorer.score_sentence(input) {
    Ok(score) => println!("Score: {}", score),
    Err(NeuralError::Tokenization(msg)) => {
        eprintln!("Tokenization failed: {}", msg);
    }
    Err(NeuralError::Inference(msg)) => {
        eprintln!("Inference failed: {}", msg);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## CPU Fallback

```rust
use libgrammstein::neural::Device;

let config = RescoringConfig {
    model_config: ModernBertConfig {
        device: Device::Cpu,  // Use CPU if no GPU
        ..Default::default()
    },
    ..Default::default()
};
```

## Related

- [Embeddings](./embedding.md): Document embeddings
- [N-gram Models](./ngram.md): N-gram scoring
- [Combined Scorer](./scorer.md): Score combination
