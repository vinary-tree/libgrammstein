# ModernBERT Rescorer

The `ModernBertRescorer` combines n-gram scores with neural language model scores to improve beam search output quality.

## What is Neural Rescoring?

Beam search from n-gram models produces multiple candidate paths. Neural rescoring re-ranks these paths using a neural language model:

```
N-gram Beam Search                     Neural Rescorer
┌────────────────────┐                ┌────────────────────┐
│ Path 1: "the cat"  │  score: -2.3   │ Combined: -1.8     │ → Rank 2
│ Path 2: "teh cat"  │  score: -2.1   │ Combined: -3.5     │ → Rank 3
│ Path 3: "the cats" │  score: -2.5   │ Combined: -1.5     │ → Rank 1 ✓
└────────────────────┘                └────────────────────┘

Final Score = α × ngram_score + β × neural_score
```

The neural model captures:
- **Semantic coherence**: Does the sentence make sense?
- **Long-range dependencies**: Context beyond n-gram window
- **Fluency**: Natural language patterns

## Scoring Methods

### Pseudo-Perplexity (MLM-based)

Uses masked language modeling to score each token:

```
Input: "The quick brown fox"

For each position:
  Mask: "The [MASK] brown fox" → P(quick) = 0.85
  Mask: "The quick [MASK] fox" → P(brown) = 0.72
  Mask: "The quick brown [MASK]" → P(fox) = 0.91

Score = geometric_mean([0.85, 0.72, 0.91]) = 0.82
```

### Embedding Coherence

Uses embedding similarity between full and partial sentences:

```
Full:    embed("The quick brown fox")      → v_full
Partial: embed("The quick brown")          → v_partial

Coherence = cosine_similarity(v_full, v_partial)
```

This measures how well the sentence "flows" when completed.

## Configuration

```rust
use libgrammstein::neural::{RescoringConfig, ModernBertConfig};

let config = RescoringConfig {
    // Model configuration
    model_config: ModernBertConfig::default(),

    // Weight for n-gram score (α)
    ngram_weight: 0.3,

    // Weight for neural score (β)
    neural_weight: 0.7,

    // Number of top paths to rescore
    top_k: 10,

    // Batch size for scoring
    batch_size: 8,

    // Use pseudo-perplexity (true) or embedding coherence (false)
    use_pseudo_perplexity: true,
};
```

### Weight Selection Guidelines

| Use Case | ngram_weight | neural_weight |
|----------|--------------|---------------|
| Domain-specific (medical, legal) | 0.5 | 0.5 |
| General text | 0.3 | 0.7 |
| Fluency-focused | 0.2 | 0.8 |
| Speed-focused | 0.7 | 0.3 |

## Creating a Rescorer

### From Configuration

```rust
use libgrammstein::neural::{ModernBertRescorer, RescoringConfig};

let config = RescoringConfig::default();
let rescorer = ModernBertRescorer::new(config)?;
```

### From Existing Model

```rust
use std::sync::Arc;
use libgrammstein::neural::{ModernBertModel, ModernBertRescorer, RescoringConfig};

let model = Arc::new(ModernBertModel::load(&model_config)?);
let rescorer = ModernBertRescorer::from_model(model, RescoringConfig::default());
```

## Scoring Sentences

### Single Sentence

```rust
let score = rescorer.score_sentence("The quick brown fox jumps")?;
println!("Neural score: {:.4}", score);
```

### Batch Scoring

```rust
let sentences = vec![
    "The quick brown fox",
    "The quik brown fox",  // Typo
    "The fast brown fox",
];

let scores = rescorer.score_batch(&sentences)?;
for (sent, score) in sentences.iter().zip(scores.iter()) {
    println!("{}: {:.4}", sent, score);
}
```

## Rescoring Beam Search Paths

### ScoredPath Structure

```rust
use libgrammstein::neural::ScoredPath;

// Create paths from beam search output
let paths: Vec<ScoredPath<f32>> = vec![
    ScoredPath::new(
        vec!["the".to_string(), "quick".to_string(), "fox".to_string()],
        -2.3,  // n-gram log probability
    ),
    ScoredPath::new(
        vec!["the".to_string(), "fast".to_string(), "fox".to_string()],
        -2.5,
    ),
];
```

### Rescoring

```rust
let result = rescorer.rescore_paths(&paths)?;

println!("Best path: {}", result.best_path.text());
println!("  N-gram score: {:.4}", result.best_path.ngram_score);
println!("  Neural score: {:.4}", result.best_path.neural_score);
println!("  Final score:  {:.4}", result.best_path.final_score);

println!("\nTop {} paths:", result.top_k_paths.len());
for path in &result.top_k_paths {
    println!("  {}: {:.4}", path.text(), path.final_score);
}
```

### RescoringResult Structure

```rust
pub struct RescoringResult {
    /// Best path after rescoring
    pub best_path: ScoredPath<f32>,

    /// Top-k paths sorted by final score
    pub top_k_paths: Vec<ScoredPath<f32>>,

    /// Total number of paths considered
    pub total_paths_considered: usize,
}
```

## Dynamic Weight Adjustment

Update weights at runtime:

```rust
// Start with default weights
let mut rescorer = ModernBertRescorer::new(config)?;

// Increase neural weight for fluency
rescorer.set_weights(0.2, 0.8);

// Revert to balanced
rescorer.set_weights(0.5, 0.5);
```

## Combined Scoring Formula

The final score combines n-gram and neural scores:

```
final_score = α × ngram_score + β × neural_score

where:
  α = ngram_weight (default: 0.3)
  β = neural_weight (default: 0.7)
```

For log-domain scores (typical for n-grams):

```
final_score = α × log_ngram + β × log_neural
            = log(P_ngram^α × P_neural^β)
```

## Integration with N-gram Models

```rust
use libgrammstein::ngram::NgramModel;
use libgrammstein::neural::{ModernBertRescorer, ScoredPath};

// Score with n-gram model
let ngram_model = NgramModel::load("model.bin")?;
let candidates = vec!["the quick fox", "the quik fox", "teh quick fox"];

let paths: Vec<ScoredPath<f32>> = candidates
    .iter()
    .map(|text| {
        let tokens: Vec<_> = text.split_whitespace().collect();
        let ngram_score = ngram_model.sentence_log_prob(&tokens);
        ScoredPath::new(
            tokens.iter().map(|s| s.to_string()).collect(),
            ngram_score,
        )
    })
    .collect();

// Rescore with neural model
let rescorer = ModernBertRescorer::new(config)?;
let result = rescorer.rescore_paths(&paths)?;

println!("Best correction: {}", result.best_path.text());
```

## RankedPath for Display

For detailed results with rankings:

```rust
pub struct RankedPath {
    pub text: String,
    pub rank: usize,
    pub ngram_score: f32,
    pub neural_score: f32,
    pub final_score: f32,
}
```

## Performance Considerations

### Batch Size

Larger batches are more efficient but use more memory:

```rust
let config = RescoringConfig {
    batch_size: 16,  // Increase for GPU
    ..Default::default()
};
```

### Top-k Filtering

Only rescore the most promising paths:

```rust
let config = RescoringConfig {
    top_k: 5,  // Only rescore top 5 n-gram paths
    ..Default::default()
};
```

### Pseudo-Perplexity vs Embedding Coherence

| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| Pseudo-perplexity | Slower | Higher | Higher |
| Embedding coherence | Faster | Good | Lower |

```rust
// Fast mode
let config = RescoringConfig {
    use_pseudo_perplexity: false,
    ..Default::default()
};
```

## Error Handling

```rust
use libgrammstein::neural::NeuralError;

match rescorer.rescore_paths(&paths) {
    Ok(result) => {
        println!("Best: {}", result.best_path.text());
    }
    Err(NeuralError::Inference(msg)) => {
        eprintln!("Scoring failed: {}", msg);
        // Fall back to n-gram ranking
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## See Also

- [Overview](overview.md) - Neural module introduction
- [Model](model.md) - ModernBERT model details
- [N-gram Model](../ngram/overview.md) - N-gram scoring
- [ASR Cascade](../../integration/asr-cascade.md) - Full ASR pipeline
