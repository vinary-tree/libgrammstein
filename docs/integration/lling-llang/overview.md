# lling-llang Integration

This document explains how libgrammstein integrates with [lling-llang](https://github.com/f1r3fly-io/lling-llang), the WFST (Weighted Finite State Transducer) framework for text correction and normalization.

## Architecture Overview

libgrammstein and lling-llang are designed to work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              lling-llang                                     │
│                        (WFST Framework)                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LayerPipeline                                   │   │
│  │                                                                     │   │
│  │   Input Lattice                                                     │   │
│  │        │                                                            │   │
│  │        ▼                                                            │   │
│  │   ┌─────────────────────────────────┐                               │   │
│  │   │   SpellingCorrectionLayer       │◄── liblevenshtein             │   │
│  │   │   (fuzzy matching candidates)   │                               │   │
│  │   └─────────────┬───────────────────┘                               │   │
│  │                 ▼                                                    │   │
│  │   ┌─────────────────────────────────┐                               │   │
│  │   │   CfgFilterLayer                │                               │   │
│  │   │   (grammar validation)          │                               │   │
│  │   └─────────────┬───────────────────┘                               │   │
│  │                 ▼                                                    │   │
│  │   ┌─────────────────────────────────┐                               │   │
│  │   │   LanguageModelLayer            │◄── lling-llang wrapper        │   │
│  │   │   (rescoring with LM)           │                               │   │
│  │   │                                 │                               │   │
│  │   │   ┌─────────────────────────┐   │                               │   │
│  │   │   │   LanguageModel trait   │◄──┼── libgrammstein implements    │   │
│  │   │   │   - score_sequence()    │   │                               │   │
│  │   │   │   - score_continuation()│   │                               │   │
│  │   │   └─────────────────────────┘   │                               │   │
│  │   └─────────────┬───────────────────┘                               │   │
│  │                 ▼                                                    │   │
│  │   Output Lattice (reweighted)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ implements
                              │ LanguageModel trait
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            libgrammstein                                     │
│                     (Language Model Library)                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HybridLanguageModel                               │   │
│  │                                                                     │   │
│  │   ┌─────────────────────┐    ┌─────────────────────┐               │   │
│  │   │     NgramModel      │    │  SubwordEmbedding   │               │   │
│  │   │  Modified Kneser-Ney│    │   FastText-style    │               │   │
│  │   └─────────────────────┘    └─────────────────────┘               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The LanguageModel Trait

lling-llang defines a `LanguageModel` trait for pluggable language models:

```rust
// Defined in lling-llang/src/layers/lm_rerank.rs

/// Language model trait for pluggable LM integration.
///
/// This trait is designed to be simple and decoupled from
/// lattice internals. It uses strings, not VocabIds, so
/// implementations don't need lattice knowledge.
pub trait LanguageModel: Send + Sync {
    /// Score a complete token sequence.
    ///
    /// Returns the log probability: log P(w₁, w₂, ..., wₙ)
    fn score_sequence(&self, tokens: &[&str]) -> f64;

    /// Score a continuation given a prefix.
    ///
    /// Returns the log probability: log P(next | prefix)
    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64;
}
```

### Design Decisions

The trait uses `&[&str]` (strings) rather than vocabulary IDs:

| Approach | Pros | Cons |
|----------|------|------|
| VocabId | Faster lookup, shared vocabulary | Tight coupling, translation needed |
| **Strings** | **Decoupled, simple interface** | Slight overhead for string handling |

libgrammstein doesn't need to know about lling-llang's lattice vocabulary or internal structure. This separation makes both libraries independently testable and maintainable.

## libgrammstein Implementation

### HybridLanguageModel implements LanguageModel

```rust
// In libgrammstein/src/integration/lling_llang.rs

#[cfg(feature = "lling-llang-integration")]
use lling_llang::layers::LanguageModel;

impl<D> LanguageModel for HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    fn score_sequence(&self, tokens: &[&str]) -> f64 {
        self.sentence_log_prob(tokens)
    }

    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64 {
        self.score(next, prefix)
    }
}
```

### NgramModel implements LanguageModel

For N-gram-only scoring:

```rust
impl<D> LanguageModel for NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    fn score_sequence(&self, tokens: &[&str]) -> f64 {
        self.sentence_log_prob(tokens)
    }

    fn score_continuation(&self, prefix: &[&str], next: &str) -> f64 {
        self.log_prob(next, prefix)
    }
}
```

## Thread Safety

lling-llang requires `LanguageModel: Send + Sync` because:
- Lattice operations may be parallelized with Rayon
- Multiple threads may query the LM concurrently

libgrammstein models satisfy this:

| Component | Thread Safety Mechanism |
|-----------|------------------------|
| `HybridLanguageModel<D>` | `D: Send + Sync`, `Arc` wrappers |
| `NgramModel<D>` | `Arc<D>` where `D: Send + Sync` |
| `SubwordEmbedding` | Immutable + `Arc<DashMap>` cache |
| Score cache | `Mutex<LruCache>` |

### Concurrent Access Pattern

```rust
use std::sync::Arc;
use rayon::prelude::*;

// Create shared model
let model: Arc<HybridLanguageModel<_>> = Arc::new(load_model()?);

// Process lattice paths in parallel
let paths: Vec<LatticePath> = ...;

let scored_paths: Vec<_> = paths
    .par_iter()
    .map(|path| {
        let tokens: Vec<_> = path.labels.iter().map(|s| s.as_str()).collect();
        let score = model.score_sequence(&tokens);
        (path.clone(), score)
    })
    .collect();
```

## Integration with LanguageModelLayer

lling-llang's `LanguageModelLayer` wraps any `LanguageModel`:

```rust
// In lling-llang/src/layers/lm_rerank.rs

pub struct LanguageModelLayer {
    lm: Box<dyn LanguageModel>,
    weight: f64,
}

impl<W: Semiring, B: LatticeBackend> CorrectionLayer<W, B> for LanguageModelLayer {
    fn name(&self) -> &str {
        "language-model"
    }

    fn apply(&self, lattice: &Lattice<W, B>) -> Result<Lattice<W, B>, LayerError> {
        // For each edge in the lattice, add LM score to weight
        let mut builder = LatticeBuilder::new(lattice.backend().clone());

        for edge in lattice.edges() {
            // Get context from path to this edge
            let context = self.get_context(lattice, &edge);
            let token = lattice.backend().id_to_string(edge.label);

            // Score with language model
            let lm_score = self.lm.score_continuation(&context, &token);

            // Combine with existing weight
            let new_weight = edge.weight.times(&W::from_log_prob(lm_score * self.weight));

            builder.add_edge(Edge {
                weight: new_weight,
                ..edge.clone()
            });
        }

        Ok(builder.build(lattice.num_nodes()))
    }

    fn estimated_reduction(&self) -> f64 {
        1.0  // Doesn't reduce, only reweights
    }
}
```

## Usage Examples

### Basic Integration

```rust
use lling_llang::prelude::*;
use lling_llang::layers::{LayerPipelineBuilder, LanguageModelLayer};
use libgrammstein::HybridLanguageModel;

// Load libgrammstein model
let lm: HybridLanguageModel<_> = HybridLanguageModel::load("model.bin")?;

// Wrap in LanguageModelLayer
let lm_layer = LanguageModelLayer::new(Box::new(lm), 1.0);

// Build pipeline
let pipeline = LayerPipelineBuilder::new()
    .add_layer(SpellingCorrectionLayer::new(dictionary, 2))
    .add_layer(CfgFilterLayer::new(&grammar))
    .add_layer(lm_layer)
    .build();

// Apply to input
let input = tokenize("teh quikc brwon fox");
let lattice = tokens_to_lattice(&input);
let result = pipeline.apply(&lattice)?;

// Extract best path
let best = viterbi(&mut result);
println!("{}", best.to_string());  // "the quick brown fox"
```

### With Custom Weights

```rust
// Adjust LM weight (higher = more influence)
let lm_layer = LanguageModelLayer::new(Box::new(lm), 2.0);

// Weight affects score combination:
// final_weight = original_weight × (lm_score ^ lm_weight)
```

### N-gram Only

For faster, simpler scoring:

```rust
use libgrammstein::NgramModel;

let ngram: NgramModel<_> = NgramModel::load("ngram.bin")?;
let lm_layer = LanguageModelLayer::new(Box::new(ngram), 1.0);
```

## Weight Conversion

lling-llang uses semirings for lattice weights. The LM layer converts log probabilities:

```rust
// Log probability from libgrammstein
let log_prob: f64 = lm.score_continuation(&context, &token);

// Convert to semiring weight
match semiring_type {
    SemiringType::Tropical => {
        // Tropical: weight = -log_prob (lower is better)
        TropicalWeight::new(-log_prob * lm_weight)
    }
    SemiringType::LogSemiring => {
        // Log: weight = log_prob (higher is better, in log space)
        LogWeight::new(log_prob * lm_weight)
    }
    SemiringType::Probability => {
        // Probability: weight = exp(log_prob)
        ProbWeight::new((log_prob * lm_weight).exp())
    }
}
```

## Batch Scoring Optimization

For large lattices, batch scoring improves efficiency:

```rust
impl LanguageModelLayer {
    fn apply_batch(&self, lattice: &Lattice<W, B>) -> Result<Lattice<W, B>, LayerError> {
        // Collect all (context, token) pairs
        let queries: Vec<_> = lattice.edges()
            .map(|edge| {
                let context = self.get_context(lattice, &edge);
                let token = lattice.backend().id_to_string(edge.label);
                (edge, context, token)
            })
            .collect();

        // Score in parallel
        let scores: Vec<f64> = queries
            .par_iter()
            .map(|(_, context, token)| {
                self.lm.score_continuation(context, token)
            })
            .collect();

        // Apply scores to edges
        // ...
    }
}
```

## Feature Flag

Enable lling-llang integration with a feature flag:

```toml
# Cargo.toml
[dependencies]
libgrammstein = { version = "0.1", features = ["lling-llang-integration"] }
```

```rust
// Only available with feature flag
#[cfg(feature = "lling-llang-integration")]
impl<D> LanguageModel for HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    // ...
}
```

## Testing the Integration

```rust
#[cfg(test)]
#[cfg(feature = "lling-llang-integration")]
mod integration_tests {
    use super::*;
    use lling_llang::layers::LanguageModel;

    #[test]
    fn test_implements_trait() {
        let model = create_test_model();

        // Verify trait implementation
        let score = model.score_sequence(&["the", "quick", "brown", "fox"]);
        assert!(score.is_finite());
        assert!(score <= 0.0);  // Log probabilities are non-positive

        let cont_score = model.score_continuation(&["the", "quick"], "brown");
        assert!(cont_score.is_finite());
    }

    #[test]
    fn test_thread_safety() {
        let model = Arc::new(create_test_model());

        // Parallel scoring
        let scores: Vec<_> = (0..1000)
            .into_par_iter()
            .map(|_| model.score_sequence(&["test", "sequence"]))
            .collect();

        // All scores should be identical
        assert!(scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10));
    }
}
```

## Error Handling

libgrammstein's scoring methods don't return errors (they always produce a score). Edge cases:

| Scenario | Behavior |
|----------|----------|
| Empty sequence | Returns 0.0 (log(1) = 0) |
| All OOV words | Uses backoff + embeddings |
| Very long sequence | Normal scoring (no overflow in log space) |
| Empty context | Unigram probability |

```rust
impl<D> HybridLanguageModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    pub fn score_sequence(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;  // P(empty) = 1, log(1) = 0
        }

        // Normal scoring...
    }
}
```

## Performance Considerations

### Caching

The LRU cache prevents redundant computation:

```rust
// Common in lattice rescoring: same context, different continuations
// Context "the quick" appears many times
model.score_continuation(&["the", "quick"], "brown");  // Computed
model.score_continuation(&["the", "quick"], "red");    // Computed
model.score_continuation(&["the", "quick"], "slow");   // Computed
// Context embedding cached after first call
```

### Pre-warming

For known high-frequency contexts:

```rust
// Prewarm with common contexts from training data
model.prewarm_contexts(&[
    &["the"],
    &["of", "the"],
    &["in", "the"],
    &["to", "the"],
]);
```

## Next Steps

- [Pipeline Usage](pipeline-usage.md): Complete pipeline examples
- [PathMap Synergy](pathmap-synergy.md): Shared infrastructure
- [Hybrid Model](../../components/hybrid/overview.md): Model details
- [Architecture](../../architecture/overview.md): System design
