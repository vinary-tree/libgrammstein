# libgrammstein

A hybrid language model library combining N-gram models with subword embeddings for robust text scoring, correction, and generation.

## What is libgrammstein?

**libgrammstein** is a Rust library that provides statistical language models for predicting word sequences. It combines two complementary approaches:

1. **N-gram Models**: Learn word sequence probabilities from training text using Modified Kneser-Ney smoothing
2. **Subword Embeddings**: Learn dense vector representations of words and their character fragments (FastText-style)

The **hybrid model** combines both to achieve precise local context modeling with robust handling of rare and unseen words.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              libgrammstein                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Training Corpus                                                           │
│   (Wikipedia, Gutenberg, etc.)                                              │
│        │                                                                    │
│        ├──────────────────────────┬──────────────────────────┐              │
│        ▼                          ▼                          │              │
│   ┌─────────────────┐      ┌─────────────────┐               │              │
│   │  N-gram Model   │      │    Subword      │               │              │
│   │                 │      │   Embeddings    │               │              │
│   │  "the quick" →  │      │                 │               │              │
│   │   P(brown) = ?  │      │  word vectors + │               │              │
│   │                 │      │  char n-grams   │               │              │
│   │  Modified KN    │      │  (FastText)     │               │              │
│   │  smoothing      │      │                 │               │              │
│   └────────┬────────┘      └────────┬────────┘               │              │
│            │                        │                        │              │
│            └───────────┬────────────┘                        │              │
│                        ▼                                     │              │
│            ┌─────────────────────┐                           │              │
│            │  HybridLanguage     │                           │              │
│            │      Model          │                           │              │
│            │                     │                           │              │
│            │  score = λ₁×ngram   │                           │              │
│            │        + λ₂×embed   │                           │              │
│            └─────────────────────┘                           │              │
│                        │                                     │              │
│                        ▼                                     │              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        Applications                                  │  │
│   │  • Spelling correction scoring    • Text quality filtering          │  │
│   │  • Grammar checking               • WFST lattice rescoring          │  │
│   │  • Autocomplete ranking           • Text generation                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Hybrid Scoring**: Combines N-gram precision with embedding semantic coverage
- **Modified Kneser-Ney Smoothing**: State-of-the-art N-gram probability estimation
- **Subword Embeddings**: FastText-style character n-gram enrichment for OOV handling
- **Streaming Corpus Processing**: Train on large corpora (10GB+) without loading into memory
- **WFST Integration**: Implements lling-llang's `LanguageModel` trait for lattice rescoring
- **Efficient Storage**: Uses liblevenshtein trie dictionaries for compact N-gram storage
- **Thread-Safe**: All models implement `Send + Sync` for concurrent access
- **Serialization**: Save/load trained models with serde

## How It Works

### N-gram Language Models

An **N-gram model** predicts the probability of a word given its preceding context. Given the sentence "the quick brown fox", the model answers questions like:

- What is P(fox | the quick brown)?
- What is P(brown | the quick)?

The model learns these probabilities by counting word sequences in training text:

```
Training: "the quick brown fox the quick red fox"

Bigram counts:
  "the quick": 2
  "quick brown": 1
  "quick red": 1
  "brown fox": 1
  "red fox": 1

P(brown | quick) = count("quick brown") / count("quick") = 1/2 = 0.5
```

**The problem**: What if we see "quick purple"? It has count 0, giving P(purple | quick) = 0. This breaks probability calculations for any sentence containing unseen N-grams.

### Modified Kneser-Ney Smoothing

libgrammstein uses **Modified Kneser-Ney (MKN) smoothing**, the state-of-the-art technique for handling unseen N-grams. It works by:

1. **Discounting**: Subtract a small amount from observed counts
2. **Backoff**: Redistribute the "stolen" probability mass to lower-order models
3. **Continuation counts**: For lower orders, count how many contexts a word appears in (not raw frequency)

```
P_MKN(w | context) = [count(context w) - D]⁺ / count(context)
                   + γ(context) × P_MKN(w | shorter_context)

Where:
- D = discount (different values for count=1, count=2, count≥3)
- γ = backoff weight (redistributed probability mass)
- [x]⁺ = max(x, 0)
```

**Continuation counts** solve the "San Francisco" problem: "Francisco" has high raw frequency but only ever follows "San". Continuation counts measure versatility—"city" follows many words, so it gets higher lower-order probability than "Francisco".

### Subword Embeddings

**Word embeddings** represent words as dense vectors where similar words have similar vectors:

```
"cat"  → [0.23, -0.15, 0.89, ..., 0.42]   (100 dimensions)
"dog"  → [0.25, -0.12, 0.85, ..., 0.39]   (similar vector)
```

The problem: embeddings fail for words not seen during training ("splendiferous", "COVID-19").

**Subword embeddings** solve this by representing words as the sum of their character n-grams:

```
"running" = embed("running") + embed("<ru") + embed("run") + embed("unn") + ...

For unseen "fastly":
"fastly" = embed("<fa") + embed("fas") + embed("ast") + embed("stl") + ...
           (character n-grams are known even if "fastly" isn't)
```

### The Hybrid Model

libgrammstein combines both approaches:

```
score(word | context) = λ₁ × log P_ngram(word | context)
                      + λ₂ × similarity(word_embedding, context_embedding)
```

This achieves:
- **N-gram strength**: Precise local context ("the quick ___" → "brown" not "quickly")
- **Embedding strength**: Semantic similarity and OOV handling

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["serde"] }

# For lling-llang integration:
# libgrammstein = { version = "0.1", features = ["lling-llang-integration", "serde"] }
```

### Loading and Using a Pre-trained Model

```rust
use libgrammstein::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a trained hybrid model
    let model = HybridLanguageModel::load("model.bin")?;

    // Score a complete sequence (returns log probability)
    let log_prob = model.score_sequence(&["the", "quick", "brown", "fox"]);
    println!("log P(the quick brown fox) = {:.3}", log_prob);

    // Score a continuation: P(next | prefix)
    let continuation = model.score_continuation(&["the", "quick"], "brown");
    println!("P(brown | the quick) = {:.4}", continuation.exp());

    // Compare alternatives
    let p_brown = model.score_continuation(&["the", "quick"], "brown");
    let p_slow = model.score_continuation(&["the", "quick"], "slow");
    println!("'brown' is {:.1}x more likely than 'slow'",
             (p_brown - p_slow).exp());

    Ok(())
}
```

### Training a Model from Corpus

```rust
use libgrammstein::prelude::*;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::TrainerBuilder;
use libgrammstein::embedding::EmbeddingTrainer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a corpus reader (streams from disk, doesn't load into memory)
    let reader = PlaintextReader::from_directory("./corpus")?;

    // Train N-gram model (5-gram with Modified Kneser-Ney)
    println!("Training N-gram model...");
    let ngram = TrainerBuilder::new()
        .order(5)                    // 5-gram model
        .min_count(2)                // Prune N-grams appearing < 2 times
        .train(&reader)?;

    // Train subword embeddings
    println!("Training embeddings...");
    let embedding = EmbeddingTrainer::new()
        .dimension(100)              // 100-dimensional vectors
        .window(5)                   // Context window size
        .epochs(10)                  // Training epochs
        .min_count(5)                // Minimum word frequency
        .train(&reader)?;

    // Combine into hybrid model
    let config = HybridConfig {
        ngram_weight: 0.8,           // 80% weight to N-gram
        embedding_weight: 0.2,       // 20% weight to embeddings
        ..Default::default()
    };
    let hybrid = HybridLanguageModel::new(ngram, embedding, config);

    // Save for later use
    hybrid.save("model.bin")?;
    println!("Model saved!");

    Ok(())
}
```

### Using N-gram Model Alone

For faster, simpler scoring without embeddings:

```rust
use libgrammstein::ngram::{NgramModel, TrainerBuilder};
use libgrammstein::corpus::PlaintextReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reader = PlaintextReader::from_file("corpus.txt")?;

    let model = TrainerBuilder::new()
        .order(3)  // Trigram
        .train(&reader)?;

    // Query probabilities
    let log_prob = model.log_prob("fox", &["quick", "brown"]);
    println!("log P(fox | quick brown) = {:.3}", log_prob);

    // Sentence probability
    let sentence_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
    println!("log P(sentence) = {:.3}", sentence_prob);

    // Perplexity (lower = better model fit)
    let perplexity = model.perplexity(&["the", "quick", "brown", "fox"]);
    println!("Perplexity = {:.1}", perplexity);

    Ok(())
}
```

## Integration with lling-llang

libgrammstein integrates with [lling-llang](https://github.com/f1r3fly-io/lling-llang), a WFST (Weighted Finite State Transducer) framework for text correction. It implements the `LanguageModel` trait for lattice rescoring:

```rust
use lling_llang::prelude::*;
use lling_llang::layers::{LayerPipelineBuilder, LanguageModelLayer, SpellingCorrectionLayer};
use libgrammstein::HybridLanguageModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load dictionary for spelling correction
    let dictionary = load_dictionary("words.txt")?;

    // Load libgrammstein language model
    let lm = HybridLanguageModel::load("model.bin")?;

    // Build correction pipeline
    let pipeline = LayerPipelineBuilder::new()
        // Layer 1: Generate spelling candidates via fuzzy matching
        .add_layer(SpellingCorrectionLayer::new(dictionary, 2))
        // Layer 2: Rescore with language model
        .add_layer(LanguageModelLayer::new(Box::new(lm), 1.0))
        .build();

    // Process misspelled input
    let input = tokenize("teh quikc brwon fox");
    let lattice = tokens_to_lattice(&input);

    // Apply pipeline
    let result = pipeline.apply(&lattice)?;
    let best = viterbi(&mut result);

    println!("Corrected: {}", best.labels.join(" "));
    // Output: "the quick brown fox"

    Ok(())
}
```

The pipeline:
1. **SpellingCorrectionLayer**: Uses liblevenshtein to find fuzzy matches for each token
2. **LanguageModelLayer**: Rescores candidates using libgrammstein, preferring fluent sequences

## Project Structure

```
libgrammstein/
├── src/
│   ├── ngram/           # N-gram model with Modified Kneser-Ney
│   ├── embedding/       # Subword embeddings (FastText-style)
│   ├── hybrid/          # Combined model
│   ├── corpus/          # Streaming corpus readers
│   ├── integration/     # lling-llang integration
│   └── scoring/         # Perplexity, sentence scoring
├── docs/                # Detailed documentation
├── examples/            # Example applications
└── benches/             # Performance benchmarks
```

## Documentation

For detailed documentation, see the [`docs/`](docs/README.md) directory:

- **Architecture**: System design, threading model, data flow
- **N-gram Model**: Modified Kneser-Ney algorithm, trie storage
- **Embeddings**: Skip-gram training, BPE tokenization, similarity
- **Hybrid Model**: Interpolation strategies, OOV handling
- **Integration**: lling-llang pipelines, liblevenshtein backends

## Performance

Typical performance characteristics:

| Operation | Time |
|-----------|------|
| N-gram lookup | ~100ns |
| Embedding lookup (cached) | ~10ns |
| Hybrid score | ~1μs |
| Sentence score (10 tokens) | ~10μs |

Training times (36 cores, 252GB RAM):

| Task | 1GB Corpus | 10GB Corpus |
|------|------------|-------------|
| N-gram counting | ~30 min | ~5 hours |
| Embedding training | ~2 hours | ~24 hours |

## Features

| Feature | Description |
|---------|-------------|
| `default` | Core N-gram and embedding functionality |
| `lling-llang-integration` | Implements `LanguageModel` trait |
| `serde` | Model serialization with bincode |
| `async` | Async corpus streaming (Tokio) |

## Related Projects

- [lling-llang](https://github.com/f1r3fly-io/lling-llang): WFST framework for text correction
- [liblevenshtein-rust](https://github.com/f1r3fly-io/liblevenshtein-rust): Fuzzy string matching and trie dictionaries
- [F1R3FLY.io](https://f1r3fly.io): Distributed computing platform

## License

MIT OR Apache-2.0
