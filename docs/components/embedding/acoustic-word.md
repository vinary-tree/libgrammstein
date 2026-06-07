# Acoustic Word Embeddings

Fixed-dimensional representations of variable-length audio for query-by-example search.

## What are Acoustic Word Embeddings?

Acoustic Word Embeddings (AWE) convert variable-length audio segments into fixed-dimensional vectors that capture pronunciation. Unlike text embeddings which represent spelling, AWE represents how words *sound*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Acoustic Word Embedding                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Variable-length audio              Fixed-dimensional embedding            │
│   (spoken word)                      (128 dimensions)                       │
│                                                                             │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐       │
│   │░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░│───►│[0.23, -0.45, 0.12, ...]    │       │
│   │     "hello" (50 frames)    │    │       128 floats             │       │
│   └─────────────────────────────┘    └─────────────────────────────┘       │
│                                                                             │
│   ┌───────────────────────────────┐  ┌─────────────────────────────┐       │
│   │░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░│───►│[0.24, -0.44, 0.11, ...]    │       │
│   │       "hello" (100 frames)    │  │       Similar vector!       │       │
│   └───────────────────────────────┘  └─────────────────────────────┘       │
│                                                                             │
│   Different durations → Same word → Similar embeddings                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Use Cases

| Application | Description |
|-------------|-------------|
| **Query-by-Example** | Find words that sound like an audio query |
| **Speaker Verification** | Compare speech samples for identity |
| **Keyword Spotting** | Detect spoken keywords without transcription |
| **Audio-Text Alignment** | Link audio to text embeddings |
| **OOV Handling** | Match unknown spoken words to vocabulary |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Acoustic Word Embedding Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Audio Frames [T, F]          Encoder              Pooling       Fixed [D] │
│                                                                             │
│   ┌───┬───┬───┬───┬───┐      ┌────────┐         ┌────────┐    ┌─────────┐ │
│   │ f₀│ f₁│ f₂│...│ fₜ│─────►│Acoustic│────────►│ Pool   │───►│Embedding│ │
│   └───┴───┴───┴───┴───┘      │Encoder │         │Strategy│    │  [D]    │ │
│                              └────────┘         └────────┘    └─────────┘ │
│    T = variable              [T, H]             [D]                        │
│    F = 40 (filterbank)                                                     │
│    H = hidden_dim                                                          │
│    D = embedding_dim                                                       │
│                                                                             │
│   Optional: Project to text embedding space for cross-modal retrieval      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pooling Strategies

Pooling aggregates variable-length frame sequences into fixed-dimensional vectors.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Pooling Strategies                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Mean Pooling:              Max Pooling:              Last Pooling:        │
│   ┌───────────────┐          ┌───────────────┐         ┌───────────────┐   │
│   │ h₀ + h₁ + h₂  │          │max(h₀,h₁,h₂) │         │      h₂       │   │
│   │ ──────────────│          │  per dim      │         │  (final state)│   │
│   │      T        │          └───────────────┘         └───────────────┘   │
│   └───────────────┘                                                         │
│                                                                             │
│   Attention Pooling:                    MeanMax Pooling:                    │
│   ┌─────────────────────────┐          ┌────────────────────────────────┐  │
│   │ Σ αᵢ × hᵢ               │          │ [mean(h) ; max(h)]              │  │
│   │ (weighted by attention) │          │  concatenate both               │  │
│   └─────────────────────────┘          └────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Strategy | Output Dim | Description | Best For |
|----------|------------|-------------|----------|
| **Mean** | D | Average all frames | General use (default) |
| **Max** | D | Max per dimension | Capturing peak activations |
| **Last** | D | Final frame only | RNN encoders |
| **Attention** | D | Learned weighting | When frames differ in importance |
| **MeanMax** | 2D | Concatenate mean+max | Maximum information |

## Configuration

### AcousticEmbeddingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | `usize` | 128 | Output embedding dimension |
| `feature_dim` | `usize` | 40 | Input feature dimension |
| `pooling` | `PoolingStrategy` | Mean | Aggregation method |
| `normalize` | `bool` | true | L2-normalize embeddings |
| `text_projection_dim` | `Option<usize>` | None | Project to text space |

```rust
use libgrammstein::embedding::{AcousticWordEmbedding, AcousticEmbeddingConfig, PoolingStrategy};

let config = AcousticEmbeddingConfig {
    embedding_dim: 128,
    feature_dim: 40,           // Match filterbank features
    pooling: PoolingStrategy::Mean,
    normalize: true,           // Unit-length embeddings
    text_projection_dim: Some(200),  // Align with text embeddings
};

let awe = AcousticWordEmbedding::new(config);
```

## Basic Usage

### Creating an Embedding Model

```rust
use libgrammstein::embedding::{AcousticWordEmbedding, AcousticEmbeddingConfig, PoolingStrategy};

// Default configuration
let awe = AcousticWordEmbedding::new(AcousticEmbeddingConfig::default());

// Custom configuration
let config = AcousticEmbeddingConfig {
    embedding_dim: 256,
    feature_dim: 80,
    pooling: PoolingStrategy::MeanMax,
    normalize: true,
    text_projection_dim: None,
};
let awe = AcousticWordEmbedding::new(config);
```

### Encoding Audio

```rust
// Audio features: [T frames, F dimensions]
let frames: Vec<Vec<f32>> = vec![vec![0.0f32; 40]; 100];  // 100 frames

// Encode to fixed-dimensional embedding
let embedding: Vec<f32> = awe.encode(&frames);
println!("Embedding dimension: {}", embedding.len());  // 128
```

### Computing Similarity

```rust
// Two audio samples of the same word
let audio1: Vec<Vec<f32>> = extract_features("hello_speaker1.wav");
let audio2: Vec<Vec<f32>> = extract_features("hello_speaker2.wav");

// Compute cosine similarity
let similarity = awe.audio_similarity(&audio1, &audio2);
println!("Similarity: {:.3}", similarity);  // ~0.85 for same word
```

## Building a Word Index

### Adding Words

```rust
let mut awe = AcousticWordEmbedding::new(config);

// Add audio examples for words
let hello_frames = extract_features("hello.wav");
awe.add_word("hello", &hello_frames);

let goodbye_frames = extract_features("goodbye.wav");
awe.add_word("goodbye", &goodbye_frames);

let thanks_frames = extract_features("thanks.wav");
awe.add_word("thanks", &thanks_frames);
```

### Adding Pre-computed Embeddings

```rust
// If you already have embeddings
let embedding: Vec<f32> = compute_embedding_elsewhere();
awe.add_word_embedding("hello", embedding);
```

### Query-by-Example Search

```rust
// Query with spoken audio
let query_frames = extract_features("query_audio.wav");

// Find most similar words in index
let results = awe.query_by_example(&query_frames, 5);

println!("Query results:");
for (word, score) in results {
    println!("  {} ({:.3})", word, score);
}
// Output:
//   hello (0.92)
//   hallo (0.78)
//   yellow (0.65)
```

### Retrieving Embeddings

```rust
// Get embedding for indexed word
if let Some(embedding) = awe.get_word_embedding("hello") {
    println!("hello embedding: {:?}", &embedding.as_slice().unwrap()[..5]);
}
```

## Statistics

```rust
use libgrammstein::embedding::AcousticEmbeddingStats;

// Get index statistics
let stats: AcousticEmbeddingStats = awe.compute_stats();

println!("Indexed words: {}", stats.num_words);
println!("Embedding dim: {}", stats.embedding_dim);
println!("Average similarity: {:.3}", stats.avg_similarity);
println!("Min similarity: {:.3}", stats.min_similarity);
println!("Max similarity: {:.3}", stats.max_similarity);

// Get full pairwise similarity matrix
let sim_matrix = awe.all_pairwise_similarities();
// Shape: [num_words, num_words]
```

## Custom Encoder

The default `LinearEncoder` is simple. For domain-specific behavior, implement
`AcousticEncoder` with the sequence model or deterministic transform you need:

```rust
use libgrammstein::embedding::AcousticEncoder;

pub struct WindowedMeanEncoder {
    hidden_dim: usize,
    feature_dim: usize,
    radius: usize,
}

impl AcousticEncoder for WindowedMeanEncoder {
    fn encode_frames(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        (0..frames.len())
            .map(|i| {
                let start = i.saturating_sub(self.radius);
                let end = (i + self.radius + 1).min(frames.len());
                let mut output = vec![0.0; self.hidden_dim];
                let count = (end - start) as f32;

                for frame in &frames[start..end] {
                    for dim in 0..self.hidden_dim {
                        let source_dim = dim % self.feature_dim;
                        output[dim] += frame.get(source_dim).copied().unwrap_or(0.0) / count;
                    }
                }

                output
            })
            .collect()
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}

// Use custom encoder
use std::sync::Arc;
let encoder = Arc::new(WindowedMeanEncoder {
    hidden_dim: 256,
    feature_dim: 40,
    radius: 2,
});
let awe = AcousticWordEmbedding::with_encoder(encoder, config);
```

## Complete Example: Keyword Spotting

```rust
use libgrammstein::acoustic::{FeatureExtractor, FeatureConfig};
use libgrammstein::embedding::{
    AcousticWordEmbedding, AcousticEmbeddingConfig, PoolingStrategy
};

fn keyword_spotter() {
    // Step 1: Configure feature extraction
    let feature_config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(feature_config);

    // Step 2: Configure acoustic word embeddings
    let awe_config = AcousticEmbeddingConfig {
        embedding_dim: 128,
        feature_dim: 40,
        pooling: PoolingStrategy::Mean,
        normalize: true,
        text_projection_dim: None,
    };
    let mut awe = AcousticWordEmbedding::new(awe_config);

    // Step 3: Register keywords with example audio
    let keywords = ["hello", "goodbye", "stop", "start"];
    for keyword in &keywords {
        let audio = load_audio_16khz(&format!("{}.wav", keyword));
        let features = extractor.extract_filterbank(&audio);
        awe.add_word(*keyword, &features);
    }

    println!("Registered {} keywords", keywords.len());

    // Step 4: Detect keywords in continuous audio stream
    let stream_audio = load_audio_16khz("conversation.wav");

    // Slide a window over the audio
    let window_size = 16000;  // 1 second
    let hop_size = 8000;      // 0.5 second overlap
    let threshold = 0.80;     // Detection threshold

    for (i, start) in (0..stream_audio.len() - window_size).step_by(hop_size).enumerate() {
        let window: Vec<f32> = stream_audio[start..start + window_size].to_vec();
        let features = extractor.extract_filterbank(&window);

        // Query for matching keywords
        let results = awe.query_by_example(&features, 1);

        if let Some((keyword, score)) = results.first() {
            if *score >= threshold {
                let time_sec = start as f32 / 16000.0;
                println!("[{:.1}s] Detected '{}' (confidence: {:.2})",
                         time_sec, keyword, score);
            }
        }
    }
}
```

## Audio-Text Alignment

Project acoustic embeddings to text embedding space for cross-modal retrieval:

```rust
let config = AcousticEmbeddingConfig {
    embedding_dim: 128,
    text_projection_dim: Some(200),  // Match text embedding dimension
    ..Default::default()
};

let mut awe = AcousticWordEmbedding::new(config);

// After projection, acoustic embeddings are in same space as text
let audio_embedding = awe.encode(&audio_frames);  // [200] after projection

// Can directly compare with text embeddings
let text_embedding: Vec<f32> = text_model.embed("hello");  // [200]
let cross_modal_sim = cosine_similarity(&audio_embedding, &text_embedding);
```

## Performance Considerations

### Memory Usage

```rust
// Estimate memory for word index
let num_words = 10_000;
let embedding_dim = 128;
let memory_mb = (num_words * embedding_dim * 4) / (1024 * 1024);
println!("Index memory: {} MB", memory_mb);  // ~5 MB
```

### Query Speed

| Index Size | Query Time | Notes |
|------------|------------|-------|
| 1,000 words | ~0.1 ms | Exhaustive search |
| 10,000 words | ~1 ms | Exhaustive search |
| 100,000 words | ~10 ms | Consider approximate NN |

### Batch Encoding

```rust
// For many words, encode in batches
let all_audio: Vec<Vec<Vec<f32>>> = load_all_audio();

for (word, audio) in words.iter().zip(all_audio.iter()) {
    awe.add_word(word, audio);
}
```

## Related Documentation

- [Acoustic Overview](../acoustic/overview.md) - Audio feature extraction
- [Phonetic Embeddings](phonetic.md) - Text-based phonetic matching
- [lling-llang Integration](../../../lling-llang/docs/acoustic/overview.md)
