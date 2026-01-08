# ModernBERT Model

The `ModernBertModel` struct provides a wrapper around the ModernBERT encoder for inference in libgrammstein.

## What is ModernBERT?

ModernBERT is a 149M parameter encoder-only transformer model optimized for:

- **Semantic understanding**: Contextual embeddings for similarity tasks
- **Masked language modeling**: Predicting masked tokens for scoring
- **Long contexts**: Up to 8,192 tokens per sequence

The model is pre-trained on large text corpora and available from [HuggingFace](https://huggingface.co/answerdotai/ModernBERT-base).

## Model Architecture

```
Input Text: "The quick brown fox"
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Tokenizer (WordPiece)                    │
│                                                              │
│  "The quick brown fox" → [101, 1996, 4248, 2829, 4419, 102] │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Embedding Layer                           │
│                                                              │
│  Token IDs → Token Embeddings (768-dim)                     │
│  + Position Embeddings                                       │
│  + Token Type Embeddings                                     │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Encoder (12 layers)                 │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Layer N:                                             │    │
│  │   Multi-Head Self-Attention (12 heads)              │    │
│  │   → Layer Norm → Feed-Forward (3072) → Layer Norm   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         × 12                                 │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Output Embeddings                           │
│                                                              │
│  [CLS] embedding (768-dim) - sentence representation        │
│  Token embeddings (768-dim each) - per-token representations│
└─────────────────────────────────────────────────────────────┘
     │
     ▼ (optional)
┌─────────────────────────────────────────────────────────────┐
│                    MLM Prediction Head                       │
│                                                              │
│  Token embeddings → Vocabulary logits (50,368)              │
│  Used for masked token prediction and scoring               │
└─────────────────────────────────────────────────────────────┘
```

## Model Specifications

| Property | Value |
|----------|-------|
| Model ID | `answerdotai/ModernBERT-base` |
| Parameters | 149M |
| Hidden size | 768 |
| Attention heads | 12 |
| Layers | 12 |
| Intermediate size | 3072 |
| Max sequence length | 8,192 |
| Vocabulary size | 50,368 |
| Model format | SafeTensors |

## Configuration

```rust
use libgrammstein::neural::{ModernBertConfig, Device};

let config = ModernBertConfig {
    // HuggingFace model identifier
    model_id: "answerdotai/ModernBERT-base".to_string(),

    // Compute device
    device: Device::Cpu,

    // Data type (F32 for accuracy, BF16 for speed)
    dtype: candle_core::DType::F32,

    // Maximum sequence length (tokens)
    max_seq_len: 8192,
};
```

### Device Options

| Device | Description | Requirements |
|--------|-------------|--------------|
| `Device::Cpu` | CPU inference | None (default) |
| `Device::Cuda(n)` | NVIDIA GPU | CUDA toolkit, cuDNN |
| `Device::Metal` | Apple GPU | macOS with Metal |

## Loading the Model

### From HuggingFace Hub

```rust
use libgrammstein::neural::{ModernBertModel, ModernBertConfig};

// Default configuration (downloads from HuggingFace)
let config = ModernBertConfig::default();
let model = ModernBertModel::load(&config)?;
```

The model files are automatically downloaded and cached in `~/.cache/huggingface/`.

### From Local Files

```rust
use std::path::Path;
use libgrammstein::neural::{ModernBertModel, ModernBertConfig, Device};

let model = ModernBertModel::load_from_files(
    Path::new("./models/tokenizer.json"),
    Path::new("./models/model.safetensors"),
    Path::new("./models/config.json"),
    Device::Cpu,
)?;
```

## Tokenization

### Encoding Text

```rust
// Single text
let tokens = model.encode("Hello, world!")?;
// Returns: TokenizedInput { input_ids, attention_mask, ... }

// Batch encoding
let texts = vec!["First sentence", "Second sentence"];
let batch = model.encode_batch(&texts)?;
```

### Decoding Tokens

```rust
let tokens = vec![101, 7592, 1010, 2088, 999, 102];
let text = model.decode(&tokens)?;
// Returns: "hello, world!"
```

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `[CLS]` | 101 | Sequence start, sentence embedding |
| `[SEP]` | 102 | Sequence end / separator |
| `[MASK]` | 103 | Masked token for MLM |
| `[PAD]` | 0 | Padding token |
| `[UNK]` | 100 | Unknown token |

## Embedding Generation

### Full Embeddings

```rust
// Get all token embeddings (batch_size × seq_len × hidden_size)
let embeddings = model.embed(&["Hello, world!"])?;
```

### Mean-Pooled Embeddings

```rust
// Get mean-pooled sentence embedding (hidden_size)
let embedding = model.embed_mean_pooled("Hello, world!")?;
```

### Batch Embeddings

```rust
let texts = vec!["First", "Second", "Third"];
let embeddings = model.embed_batch(&texts)?;
// Returns: Vec of (hidden_size,) embeddings
```

## Forward Pass

For advanced usage, access the raw transformer output:

```rust
let input_ids = model.encode("Hello")?.input_ids;
let attention_mask = /* ... */;

// Raw forward pass
let hidden_states = model.forward(&input_ids, &attention_mask)?;
// Shape: (batch_size, seq_len, hidden_size)
```

## MLM Prediction

Get vocabulary logits for masked token prediction:

```rust
// Input with [MASK] token
let text = "The [MASK] fox jumps";
let tokens = model.encode(text)?;

// Get MLM logits
let logits = model.get_mlm_logits(&tokens.input_ids, &tokens.attention_mask)?;
// Shape: (batch_size, seq_len, vocab_size)

// Find predicted token at mask position
let mask_pos = 2;  // Position of [MASK]
let predicted_id = logits.slice(/* ... */).argmax()?;
let predicted_token = model.decode(&[predicted_id])?;
```

## Model Properties

```rust
// Hidden dimension (768)
let hidden_size = model.hidden_size();

// Vocabulary size (50,368)
let vocab_size = model.vocab_size();

// Mask token ID (103)
let mask_id = model.mask_token_id();

// Get tokenizer reference
let tokenizer = model.tokenizer();

// Get device
let device = model.device();
```

## Memory Management

### GPU Memory

ModernBERT-base requires approximately:

| Precision | Model Size | Peak Memory (batch=1) |
|-----------|------------|----------------------|
| F32 | ~600 MB | ~1.5 GB |
| BF16 | ~300 MB | ~800 MB |

### Sequence Length Impact

Memory scales linearly with sequence length for embeddings and quadratically for attention:

```
Memory ≈ O(batch × seq_len × hidden) + O(batch × heads × seq_len²)
```

For long sequences, use `SlidingWindowCache` (see [Cache](cache.md)).

## Thread Safety

`ModernBertModel` is designed for shared ownership:

```rust
use std::sync::Arc;

// Wrap in Arc for sharing across threads
let model = Arc::new(ModernBertModel::load(&config)?);

// Clone Arc for each thread (zero-copy)
let model_clone = Arc::clone(&model);
std::thread::spawn(move || {
    model_clone.embed_mean_pooled("text")
});
```

## Error Handling

```rust
use libgrammstein::neural::{NeuralError, Result};

fn embed_text(model: &ModernBertModel, text: &str) -> Result<Vec<f32>> {
    match model.embed_mean_pooled(text) {
        Ok(embedding) => Ok(embedding),
        Err(NeuralError::Tokenization(msg)) => {
            eprintln!("Tokenization failed: {}", msg);
            Err(NeuralError::Tokenization(msg))
        }
        Err(NeuralError::Inference(msg)) => {
            eprintln!("Inference failed: {}", msg);
            Err(NeuralError::Inference(msg))
        }
        Err(e) => Err(e),
    }
}
```

## See Also

- [Overview](overview.md) - Neural module introduction
- [Embedder](embedder.md) - High-level embedding API
- [Rescorer](rescorer.md) - MLM-based scoring
- [Cache](cache.md) - Inference optimization
