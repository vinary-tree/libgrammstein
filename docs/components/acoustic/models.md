# Acoustic Models

Neural network acoustic models using the Candle ML framework.

## What are Acoustic Models?

Acoustic models map audio features to probability distributions over output units (phonemes, characters, or subword tokens). In the ASR cascade, they compute the emission probabilities that drive CTC or HMM decoding.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Acoustic Model in ASR                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Audio Features              Acoustic Model                Posteriors      │
│   [T, F]                      (Neural Network)              [T, U]          │
│                                                                             │
│   ┌───┬───┬───┐              ┌──────────────────┐         ┌───┬───┬───┐   │
│   │f₀₀│f₀₁│...│──────────────│  Transformer or  │─────────│p₀₀│p₀₁│...│   │
│   ├───┼───┼───┤              │  Linear Encoder  │         ├───┼───┼───┤   │
│   │f₁₀│f₁₁│...│              └──────────────────┘         │p₁₀│p₁₁│...│   │
│   ├───┼───┼───┤                                           ├───┼───┼───┤   │
│   │...│...│...│              T = time frames              │...│...│...│   │
│   └───┴───┴───┘              F = feature dim (40)         └───┴───┴───┘   │
│                              U = output units (4096)                        │
│                                                                             │
│   Posteriors represent: log P(unit | features)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Flag

Acoustic models require the `candle-model` feature:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["candle-model"] }
```

This enables:
- `candle-core` - Tensor operations
- `candle-nn` - Neural network layers
- `acoustic` - Feature extraction (automatically included)

## AcousticModel Trait

The core interface for acoustic models.

```rust
pub trait AcousticModel: Send + Sync {
    /// Input feature dimension (e.g., 40 for filterbank)
    fn feature_dim(&self) -> usize;

    /// Number of output units (vocabulary size)
    fn num_units(&self) -> usize;

    /// Compute log posteriors for input frames
    /// Input: [batch_size, feature_dim]
    /// Output: [batch_size, num_units]
    fn forward(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// CTC blank token ID (if using CTC)
    fn blank_id(&self) -> Option<u32> { None }

    /// Optional: Get unit name for debugging
    fn unit_name(&self, unit: u32) -> Option<String> { None }
}
```

## AcousticModelConfig

Configuration for all acoustic model types.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_dim` | `usize` | 40 | Input feature dimension |
| `hidden_dim` | `usize` | 256 | Hidden layer dimension |
| `num_units` | `usize` | 4096 | Output vocabulary size |
| `num_layers` | `usize` | 6 | Encoder layers |
| `dropout` | `f64` | 0.1 | Dropout probability |
| `num_heads` | `usize` | 4 | Attention heads (transformer) |
| `ff_dim` | `usize` | 1024 | Feed-forward dimension |
| `is_ctc` | `bool` | false | Has CTC blank token |
| `blank_id` | `u32` | 0 | Blank token ID |

### Preset Configurations

```rust
use libgrammstein::acoustic::AcousticModelConfig;

// Small: Fast inference, lower accuracy
let small = AcousticModelConfig::small();
// hidden_dim: 128, num_layers: 2, num_heads: 2

// Medium: Balanced (default)
let medium = AcousticModelConfig::medium();
// hidden_dim: 256, num_layers: 6, num_heads: 4

// Large: High accuracy, slower
let large = AcousticModelConfig::large();
// hidden_dim: 512, num_layers: 12, num_heads: 8
```

### Builder Pattern

```rust
let config = AcousticModelConfig::default()
    .with_feature_dim(80)      // 80-dim filterbank
    .with_num_units(4096)      // Vocabulary size
    .with_hidden_dim(512)      // Larger hidden layer
    .with_num_layers(12)       // More layers
    .with_ctc(0);              // Enable CTC with blank_id=0
```

## LinearAcousticModel

A simple baseline model with a single hidden layer.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LinearAcousticModel                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input [B, F]                                                              │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │  Linear (F → H)     │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │       ReLU          │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │  Linear (H → U)     │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │    Log Softmax      │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   Output [B, U]  (log posteriors)                                           │
│                                                                             │
│   B = batch size, F = feature_dim, H = hidden_dim, U = num_units            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```rust
use libgrammstein::acoustic::{LinearAcousticModel, AcousticModelConfig, AcousticModel};
use candle_core::Device;

// Configure model
let config = AcousticModelConfig {
    feature_dim: 40,
    hidden_dim: 256,
    num_units: 4096,
    ..Default::default()
};

// Create on GPU if available
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
let model = LinearAcousticModel::new(config, &device).expect("Failed to create model");

// Forward pass
let features = vec![vec![0.0f32; 40]; 100];  // 100 frames
let posteriors = model.forward(&features);    // [100, 4096]

// Get best unit per frame
for (i, frame_post) in posteriors.iter().enumerate() {
    let best_unit = frame_post
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    println!("Frame {}: best unit = {}", i, best_unit);
}
```

### Loading Pretrained Weights

```rust
// Load from safetensors file
let model = LinearAcousticModel::load(
    "linear_acoustic.safetensors",
    config,
    &device,
).expect("Failed to load model");
```

## TransformerAcousticModel

State-of-the-art acoustic model using transformer encoder layers with self-attention.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TransformerAcousticModel                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input [B, T, F]                                                           │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────┐                                                   │
│   │ Linear (F → H)      │  Input projection                                 │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │ + Positional Enc    │  Sinusoidal position encoding                     │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Transformer Layers (×N)                          │   │
│   │  ┌────────────────────────────────────────────────────────────────┐ │   │
│   │  │  ┌─────────────────┐    ┌─────────────────┐                    │ │   │
│   │  │  │  Multi-Head     │───►│   Add & Norm    │                    │ │   │
│   │  │  │  Self-Attention │    └────────┬────────┘                    │ │   │
│   │  │  │  (H heads)      │             │                             │ │   │
│   │  │  └─────────────────┘             ▼                             │ │   │
│   │  │                      ┌─────────────────┐    ┌─────────────────┐│ │   │
│   │  │                      │  Feed-Forward   │───►│   Add & Norm    ││ │   │
│   │  │                      │  (H → FF → H)   │    └────────┬────────┘│ │   │
│   │  │                      │  + GELU         │             │         │ │   │
│   │  │                      └─────────────────┘             ▼         │ │   │
│   │  └─────────────────────────────────────────────────────────────────┘ │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────┐                                                   │
│   │ Linear (H → U)      │  Output projection                                │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   ┌─────────────────────┐                                                   │
│   │    Log Softmax      │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   Output [B, T, U]  (log posteriors per frame)                              │
│                                                                             │
│   B = batch, T = time, F = features, H = hidden, U = units, N = num_layers  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Self-Attention

The self-attention mechanism allows each frame to attend to all other frames:

```
                    Query (Q)      Key (K)       Value (V)
                       ↓             ↓              ↓
                    ┌──────────────────────────────────┐
                    │                                  │
                    │    Attention = softmax(QK^T/√d)V │
                    │                                  │
                    └──────────────────────────────────┘
                                    │
                                    ▼
                              Context-aware
                              representation
```

### Usage

```rust
use libgrammstein::acoustic::{
    TransformerAcousticModel, AcousticModelConfig, AcousticModel
};
use candle_core::Device;

// Configure transformer model
let config = AcousticModelConfig {
    feature_dim: 40,
    hidden_dim: 256,
    num_units: 4096,
    num_layers: 6,
    num_heads: 4,
    ff_dim: 1024,
    dropout: 0.1,
    is_ctc: true,
    blank_id: 0,
    ..Default::default()
};

// Create model
let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
let model = TransformerAcousticModel::new(config, &device)
    .expect("Failed to create model");

// Print model info
println!("Feature dim: {}", model.feature_dim());   // 40
println!("Output units: {}", model.num_units());    // 4096
println!("Blank ID: {:?}", model.blank_id());       // Some(0)

// Forward pass maintains temporal structure
let features = vec![vec![0.0f32; 40]; 100];  // 100 frames, 40 dims each
let posteriors = model.forward(&features);    // [100, 4096]

assert_eq!(posteriors.len(), 100);
assert_eq!(posteriors[0].len(), 4096);
```

### Loading Pretrained Model

```rust
// Load pretrained weights
let model = TransformerAcousticModel::load(
    "transformer_acoustic.safetensors",
    config,
    &device,
).expect("Failed to load model");
```

## MockAcousticModel

A testing model that returns uniform log probabilities.

```rust
use libgrammstein::acoustic::{MockAcousticModel, AcousticModelConfig, AcousticModel};

let config = AcousticModelConfig::default();
let model = MockAcousticModel::new(config);

// All outputs are uniform log probabilities
let features = vec![vec![0.0f32; 40]; 100];
let posteriors = model.forward(&features);

// Each frame has uniform distribution over units
let first_posterior = &posteriors[0];
let expected_log_prob = -(config.num_units as f32).ln();
assert!((first_posterior[0] - expected_log_prob).abs() < 1e-5);
```

## CTC Integration

When using CTC decoding, configure the blank token:

```rust
let config = AcousticModelConfig::default()
    .with_ctc(0);  // blank_id = 0

let model = TransformerAcousticModel::new(config, &device)?;

// Blank token is first output unit
assert_eq!(model.blank_id(), Some(0));

// Forward pass includes blank probability
let posteriors = model.forward(&features);
let blank_log_prob = posteriors[0][0];  // log P(blank | frame_0)
```

## Complete ASR Example

```rust
use libgrammstein::acoustic::{
    FeatureExtractor, FeatureConfig,
    TransformerAcousticModel, AcousticModelConfig,
    AcousticModel,
};
use candle_core::Device;

fn transcribe(audio_path: &str) -> String {
    // Step 1: Configure feature extraction
    let feature_config = FeatureConfig::default();
    let extractor = FeatureExtractor::new(feature_config);

    // Step 2: Load audio
    let audio = load_audio_16khz(audio_path);

    // Step 3: Extract features
    let features = extractor.extract_filterbank(&audio);
    println!("Extracted {} frames", features.len());

    // Step 4: Configure acoustic model
    let model_config = AcousticModelConfig::default()
        .with_num_units(4096)
        .with_ctc(0);

    // Step 5: Load acoustic model
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let model = TransformerAcousticModel::load(
        "acoustic_model.safetensors",
        model_config,
        &device,
    ).expect("Failed to load model");

    // Step 6: Get posteriors
    let posteriors = model.forward(&features);

    // Step 7: Greedy CTC decode
    let blank_id = model.blank_id().unwrap_or(0);
    let mut prev_unit = blank_id;
    let mut decoded = Vec::new();

    for frame_posteriors in &posteriors {
        let best_unit = frame_posteriors
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        // CTC rule: skip blanks and repeated units
        if best_unit != blank_id && best_unit != prev_unit {
            decoded.push(best_unit);
        }
        prev_unit = best_unit;
    }

    // Convert to text (depends on your vocabulary)
    decode_units_to_text(&decoded)
}
```

## Device Selection

```rust
use candle_core::Device;

// CPU (always available)
let device = Device::Cpu;

// CUDA GPU (if available)
let device = Device::cuda_if_available(0)?;  // GPU index 0

// Metal (Apple Silicon)
#[cfg(target_os = "macos")]
let device = Device::new_metal(0)?;

// Automatic best device
let device = if Device::is_cuda_available() {
    Device::new_cuda(0)?
} else if Device::is_metal_available() {
    Device::new_metal(0)?
} else {
    Device::Cpu
};
```

## Performance Tips

### Batch Processing

```rust
// Process multiple frames together for better GPU utilization
let batch_size = 32;
let features: Vec<Vec<f32>> = /* ... */;

for batch in features.chunks(batch_size) {
    let posteriors = model.forward(batch);
    // Process batch...
}
```

### Model Size Trade-offs

| Model | Hidden | Layers | Params | Speed | Accuracy |
|-------|--------|--------|--------|-------|----------|
| Small | 128 | 2 | ~1M | Fast | Lower |
| Medium | 256 | 6 | ~10M | Medium | Good |
| Large | 512 | 12 | ~50M | Slow | Best |

## Related Documentation

- [Acoustic Overview](overview.md) - Module introduction
- [Feature Extraction](features.md) - Audio preprocessing
- [lling-llang AcousticModel](../../../lling-llang/docs/acoustic/overview.md) - Integration
