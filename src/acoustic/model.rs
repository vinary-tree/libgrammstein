//! Neural acoustic model implementations using Candle.
//!
//! This module provides acoustic model implementations for speech recognition:
//!
//! - **LinearAcousticModel**: Simple linear projection (baseline/testing)
//! - **BiLstmAcousticModel**: Bidirectional LSTM encoder
//! - **TransformerAcousticModel**: Transformer encoder
//!
//! # Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         Acoustic Model Architecture                          │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                              │
//! │   Input: [batch, time, feature_dim]                                          │
//! │                    │                                                         │
//! │                    ▼                                                         │
//! │   ┌────────────────────────────────┐                                        │
//! │   │  Input Projection (Linear)     │                                        │
//! │   │  [feature_dim] → [hidden_dim]  │                                        │
//! │   └────────────────────────────────┘                                        │
//! │                    │                                                         │
//! │                    ▼                                                         │
//! │   ┌────────────────────────────────┐                                        │
//! │   │  Encoder (LSTM/Transformer)    │                                        │
//! │   │  [hidden_dim] → [hidden_dim]   │                                        │
//! │   └────────────────────────────────┘                                        │
//! │                    │                                                         │
//! │                    ▼                                                         │
//! │   ┌────────────────────────────────┐                                        │
//! │   │  Output Projection (Linear)    │                                        │
//! │   │  [hidden_dim] → [num_units]    │                                        │
//! │   └────────────────────────────────┘                                        │
//! │                    │                                                         │
//! │                    ▼                                                         │
//! │   ┌────────────────────────────────┐                                        │
//! │   │  Log Softmax                   │                                        │
//! │   │  [num_units] → log posteriors  │                                        │
//! │   └────────────────────────────────┘                                        │
//! │                                                                              │
//! │   Output: [batch, time, num_units]                                           │
//! │                                                                              │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::acoustic::{TransformerAcousticModel, AcousticModelConfig};
//! use candle_core::Device;
//!
//! let config = AcousticModelConfig {
//!     feature_dim: 40,
//!     hidden_dim: 256,
//!     num_units: 4096,
//!     num_layers: 6,
//!     ..Default::default()
//! };
//!
//! let device = Device::cuda_if_available(0)?;
//! let model = TransformerAcousticModel::new(config, &device)?;
//!
//! // Forward pass
//! let frames = vec![vec![0.0f32; 40]; 100];
//! let posteriors = model.forward(&frames);
//! ```

use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, ops, Linear, VarBuilder, VarMap};

/// Configuration for acoustic models.
#[derive(Clone, Debug)]
pub struct AcousticModelConfig {
    /// Input feature dimension (e.g., 40 for filterbank).
    pub feature_dim: usize,

    /// Hidden layer dimension.
    pub hidden_dim: usize,

    /// Number of output units (senones, phonemes, or characters + blank).
    pub num_units: usize,

    /// Number of encoder layers.
    pub num_layers: usize,

    /// Dropout probability (0.0 to disable).
    pub dropout: f64,

    /// Number of attention heads (for transformer models).
    pub num_heads: usize,

    /// Feed-forward dimension in transformer (typically 4x hidden_dim).
    pub ff_dim: usize,

    /// Whether this is a CTC model (has blank token).
    pub is_ctc: bool,

    /// Blank token ID (typically 0 for CTC).
    pub blank_id: u32,
}

impl Default for AcousticModelConfig {
    fn default() -> Self {
        Self {
            feature_dim: 40,
            hidden_dim: 256,
            num_units: 4096,
            num_layers: 6,
            dropout: 0.1,
            num_heads: 4,
            ff_dim: 1024,
            is_ctc: true,
            blank_id: 0,
        }
    }
}

impl AcousticModelConfig {
    /// Create config for a small model (fast inference).
    pub fn small() -> Self {
        Self {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 2,
            ff_dim: 512,
            ..Default::default()
        }
    }

    /// Create config for a medium model (balanced).
    pub fn medium() -> Self {
        Self::default()
    }

    /// Create config for a large model (high accuracy).
    pub fn large() -> Self {
        Self {
            hidden_dim: 512,
            num_layers: 12,
            num_heads: 8,
            ff_dim: 2048,
            ..Default::default()
        }
    }

    /// Set the number of output units.
    pub fn with_num_units(mut self, num_units: usize) -> Self {
        self.num_units = num_units;
        self
    }

    /// Set the feature dimension.
    pub fn with_feature_dim(mut self, feature_dim: usize) -> Self {
        self.feature_dim = feature_dim;
        self
    }

    /// Configure for CTC.
    pub fn with_ctc(mut self, blank_id: u32) -> Self {
        self.is_ctc = true;
        self.blank_id = blank_id;
        self
    }
}

/// Trait for acoustic models (mirror of lling-llang's trait).
///
/// This is a local trait for libgrammstein to avoid circular dependencies.
/// Models implementing this can be adapted to lling-llang's `AcousticModel`.
pub trait AcousticModel: Send + Sync {
    /// Frame feature dimensionality.
    fn feature_dim(&self) -> usize;

    /// Number of output units.
    fn num_units(&self) -> usize;

    /// Compute log posteriors for frames.
    fn forward(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Optional: Get blank token ID for CTC models.
    fn blank_id(&self) -> Option<u32> {
        None
    }

    /// Optional: Get unit name.
    fn unit_name(&self, _unit: u32) -> Option<String> {
        None
    }
}

/// Simple linear acoustic model for testing.
///
/// This is a single-layer linear projection from input features to output units.
/// Useful for baseline comparisons and debugging.
pub struct LinearAcousticModel {
    input_proj: Linear,
    output_proj: Linear,
    device: Device,
    config: AcousticModelConfig,
}

impl LinearAcousticModel {
    /// Create a new linear acoustic model.
    pub fn new(config: AcousticModelConfig, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let input_proj = linear(config.feature_dim, config.hidden_dim, vb.pp("input_proj"))?;
        let output_proj = linear(config.hidden_dim, config.num_units, vb.pp("output_proj"))?;

        Ok(Self {
            input_proj,
            output_proj,
            device: device.clone(),
            config,
        })
    }

    /// Load model from safetensors file.
    pub fn load<P: AsRef<Path>>(
        path: P,
        config: AcousticModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };

        let input_proj = linear(config.feature_dim, config.hidden_dim, vb.pp("input_proj"))?;
        let output_proj = linear(config.hidden_dim, config.num_units, vb.pp("output_proj"))?;

        Ok(Self {
            input_proj,
            output_proj,
            device: device.clone(),
            config,
        })
    }

    fn forward_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.input_proj.forward(x)?;
        let h = h.relu()?;
        let logits = self.output_proj.forward(&h)?;
        ops::log_softmax(&logits, candle_core::D::Minus1)
    }
}

impl AcousticModel for LinearAcousticModel {
    fn feature_dim(&self) -> usize {
        self.config.feature_dim
    }

    fn num_units(&self) -> usize {
        self.config.num_units
    }

    fn forward(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if frames.is_empty() {
            return vec![];
        }

        let batch_size = frames.len();
        let feature_dim = self.config.feature_dim;

        // Flatten input
        let flat: Vec<f32> = frames.iter().flat_map(|f| f.iter().copied()).collect();

        // Create tensor
        let x = match Tensor::from_vec(flat, (batch_size, feature_dim), &self.device) {
            Ok(t) => t,
            Err(_) => return vec![vec![0.0; self.config.num_units]; batch_size],
        };

        // Forward pass
        let output = match self.forward_tensor(&x) {
            Ok(t) => t,
            Err(_) => return vec![vec![0.0; self.config.num_units]; batch_size],
        };

        // Extract results
        match output.to_vec2::<f32>() {
            Ok(v) => v,
            Err(_) => vec![vec![0.0; self.config.num_units]; batch_size],
        }
    }

    fn blank_id(&self) -> Option<u32> {
        if self.config.is_ctc {
            Some(self.config.blank_id)
        } else {
            None
        }
    }
}

/// Feed-forward network for transformer layers.
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn new(hidden_dim: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(hidden_dim, ff_dim, vb.pp("linear1"))?;
        let linear2 = linear(ff_dim, hidden_dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = h.gelu_erf()?;
        self.linear2.forward(&h)
    }
}

/// Simple multi-head self-attention.
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(hidden_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        assert!(
            hidden_dim % num_heads == 0,
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;

        let q_proj = linear(hidden_dim, hidden_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_dim, hidden_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_dim, hidden_dim, vb.pp("v_proj"))?;
        let out_proj = linear(hidden_dim, hidden_dim, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention: [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_heads, self.head_dim))?;

        // Transpose to [batch, num_heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq, hidden]
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape((batch, seq_len, hidden))?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

/// Single transformer encoder layer.
struct TransformerLayer {
    self_attn: SelfAttention,
    ff: FeedForward,
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    norm2_weight: Tensor,
    norm2_bias: Tensor,
    hidden_dim: usize,
}

impl TransformerLayer {
    fn new(hidden_dim: usize, num_heads: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = SelfAttention::new(hidden_dim, num_heads, vb.pp("self_attn"))?;
        let ff = FeedForward::new(hidden_dim, ff_dim, vb.pp("ff"))?;

        let norm1_weight =
            vb.get_with_hints(hidden_dim, "norm1.weight", candle_nn::Init::Const(1.0))?;
        let norm1_bias =
            vb.get_with_hints(hidden_dim, "norm1.bias", candle_nn::Init::Const(0.0))?;
        let norm2_weight =
            vb.get_with_hints(hidden_dim, "norm2.weight", candle_nn::Init::Const(1.0))?;
        let norm2_bias =
            vb.get_with_hints(hidden_dim, "norm2.bias", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            self_attn,
            ff,
            norm1_weight,
            norm1_bias,
            norm2_weight,
            norm2_bias,
            hidden_dim,
        })
    }

    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let var = x.var_keepdim(candle_core::D::Minus1)?;
        let eps = 1e-5;
        let x_norm = x
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + eps)?.sqrt()?)?;
        x_norm.broadcast_mul(weight)?.broadcast_add(bias)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let attn_out = self.self_attn.forward(x)?;
        let x = (x + attn_out)?;
        let x = self.layer_norm(&x, &self.norm1_weight, &self.norm1_bias)?;

        // Feed-forward with residual
        let ff_out = self.ff.forward(&x)?;
        let x = (x + ff_out)?;
        self.layer_norm(&x, &self.norm2_weight, &self.norm2_bias)
    }
}

/// Transformer-based acoustic model.
///
/// Uses a stack of transformer encoder layers with positional encoding.
pub struct TransformerAcousticModel {
    input_proj: Linear,
    layers: Vec<TransformerLayer>,
    output_proj: Linear,
    device: Device,
    config: AcousticModelConfig,
}

impl TransformerAcousticModel {
    /// Create a new transformer acoustic model.
    pub fn new(config: AcousticModelConfig, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let input_proj = linear(config.feature_dim, config.hidden_dim, vb.pp("input_proj"))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerLayer::new(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
                vb.pp(format!("layer_{}", i)),
            )?);
        }

        let output_proj = linear(config.hidden_dim, config.num_units, vb.pp("output_proj"))?;

        Ok(Self {
            input_proj,
            layers,
            output_proj,
            device: device.clone(),
            config,
        })
    }

    /// Load model from safetensors file.
    pub fn load<P: AsRef<Path>>(
        path: P,
        config: AcousticModelConfig,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };

        let input_proj = linear(config.feature_dim, config.hidden_dim, vb.pp("input_proj"))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerLayer::new(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
                vb.pp(format!("layer_{}", i)),
            )?);
        }

        let output_proj = linear(config.hidden_dim, config.num_units, vb.pp("output_proj"))?;

        Ok(Self {
            input_proj,
            layers,
            output_proj,
            device: device.clone(),
            config,
        })
    }

    /// Add sinusoidal positional encoding.
    fn add_positional_encoding(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch_size, seq_len, hidden_dim) = x.dims3()?;

        // Generate positional encodings
        let mut pe = vec![0.0f32; seq_len * hidden_dim];
        for pos in 0..seq_len {
            for i in 0..hidden_dim {
                let angle = pos as f64 / 10000_f64.powf((2 * (i / 2)) as f64 / hidden_dim as f64);
                pe[pos * hidden_dim + i] = if i % 2 == 0 {
                    angle.sin() as f32
                } else {
                    angle.cos() as f32
                };
            }
        }

        let pe_tensor = Tensor::from_vec(pe, (1, seq_len, hidden_dim), &self.device)?;
        x.broadcast_add(&pe_tensor)
    }

    fn forward_tensor(&self, x: &Tensor) -> Result<Tensor> {
        // Input projection
        let mut h = self.input_proj.forward(x)?;

        // Add positional encoding
        h = self.add_positional_encoding(&h)?;

        // Pass through transformer layers
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }

        // Output projection
        let logits = self.output_proj.forward(&h)?;

        // Log softmax
        ops::log_softmax(&logits, candle_core::D::Minus1)
    }
}

impl AcousticModel for TransformerAcousticModel {
    fn feature_dim(&self) -> usize {
        self.config.feature_dim
    }

    fn num_units(&self) -> usize {
        self.config.num_units
    }

    fn forward(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if frames.is_empty() {
            return vec![];
        }

        let batch_size = frames.len();
        let feature_dim = self.config.feature_dim;

        // For transformer, we treat all frames as a sequence (batch=1, seq=frames.len())
        let flat: Vec<f32> = frames.iter().flat_map(|f| f.iter().copied()).collect();

        let x = match Tensor::from_vec(flat, (1, batch_size, feature_dim), &self.device) {
            Ok(t) => t,
            Err(_) => return vec![vec![0.0; self.config.num_units]; batch_size],
        };

        let output = match self.forward_tensor(&x) {
            Ok(t) => t,
            Err(_) => return vec![vec![0.0; self.config.num_units]; batch_size],
        };

        // Squeeze batch dimension and get frame posteriors
        let output = match output.squeeze(0) {
            Ok(t) => t,
            Err(_) => return vec![vec![0.0; self.config.num_units]; batch_size],
        };

        match output.to_vec2::<f32>() {
            Ok(v) => v,
            Err(_) => vec![vec![0.0; self.config.num_units]; batch_size],
        }
    }

    fn blank_id(&self) -> Option<u32> {
        if self.config.is_ctc {
            Some(self.config.blank_id)
        } else {
            None
        }
    }
}

/// Mock acoustic model for testing (no neural network).
///
/// Returns uniform random posteriors for testing purposes.
pub struct MockAcousticModel {
    config: AcousticModelConfig,
}

impl MockAcousticModel {
    /// Create a new mock acoustic model.
    pub fn new(config: AcousticModelConfig) -> Self {
        Self { config }
    }
}

impl AcousticModel for MockAcousticModel {
    fn feature_dim(&self) -> usize {
        self.config.feature_dim
    }

    fn num_units(&self) -> usize {
        self.config.num_units
    }

    fn forward(&self, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Return uniform log probabilities
        let log_prob = -((self.config.num_units as f32).ln());
        frames
            .iter()
            .map(|_| vec![log_prob; self.config.num_units])
            .collect()
    }

    fn blank_id(&self) -> Option<u32> {
        if self.config.is_ctc {
            Some(self.config.blank_id)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acoustic_model_config_default() {
        let config = AcousticModelConfig::default();
        assert_eq!(config.feature_dim, 40);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_units, 4096);
        assert!(config.is_ctc);
        assert_eq!(config.blank_id, 0);
    }

    #[test]
    fn test_acoustic_model_config_small() {
        let config = AcousticModelConfig::small();
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.num_layers, 2);
    }

    #[test]
    fn test_acoustic_model_config_large() {
        let config = AcousticModelConfig::large();
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_mock_acoustic_model() {
        let config = AcousticModelConfig {
            feature_dim: 40,
            num_units: 100,
            ..Default::default()
        };

        let model = MockAcousticModel::new(config);

        assert_eq!(model.feature_dim(), 40);
        assert_eq!(model.num_units(), 100);
        assert_eq!(model.blank_id(), Some(0));

        let frames = vec![vec![0.0f32; 40]; 5];
        let posteriors = model.forward(&frames);

        assert_eq!(posteriors.len(), 5);
        assert_eq!(posteriors[0].len(), 100);
    }

    #[test]
    fn test_linear_acoustic_model() {
        let config = AcousticModelConfig {
            feature_dim: 40,
            hidden_dim: 64,
            num_units: 100,
            ..Default::default()
        };

        let device = Device::Cpu;
        let model = LinearAcousticModel::new(config, &device).expect("Failed to create model");

        assert_eq!(model.feature_dim(), 40);
        assert_eq!(model.num_units(), 100);

        let frames = vec![vec![0.0f32; 40]; 5];
        let posteriors = model.forward(&frames);

        assert_eq!(posteriors.len(), 5);
        assert_eq!(posteriors[0].len(), 100);

        // Check that outputs are log probabilities (should sum to ~1 when exponentiated)
        let sum: f32 = posteriors[0].iter().map(|&p| p.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Log softmax output should sum to ~1, got {}",
            sum
        );
    }

    #[test]
    fn test_transformer_acoustic_model() {
        let config = AcousticModelConfig {
            feature_dim: 40,
            hidden_dim: 64,
            num_units: 100,
            num_layers: 2,
            num_heads: 2,
            ff_dim: 128,
            ..Default::default()
        };

        let device = Device::Cpu;
        let model = TransformerAcousticModel::new(config, &device).expect("Failed to create model");

        assert_eq!(model.feature_dim(), 40);
        assert_eq!(model.num_units(), 100);

        let frames = vec![vec![0.1f32; 40]; 10]; // 10 frames
        let posteriors = model.forward(&frames);

        assert_eq!(posteriors.len(), 10);
        assert_eq!(posteriors[0].len(), 100);

        // Check that outputs are log probabilities
        let sum: f32 = posteriors[0].iter().map(|&p| p.exp()).sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Log softmax output should sum to ~1, got {}",
            sum
        );
    }

    #[test]
    fn test_empty_frames() {
        let config = AcousticModelConfig::default();
        let model = MockAcousticModel::new(config);

        let posteriors = model.forward(&[]);
        assert!(posteriors.is_empty());
    }

    #[test]
    fn test_config_builder() {
        let config = AcousticModelConfig::default()
            .with_feature_dim(80)
            .with_num_units(256)
            .with_ctc(0);

        assert_eq!(config.feature_dim, 80);
        assert_eq!(config.num_units, 256);
        assert!(config.is_ctc);
        assert_eq!(config.blank_id, 0);
    }
}
