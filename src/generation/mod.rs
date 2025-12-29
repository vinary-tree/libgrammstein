//! Text generation via autoregressive sampling.
//!
//! This module provides text generation capabilities using trained language models.
//! It supports multiple sampling strategies:
//!
//! - **Greedy decoding**: Always select the highest probability token
//! - **Nucleus (top-p) sampling**: Sample from the smallest set with cumulative probability >= p
//! - **Top-k sampling**: Sample from the k highest probability tokens
//! - **Temperature scaling**: Adjust the sharpness of the probability distribution
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::generation::{TextGenerator, GenerationConfig};
//! use libgrammstein::ngram::NgramModel;
//!
//! let model = NgramModel::load("model.bin")?;
//! let generator = TextGenerator::new(model, GenerationConfig::default());
//!
//! // Generate text with default nucleus sampling
//! let text = generator.generate(&["the", "quick"]);
//! println!("Generated: {}", text.join(" "));
//!
//! // Or use greedy decoding for deterministic output
//! let config = GenerationConfig::greedy().with_max_tokens(10);
//! let generator = TextGenerator::new(model, config);
//! let text = generator.generate(&["hello"]);
//! ```

mod sampler;

pub use sampler::{GenerationConfig, TextGenerator};
