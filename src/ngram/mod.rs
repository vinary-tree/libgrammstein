//! N-gram language model with Modified Kneser-Ney smoothing.
//!
//! This module provides a complete n-gram language model implementation that uses
//! liblevenshtein-rust's dictionary backends for efficient storage and retrieval.
//!
//! # Overview
//!
//! The n-gram model supports:
//! - Orders 1-5 (unigrams through 5-grams)
//! - Modified Kneser-Ney smoothing for probability estimation
//! - Streaming corpus training with Rayon parallelism
//! - Efficient probability queries via trie navigation
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::NgramModel;
//! use libgrammstein::corpus::PlaintextReader;
//!
//! let reader = PlaintextReader::from_directory("corpus/")?;
//! let model = NgramModel::train(reader, 3)?; // trigram model
//!
//! let log_prob = model.log_prob("fox", &["quick", "brown"]);
//! ```

mod entry;
mod model;
pub mod smoothing;
mod trainer;
mod trie;

pub use entry::NgramEntry;
pub use model::NgramModel;
pub use trainer::{NgramTrainer, TrainerBuilder, TrainingConfig, TrainingProgress, TrainingStats};
pub use trie::NgramTrie;
