//! Corpus processing with streaming readers.
//!
//! This module provides traits and implementations for reading text corpora
//! in a memory-efficient streaming fashion.
//!
//! # Supported Formats
//!
//! - **Wikipedia**: XML dump format with bz2 compression (streaming)
//! - **Project Gutenberg**: Plain text files with boilerplate stripping
//! - **Plaintext**: Generic text files or directories
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::{CorpusReader, PlaintextReader};
//!
//! let reader = PlaintextReader::from_file("corpus.txt")?;
//! for sentence in reader.sentences() {
//!     println!("{}", sentence);
//! }
//! ```

mod reader;
mod tokenizer;
mod normalizer;
mod plaintext;
mod wikipedia;
mod gutenberg;
mod quality;
mod dedup;
mod preprocessing;

#[cfg(feature = "subword")]
mod subword;

pub use reader::{CorpusReader, Document};
pub use tokenizer::Tokenizer;
pub use normalizer::Normalizer;
pub use plaintext::PlaintextReader;
pub use wikipedia::{WikipediaReader, WikipediaConfig};
pub use gutenberg::GutenbergReader;
pub use quality::{QualityFilter, QualityFilterBuilder, QualityMetrics, QualityStats, RejectionReason};
pub use dedup::{Deduplicator, DeduplicatorBuilder, DeduplicationMode, DeduplicationStats};
pub use preprocessing::{
    TextPreprocessor, TextPreprocessorBuilder, UnicodeNorm,
    PreprocessingPipeline, PreprocessingPipelineBuilder,
    tokens,
};

#[cfg(feature = "http-corpus")]
pub use wikipedia::LoadStrategy;

#[cfg(feature = "subword")]
pub use subword::{SubwordTokenizer, SubwordError, BpeConfig, special_tokens, TokenizeExt};
