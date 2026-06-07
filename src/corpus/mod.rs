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

mod dedup;
mod gutenberg;
mod normalizer;
mod plaintext;
mod prefetch;
mod preprocessing;
mod quality;
mod reader;
mod tokenizer;
mod wikipedia;

#[cfg(feature = "subword")]
mod subword;

pub use dedup::{DeduplicationMode, DeduplicationStats, Deduplicator, DeduplicatorBuilder};
pub use gutenberg::GutenbergReader;
pub use normalizer::Normalizer;
pub use plaintext::{LineIterator, PlaintextReader};
pub use prefetch::{PrefetchBatchIterator, PrefetchConfig, PrefetchingReader};
pub use preprocessing::{
    tokens, PreprocessingPipeline, PreprocessingPipelineBuilder, TextPreprocessor,
    TextPreprocessorBuilder, UnicodeNorm,
};
pub use quality::{
    QualityFilter, QualityFilterBuilder, QualityMetrics, QualityStats, RejectionReason,
};
pub use reader::{CorpusReader, Document};
pub use tokenizer::Tokenizer;
pub use wikipedia::{WikipediaConfig, WikipediaReader};

#[cfg(feature = "http-corpus")]
pub use wikipedia::LoadStrategy;

#[cfg(feature = "subword")]
pub use subword::{special_tokens, BpeConfig, SubwordError, SubwordTokenizer, TokenizeExt};
