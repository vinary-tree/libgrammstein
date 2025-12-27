//! Corpus processing with streaming readers.
//!
//! This module provides traits and implementations for reading text corpora
//! in a memory-efficient streaming fashion.
//!
//! # Supported Formats
//!
//! - **Wikipedia**: XML dump format with bz2 compression
//! - **Project Gutenberg**: Plain text files
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

pub use reader::{CorpusReader, Document};
pub use tokenizer::Tokenizer;
pub use normalizer::Normalizer;
pub use plaintext::PlaintextReader;

// These will be implemented in later phases
// mod wikipedia;
// mod gutenberg;
// pub use wikipedia::WikipediaReader;
// pub use gutenberg::GutenbergReader;
