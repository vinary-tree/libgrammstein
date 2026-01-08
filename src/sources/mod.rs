//! External language model sources for libgrammstein.
//!
//! This module provides integrations with external n-gram datasets that can be
//! used to construct language models without training from raw text corpora.
//!
//! ## Available Sources
//!
//! - **Google Books N-grams**: High-quality n-gram frequencies from Google Books corpus.
//!   Supports 1-5 grams for multiple languages. Requires the `google-books` feature.
//!
//! ## Design Philosophy
//!
//! External sources follow a multi-stage pipeline:
//!
//! 1. **Import**: Stream n-grams (HTTP or local files) into PersistentARTrie for training
//! 2. **MKN Computation**: Compute Modified Kneser-Ney smoothing statistics
//! 3. **Translation**: Convert to PathMap for production deployment (optional)
//! 4. **Dictionary Extraction**: Build DoubleArrayTrieChar for lexical correction (optional)
//!
//! ## Concurrency
//!
//! All importers use non-blocking algorithms, atomics, and persistent data structures
//! to maximize parallelism:
//!
//! - Parallel HTTP downloads for different prefix files
//! - Atomic counters in `NgramEntry` for lock-free aggregation
//! - Rayon for CPU-bound parallel iteration
//! - Checkpoint/resume support for long-running imports
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::{GoogleBooksConfig, GoogleBooksImporter};
//!
//! let config = GoogleBooksConfig {
//!     language: "en".to_string(),
//!     orders: 1..=5,
//!     min_count: 40,
//!     output_path: "english.artrie".into(),
//!     ..Default::default()
//! };
//!
//! let mut importer = GoogleBooksImporter::resume_or_start(config).await?;
//! importer.import_http(|progress| {
//!     println!("Progress: {:?}", progress);
//! }).await?;
//!
//! let stats = importer.finalize()?;
//! println!("Imported {} n-grams", stats.total_entries);
//! ```

#[cfg(feature = "google-books")]
pub mod google_books;
