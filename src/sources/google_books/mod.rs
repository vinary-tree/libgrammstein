//! Google Books N-gram integration for libgrammstein.
//!
//! This module provides import capabilities from the Google Books N-gram dataset,
//! enabling high-quality n-gram language models without training from raw text.
//!
//! ## Dataset Overview
//!
//! The Google Books N-gram dataset contains n-gram frequencies (1-5 grams) extracted
//! from over 8 million books spanning several centuries. Data is available for:
//!
//! - English (eng), German (ger), French (fre), Spanish (spa), Italian (ita)
//! - Russian (rus), Hebrew (heb), Chinese Simplified (chi-sim)
//!
//! ## Data Format
//!
//! Files are tab-separated with format: `ngram\tyear\tmatch_count\tvolume_count`
//!
//! Example:
//! ```text
//! the cat sat    1950    12345    678
//! the cat sat    1960    23456    789
//! ```
//!
//! ## Import Pipeline
//!
//! ```text
//! HTTP/Files → Parser → YearAggregator → MknStatsComputer → PersistentARTrie
//!                                                                   │
//!                                                                   ▼
//!                                                            PathMap (prod)
//! ```
//!
//! ## Features
//!
//! - **Streaming import**: Process terabytes without loading into memory
//! - **Parallel downloads**: Multiple prefix files downloaded concurrently
//! - **Checkpoint/resume**: Graceful handling of interruptions
//! - **MKN smoothing**: Full Modified Kneser-Ney statistics computation
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::{GoogleBooksConfig, GoogleBooksImporter};
//!
//! // Configure import
//! let config = GoogleBooksConfig::builder()
//!     .language("en")
//!     .orders(1..=5)
//!     .min_count(40)
//!     .output_path("english.artrie")
//!     .build()?;
//!
//! // Import with resume support
//! let mut importer = GoogleBooksImporter::resume_or_start(config).await?;
//! importer.import_http(|progress| println!("{:?}", progress)).await?;
//! importer.finalize()?;
//! ```

mod aggregator;
mod checkpoint;
mod checkpoint_decoder;
mod config;
mod events;
mod extractor;
mod importer;
mod languages;
mod parser;
mod reader;
pub mod sharding;
pub mod state_machine;
mod storage;
pub mod task_manager;
mod translator;

pub use aggregator::{AggregateExt, AggregatingIterator, YearAggregator};
pub use checkpoint::{ImportCheckpoint, MknPhase, PrefixState};
pub use checkpoint_decoder::{decode_checkpoint, decode_checkpoint_summary};
pub use config::{GoogleBooksConfig, ShardingGranularity, ShardingMode, ShardingOptions};
pub use events::{ImportCommand, ImportEvent, LogLevel};
pub use extractor::{DictionaryExtractor, ExtractionPhase, ExtractionProgress, ExtractionStats};
#[cfg(feature = "google-books")]
pub use importer::{
    run_import_with_periodic_checkpoints, CheckpointState, DEFAULT_CHECKPOINT_INTERVAL_MS,
};
pub use importer::{
    run_import_with_shutdown, shutdown_signal, GoogleBooksImporter, ImportPhase, ImportProgress,
    ImportStats, WorkerUpdate,
};
pub use languages::{
    get_order_urls, is_valid_prefix, list_languages, LanguageInfo, LanguageMetadata,
    SUPPORTED_LANGUAGES,
};
pub use parser::{parse_ngram_line, parse_ngram_lines, NgramRecord};
pub use reader::{AggregateReaderExt, AggregatingReaderIterator, ReaderBuilder};
pub use reader::{FileNgramReader, HttpNgramReader, NgramReader, ReaderError};
pub use storage::{NgramStorage, StorageError, StorageResult, StorageStats};
#[cfg(feature = "google-books")]
pub use task_manager::{
    Job, MetricsSnapshot, RetryAfter, TaskManager, TaskManagerConfig, TaskManagerMetrics,
    TaskSubmitter,
};
pub use translator::{PathMapTranslator, TranslationPhase, TranslationProgress, TranslationStats};
