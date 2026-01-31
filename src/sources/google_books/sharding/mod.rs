//! Sharded trie storage for Google Books n-gram import.
//!
//! This module provides a sharded storage architecture that distributes n-grams
//! across multiple trie instances based on prefix routing. This eliminates the
//! single-writer bottleneck of a centralized trie, enabling true parallel writes.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           ShardCoordinator              │
//! │  (orchestrates shards, checkpoints)     │
//! └───────────┬─────────────────────────────┘
//!             │
//!     ┌───────┼───────┐───────┐
//!     │       │       │       │
//! ┌───┴───┐ ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
//! │Shard a│ │...│   │th │   │zz │  ← Each shard has exclusive writer
//! └───────┘ └───┘   └───┘   └───┘
//! ```
//!
//! # Sharding Strategy
//!
//! N-grams are routed to shards based on the first character(s) of the first word:
//!
//! - **1-grams**: 26 shards (a-z)
//! - **2-5 grams**: 676 shards (aa-zz)
//!
//! This matches Google Books file partitioning, enabling lock-free parallel writes
//! where each worker writes to its own shard without coordination.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::sharding::{
//!     ShardCoordinator, ShardConfig, ShardGranularity,
//! };
//!
//! // Create coordinator with adaptive sharding
//! let config = ShardConfig::new("/tmp/shards")
//!     .with_granularity(ShardGranularity::Adaptive)
//!     .with_max_writers(8);
//!
//! let mut coordinator = ShardCoordinator::create(config)?;
//!
//! // Workers can write to different shards in parallel
//! let token = coordinator.acquire_writer("th", 0).await?;
//! coordinator.store_ngram("the|quick|brown", 100, &token)?;
//! coordinator.release_writer(token);
//!
//! // After import, merge shards into PathMap
//! coordinator.merge_to_pathmap("/tmp/final.pathmap").await?;
//! ```
//!
//! # Checkpoint & Recovery
//!
//! Each shard maintains its own WAL (Write-Ahead Log) for crash recovery.
//! A global checkpoint coordinates per-shard checkpoints for consistent recovery.
//!
//! # Merge Strategy
//!
//! After import completes, shards are merged using parallel reduction:
//!
//! 1. **Pairwise merge**: Merge adjacent shards in parallel
//! 2. **Reduce**: Continue until single shard remains
//! 3. **Export**: Convert to PathMap for production use

pub mod checkpoint;
pub mod config;
pub mod coordinator;
pub mod merge;
pub mod mkn;
pub mod query;
pub mod routing;
pub mod shard;

// Re-export commonly used types
pub use checkpoint::{
    CheckpointError, CheckpointManager, CheckpointResult, CheckpointSummary,
    GlobalCheckpoint, ImportPhase, ImportState, ShardCheckpointRecord,
};
pub use config::{MergeConfig, MergeMode, ShardConfig, ShardGranularity};
pub use coordinator::{
    CoordinatorError, CoordinatorPrefixTx, CoordinatorResult, CoordinatorStats, ShardCoordinator,
    ShardSummary,
};
pub use merge::{MergeBuilder, MergeCoordinator, MergeError, MergeProgress, MergeResult, MergeStats};
pub use mkn::{
    ContinuationCounts, DiscountParams, FrequencyCounts, MknAggregator, MknError, MknResult,
    MknStats, MknSummary, OrderSummary,
};
pub use query::{ShardedTrieView, ViewStats};
pub use routing::{
    all_shard_keys, compute_shard_key, compute_shard_key_from_token, ngram_order,
    shard_key_for_file_prefix, ShardKey,
};
pub use shard::{PrefixTransaction, ShardError, ShardHandle, ShardResult, ShardStats, WriteToken};
