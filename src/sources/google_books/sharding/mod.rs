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
//! │Shard a│ │...│   │th │   │zz │  ← Each shard written lock-free
//! └───────┘ └───┘   └───┘   └───┘
//! ```
//!
//! # Sharding Strategy
//!
//! An n-gram's live key is a concatenated LEB128 varint **term-id** byte sequence
//! (`crate::ngram::vocabulary`), never a delimited string. Routing consults only
//! the **first token's characters** and the **sequence length**
//! ([`compute_shard_key_from_token`]) — never a term-id value:
//!
//! - **1-grams**: 26 shards (a-z) under prefix-based granularities
//! - **2-5 grams**: 676 shards (aa-zz) under `TwoChar`/`Adaptive`
//! - **any order**: `max(cpus * multiplier, minimum)` shards under the default
//!   hash-based `CpuProportional`, indexed by `hash(first_token) % num_shards`
//!
//! This matches Google Books file partitioning, enabling lock-free parallel writes
//! where each worker writes to its own shard without coordination. The only place
//! delimited text is parsed on the live path is the labeled Google-Books input
//! boundary (`NgramStorage::store_ngram`, `str::split(' ')`), which immediately
//! transcodes space-delimited tokens to term-ids.
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::ngram::vocabulary::{create_vocabulary, encode_varint};
//! use libgrammstein::sources::google_books::sharding::{
//!     compute_shard_key_from_token, MergeCoordinator, ShardConfig, ShardCoordinator,
//!     ShardGranularity,
//! };
//!
//! // Create coordinator with adaptive sharding.
//! let config = ShardConfig::new("/tmp/shards").with_granularity(ShardGranularity::Adaptive);
//! let coordinator = ShardCoordinator::new(config)?;
//! let vocab = create_vocabulary("/tmp/vocab")?;
//!
//! // Encode ["the", "quick", "brown"] as a term-id byte key, route by the first
//! // token's characters, and store lock-free — no delimited representation is
//! // ever formed (concurrent `store_in_shard` calls need no writer token or lock).
//! let ids = ["the", "quick", "brown"].map(|w| vocab.as_ref().insert(w).unwrap());
//! let mut key = Vec::new();
//! for id in ids { encode_varint(id, &mut key); }
//! let first = vocab.get_term(ids[0]).unwrap();
//! let shard = compute_shard_key_from_token(&first, ids.len() as u8, &ShardGranularity::Adaptive);
//! coordinator.store_in_shard(&shard, &key, 100)?;
//!
//! // After import, merge all shards into a single in-memory n-gram map.
//! let merged = MergeCoordinator::new(&coordinator).merge_to_memory()?;
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
//! 3. **Export**: Materialize as a byte-keyed trie (`merge_to_trie`) or in-memory map (`merge_to_memory`)

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
    CheckpointError, CheckpointManager, CheckpointResult, CheckpointSummary, GlobalCheckpoint,
    ImportPhase, ImportState, ShardCheckpointRecord,
};
pub use config::{MergeConfig, MergeMode, ShardConfig, ShardGranularity};
pub use coordinator::{
    CheckpointHandle, CoordinatorError, CoordinatorPrefixTx, CoordinatorResult, CoordinatorStats,
    ShardCoordinator, ShardSummary,
};
pub use merge::{
    MergeBuilder, MergeCoordinator, MergeError, MergeProgress, MergeResult, MergeStats,
};
pub use mkn::{
    ContinuationCounts, DiscountParams, FrequencyCounts, MknAggregator, MknError, MknResult,
    MknStats, MknSummary, OrderSummary,
};
pub use query::{ShardedTrieView, ViewStats};
pub use routing::{
    all_shard_keys, compute_shard_key_from_token, shard_key_for_file_prefix, ShardKey,
};
pub use shard::{
    PrefixTransaction, ShardError, ShardHandle, ShardResult, ShardStats, ShardSyncHandle,
};
