//! Configuration types for sharded trie storage.
//!
//! Sharding distributes n-grams across multiple trie instances based on
//! prefix keys, enabling parallel writes without lock contention.

use libdictenstein::persistent_artrie::eviction::EvictionConfig;
use std::path::PathBuf;

/// Floor for the per-shard overlay resident budget (~one arena), so a tiny
/// global budget or a huge shard count cannot drive eviction into thrashing.
const MIN_PER_SHARD_OVERLAY_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// Per-checkpoint cap on overlay nodes evicted in one pass. Bounds the
/// checkpoint-tail latency while staying large enough to keep up with the
/// per-file cold-set growth (tune from massif if the resident set accumulates).
const OVERLAY_EVICTION_CAP_NODES: usize = 200_000;

/// Sharding granularity options.
///
/// Determines how many shards are created and how n-grams are routed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardGranularity {
    /// 26 shards (a-z) - for 1-grams or smaller datasets.
    /// Uses prefix-based routing (all words starting with 'a' go to shard 'a').
    FirstChar,

    /// 676 shards (aa-zz) - for 2-5 grams.
    /// Uses prefix-based routing (all words starting with 'th' go to shard 'th').
    TwoChar,

    /// Adaptive: FirstChar for 1-grams, TwoChar for 2-5 grams.
    /// This matches Google Books file partitioning but creates many shards.
    Adaptive,

    /// Custom prefix length (1-4 characters).
    /// Uses prefix-based routing with specified prefix length.
    Custom {
        /// Number of prefix characters to use for sharding.
        prefix_len: usize,
    },

    /// CPU-proportional sharding with consistent hashing.
    ///
    /// Creates `max(num_cpus * multiplier, minimum)` shards and distributes
    /// n-grams using hash-based routing. This is the recommended option as it:
    /// - Reduces file overhead (16-64 files instead of 1400+)
    /// - Keeps all shards in memory (no LRU eviction needed)
    /// - Maintains parallel write benefits without lock contention
    CpuProportional {
        /// Multiplier for CPU count (default: 2).
        /// With 8 cores, multiplier=2 creates 16 shards.
        multiplier: usize,
        /// Minimum number of shards (default: 8).
        /// Ensures reasonable parallelism on low-core systems.
        minimum: usize,
    },
}

impl Default for ShardGranularity {
    fn default() -> Self {
        Self::CpuProportional {
            multiplier: 2,
            minimum: 8,
        }
    }
}

impl ShardGranularity {
    /// Get the prefix length for a given n-gram order.
    ///
    /// # Arguments
    ///
    /// * `order` - The n-gram order (1-5)
    ///
    /// # Returns
    ///
    /// The number of prefix characters to use for shard routing.
    /// Returns 0 for hash-based granularities (CpuProportional).
    pub fn prefix_len_for_order(&self, order: u8) -> usize {
        match self {
            Self::FirstChar => 1,
            Self::TwoChar => 2,
            Self::Adaptive => {
                if order == 1 {
                    1 // 1-grams: 26 shards
                } else {
                    2 // 2-5 grams: 676 shards
                }
            }
            Self::Custom { prefix_len } => *prefix_len,
            Self::CpuProportional { .. } => 0, // Hash-based, no prefix
        }
    }

    /// Get the maximum number of shards for this granularity.
    pub fn max_shards(&self) -> usize {
        match self {
            Self::FirstChar => 26,
            Self::TwoChar => 676,
            Self::Adaptive => 676, // Max across all orders
            Self::Custom { prefix_len } => 26_usize.pow(*prefix_len as u32),
            Self::CpuProportional {
                multiplier,
                minimum,
            } => Self::compute_cpu_proportional_shards(*multiplier, *minimum),
        }
    }

    /// Get the actual number of shards that will be created.
    ///
    /// For prefix-based granularities, this equals `max_shards()`.
    /// For `CpuProportional`, this is computed based on available CPUs.
    pub fn num_shards(&self) -> usize {
        self.max_shards()
    }

    /// Check if this granularity uses hash-based routing.
    ///
    /// Hash-based routing distributes n-grams using consistent hashing
    /// rather than prefix matching.
    pub fn is_hash_based(&self) -> bool {
        matches!(self, Self::CpuProportional { .. })
    }

    /// Check if this granularity uses prefix-based routing.
    pub fn is_prefix_based(&self) -> bool {
        !self.is_hash_based()
    }

    /// Compute the number of shards for CpuProportional mode.
    fn compute_cpu_proportional_shards(multiplier: usize, minimum: usize) -> usize {
        let cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        (cpus * multiplier).max(minimum)
    }
}

/// Merge mode for combining shards after import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MergeMode {
    /// Merge only after import completes (default).
    /// Simplest mode with best import throughput.
    PostImport,

    /// Merge periodically during import when shards become full.
    /// Reduces peak memory but adds overhead.
    Periodic,

    /// Hierarchical merging with levels (L0 → L1 → L2 → ...).
    /// Best for very large imports (>100M n-grams).
    Hierarchical {
        /// Number of merge levels.
        levels: usize,
    },
}

impl Default for MergeMode {
    fn default() -> Self {
        Self::PostImport
    }
}

/// Configuration for shard merging.
#[derive(Clone, Debug)]
pub struct MergeConfig {
    /// Merge mode strategy.
    pub mode: MergeMode,

    /// For periodic mode: trigger merge when this many shards are "full".
    pub merge_trigger_shards: usize,

    /// Size threshold (entry count) to consider a shard "full".
    pub shard_full_threshold: u64,

    /// Parallelism level for merge operations.
    pub merge_parallelism: usize,

    /// Delete source shards after successful merge.
    pub cleanup_after_merge: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            mode: MergeMode::PostImport,
            merge_trigger_shards: 10,
            shard_full_threshold: 1_000_000, // 1M entries
            merge_parallelism: 4,
            cleanup_after_merge: true,
        }
    }
}

/// Configuration for sharded import.
#[derive(Clone, Debug)]
pub struct ShardConfig {
    /// Sharding granularity strategy.
    pub granularity: ShardGranularity,

    /// Base directory for shard files.
    /// Each shard creates a file: `{base_dir}/shard_{prefix}.artrie`
    pub shard_dir: PathBuf,

    /// Maximum concurrent shard writers.
    /// Usually matches `parallel_downloads` in GoogleBooksConfig.
    pub max_writers: usize,

    /// Maximum number of open shards (LRU eviction for others).
    /// Set to 0 for unlimited (not recommended for many shards).
    pub max_open_shards: usize,

    /// Per-shard memory budget in bytes (for buffer manager).
    /// Default: 64MB per shard.
    pub shard_memory_budget: usize,

    /// Checkpoint interval in milliseconds per shard.
    pub checkpoint_interval_ms: u64,

    /// Threshold for automatic sharding (estimated n-gram count).
    /// If estimated count exceeds this, sharding is enabled.
    pub auto_shard_threshold: u64,

    /// Merge configuration.
    pub merge: MergeConfig,

    /// Global resident-overlay heap budget across all simultaneously-resident
    /// shards, in bytes. `None` = unbounded (legacy). The checkpoint tail evicts
    /// each shard's cold overlay down to `budget / resident_shard_count`.
    pub overlay_budget_bytes: Option<usize>,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            granularity: ShardGranularity::default(),
            shard_dir: PathBuf::from("shards"),
            max_writers: 4,
            max_open_shards: 32, // Reduced from 100 to prevent OOM with many workers
            shard_memory_budget: 64 * 1024 * 1024, // 64MB
            checkpoint_interval_ms: 30_000, // 30 seconds
            auto_shard_threshold: 10_000_000, // 10M n-grams
            merge: MergeConfig::default(),
            overlay_budget_bytes: None,
        }
    }
}

impl ShardConfig {
    /// Create a new shard configuration with the given shard directory.
    pub fn new(shard_dir: impl Into<PathBuf>) -> Self {
        Self {
            shard_dir: shard_dir.into(),
            ..Default::default()
        }
    }

    /// Set the global overlay-heap resident budget (bytes; `None` = unbounded).
    pub fn with_overlay_budget_bytes(mut self, budget: Option<usize>) -> Self {
        self.overlay_budget_bytes = budget;
        self
    }

    /// Build the per-shard overlay [`EvictionConfig`] from the global budget.
    ///
    /// The global budget is divided by the number of SIMULTANEOUSLY-RESIDENT
    /// shards: hash-based granularities (`CpuProportional`) and an unlimited
    /// `max_open_shards` keep all `num_shards` resident; otherwise the LRU cap
    /// bounds residents to `max_open_shards`. So `SUM(per-shard budget)` over the
    /// resident set ≈ the global budget, granularity-invariant. Returns `None`
    /// (unbounded overlay) when no budget is configured.
    pub fn overlay_eviction_config(&self) -> Option<EvictionConfig> {
        let global = self.overlay_budget_bytes?;
        let num_shards = self.granularity.num_shards().max(1);
        let resident = if self.granularity.is_hash_based() || self.max_open_shards == 0 {
            num_shards
        } else {
            self.max_open_shards.min(num_shards)
        };
        let per_shard = (global / resident).max(MIN_PER_SHARD_OVERLAY_BUDGET_BYTES);
        let mut config = EvictionConfig::without_memory_monitor();
        config.resident_budget_bytes = Some(per_shard);
        config.resident_budget_eviction_cap = Some(OVERLAY_EVICTION_CAP_NODES);
        Some(config)
    }

    /// Set the sharding granularity.
    pub fn with_granularity(mut self, granularity: ShardGranularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Set the maximum number of concurrent writers.
    pub fn with_max_writers(mut self, max_writers: usize) -> Self {
        self.max_writers = max_writers;
        self
    }

    /// Set the maximum number of open shards.
    pub fn with_max_open_shards(mut self, max_open_shards: usize) -> Self {
        self.max_open_shards = max_open_shards;
        self
    }

    /// Set the auto-shard threshold.
    pub fn with_auto_shard_threshold(mut self, threshold: u64) -> Self {
        self.auto_shard_threshold = threshold;
        self
    }

    /// Set the merge configuration.
    pub fn with_merge_config(mut self, merge: MergeConfig) -> Self {
        self.merge = merge;
        self
    }

    /// Get the path for a shard file given its prefix.
    pub fn shard_path(&self, prefix: &str) -> PathBuf {
        self.shard_dir.join(format!("shard_{}.artrie", prefix))
    }

    /// Get the path for the global checkpoint file.
    pub fn global_checkpoint_path(&self) -> PathBuf {
        self.shard_dir.join("global_checkpoint.json")
    }

    /// Check if the estimated n-gram count exceeds the auto-shard threshold.
    pub fn should_shard(&self, estimated_ngrams: u64) -> bool {
        estimated_ngrams >= self.auto_shard_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granularity_prefix_len() {
        assert_eq!(ShardGranularity::FirstChar.prefix_len_for_order(1), 1);
        assert_eq!(ShardGranularity::FirstChar.prefix_len_for_order(2), 1);

        assert_eq!(ShardGranularity::TwoChar.prefix_len_for_order(1), 2);
        assert_eq!(ShardGranularity::TwoChar.prefix_len_for_order(2), 2);

        // Adaptive: 1 for unigrams, 2 for higher orders
        assert_eq!(ShardGranularity::Adaptive.prefix_len_for_order(1), 1);
        assert_eq!(ShardGranularity::Adaptive.prefix_len_for_order(2), 2);
        assert_eq!(ShardGranularity::Adaptive.prefix_len_for_order(5), 2);

        assert_eq!(
            ShardGranularity::Custom { prefix_len: 3 }.prefix_len_for_order(1),
            3
        );

        // CpuProportional uses hash-based routing, so prefix_len is 0
        let cpu_prop = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };
        assert_eq!(cpu_prop.prefix_len_for_order(1), 0);
        assert_eq!(cpu_prop.prefix_len_for_order(5), 0);
    }

    #[test]
    fn test_granularity_max_shards() {
        assert_eq!(ShardGranularity::FirstChar.max_shards(), 26);
        assert_eq!(ShardGranularity::TwoChar.max_shards(), 676);
        assert_eq!(ShardGranularity::Adaptive.max_shards(), 676);
        assert_eq!(
            ShardGranularity::Custom { prefix_len: 3 }.max_shards(),
            17576
        );

        // CpuProportional depends on available CPUs
        let cpu_prop = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };
        let expected = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            * 2;
        assert_eq!(cpu_prop.max_shards(), expected.max(8));
    }

    #[test]
    fn test_cpu_proportional_minimum() {
        // With minimum=16, even on a 2-core system we get at least 16 shards
        let cpu_prop = ShardGranularity::CpuProportional {
            multiplier: 1,
            minimum: 16,
        };
        assert!(cpu_prop.max_shards() >= 16);
    }

    #[test]
    fn test_is_hash_based() {
        assert!(!ShardGranularity::FirstChar.is_hash_based());
        assert!(!ShardGranularity::TwoChar.is_hash_based());
        assert!(!ShardGranularity::Adaptive.is_hash_based());
        assert!(!ShardGranularity::Custom { prefix_len: 2 }.is_hash_based());

        let cpu_prop = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };
        assert!(cpu_prop.is_hash_based());
        assert!(!cpu_prop.is_prefix_based());
    }

    #[test]
    fn test_default_is_cpu_proportional() {
        let default = ShardGranularity::default();
        assert!(default.is_hash_based());
        assert!(matches!(
            default,
            ShardGranularity::CpuProportional {
                multiplier: 2,
                minimum: 8
            }
        ));
    }

    #[test]
    fn test_shard_config_paths() {
        let config = ShardConfig::new("/tmp/shards");
        assert_eq!(
            config.shard_path("th"),
            PathBuf::from("/tmp/shards/shard_th.artrie")
        );
        assert_eq!(
            config.global_checkpoint_path(),
            PathBuf::from("/tmp/shards/global_checkpoint.json")
        );
    }

    #[test]
    fn test_should_shard() {
        let config = ShardConfig::default();
        assert!(!config.should_shard(1_000_000)); // 1M < 10M
        assert!(config.should_shard(10_000_000)); // 10M >= 10M
        assert!(config.should_shard(100_000_000)); // 100M > 10M
    }
}
