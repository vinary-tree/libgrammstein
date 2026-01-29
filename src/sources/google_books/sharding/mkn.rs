//! Modified Kneser-Ney (MKN) statistics aggregation across shards.
//!
//! This module computes the statistics needed for Modified Kneser-Ney smoothing
//! across all shards in a sharded n-gram storage system.
//!
//! # MKN Statistics Overview
//!
//! Modified Kneser-Ney smoothing uses discount parameters computed from:
//!
//! - **n1**: Count of n-grams occurring exactly once
//! - **n2**: Count of n-grams occurring exactly twice
//! - **n3**: Count of n-grams occurring exactly 3 times
//! - **n4**: Count of n-grams occurring exactly 4 times
//!
//! From these, we compute three discount values:
//!
//! ```text
//! Y = n1 / (n1 + 2*n2)
//! D1 = 1 - 2*Y*(n2/n1)
//! D2 = 2 - 3*Y*(n3/n2)
//! D3+ = 3 - 4*Y*(n4/n3)
//! ```
//!
//! # Continuation Counts
//!
//! For lower-order interpolation, MKN uses continuation counts:
//!
//! - **N1+(•w)**: Number of unique words preceding w
//! - **N1+(w•)**: Number of unique words following w
//! - **N1+(•w•)**: Number of unique (preceding, following) word pairs for w
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::sources::google_books::sharding::mkn::{MknAggregator, MknStats};
//!
//! let aggregator = MknAggregator::new(&coordinator);
//! let stats = aggregator.compute_all()?;
//!
//! println!("Bigram discounts: D1={}, D2={}, D3+={}",
//!     stats.discounts[2].d1,
//!     stats.discounts[2].d2,
//!     stats.discounts[2].d3_plus);
//! ```

use super::coordinator::ShardCoordinator;
use crate::ngram::vocabulary::{decode_ngram_key, encode_indices_to_key, ngram_order};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Error type for MKN computation.
#[derive(Error, Debug)]
pub enum MknError {
    /// Shard access error.
    #[error("Shard error: {0}")]
    Shard(#[from] super::shard::ShardError),

    /// Coordinator error.
    #[error("Coordinator error: {0}")]
    Coordinator(#[from] super::coordinator::CoordinatorError),

    /// Insufficient data for statistics.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    /// Computation error.
    #[error("Computation error: {0}")]
    Computation(String),
}

/// Result type for MKN operations.
pub type MknResult<T> = Result<T, MknError>;

/// Frequency counts for a single order.
#[derive(Clone, Debug, Default)]
pub struct FrequencyCounts {
    /// Number of n-grams occurring exactly once.
    pub n1: u64,

    /// Number of n-grams occurring exactly twice.
    pub n2: u64,

    /// Number of n-grams occurring exactly 3 times.
    pub n3: u64,

    /// Number of n-grams occurring exactly 4 times.
    pub n4: u64,

    /// Total unique n-grams.
    pub total_unique: u64,

    /// Total n-gram occurrences (sum of all counts).
    pub total_count: u64,
}

impl FrequencyCounts {
    /// Add another FrequencyCounts to this one.
    ///
    /// This operation is proven associative and commutative in
    /// `formal/rocq/FrequencyCountsMerge.v`, enabling parallel tree reduction.
    pub fn merge(&mut self, other: &FrequencyCounts) {
        self.n1 += other.n1;
        self.n2 += other.n2;
        self.n3 += other.n3;
        self.n4 += other.n4;
        self.total_unique += other.total_unique;
        self.total_count += other.total_count;
    }
}

/// Atomic version of FrequencyCounts for parallel aggregation.
#[derive(Debug, Default)]
pub struct AtomicFrequencyCounts {
    pub n1: AtomicU64,
    pub n2: AtomicU64,
    pub n3: AtomicU64,
    pub n4: AtomicU64,
    pub total_unique: AtomicU64,
    pub total_count: AtomicU64,
}

impl AtomicFrequencyCounts {
    /// Add a count observation.
    pub fn observe(&self, count: u64) {
        match count {
            1 => self.n1.fetch_add(1, Ordering::Relaxed),
            2 => self.n2.fetch_add(1, Ordering::Relaxed),
            3 => self.n3.fetch_add(1, Ordering::Relaxed),
            4 => self.n4.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };
        self.total_unique.fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(count, Ordering::Relaxed);
    }

    /// Convert to non-atomic version.
    pub fn into_counts(self) -> FrequencyCounts {
        FrequencyCounts {
            n1: self.n1.into_inner(),
            n2: self.n2.into_inner(),
            n3: self.n3.into_inner(),
            n4: self.n4.into_inner(),
            total_unique: self.total_unique.into_inner(),
            total_count: self.total_count.into_inner(),
        }
    }

    /// Load into non-atomic version (without consuming).
    pub fn load(&self) -> FrequencyCounts {
        FrequencyCounts {
            n1: self.n1.load(Ordering::Relaxed),
            n2: self.n2.load(Ordering::Relaxed),
            n3: self.n3.load(Ordering::Relaxed),
            n4: self.n4.load(Ordering::Relaxed),
            total_unique: self.total_unique.load(Ordering::Relaxed),
            total_count: self.total_count.load(Ordering::Relaxed),
        }
    }
}

/// Discount parameters for a single order.
#[derive(Clone, Debug)]
pub struct DiscountParams {
    /// Discount for n-grams occurring once.
    pub d1: f64,

    /// Discount for n-grams occurring twice.
    pub d2: f64,

    /// Discount for n-grams occurring 3+ times.
    pub d3_plus: f64,

    /// Y parameter (intermediate value).
    pub y: f64,
}

impl Default for DiscountParams {
    fn default() -> Self {
        Self {
            d1: 0.5,
            d2: 0.75,
            d3_plus: 0.9,
            y: 0.5,
        }
    }
}

impl DiscountParams {
    /// Compute discount parameters from frequency counts.
    ///
    /// Returns default values if counts are insufficient.
    pub fn from_counts(counts: &FrequencyCounts) -> Self {
        // Need at least some data
        if counts.n1 == 0 || counts.n2 == 0 {
            log::warn!(
                "Insufficient data for MKN discounts (n1={}, n2={}), using defaults",
                counts.n1,
                counts.n2
            );
            return Self::default();
        }

        let n1 = counts.n1 as f64;
        let n2 = counts.n2 as f64;
        let n3 = counts.n3.max(1) as f64; // Avoid division by zero
        let n4 = counts.n4.max(1) as f64;

        // Y = n1 / (n1 + 2*n2)
        let y = n1 / (n1 + 2.0 * n2);

        // Discount parameters with clamping to valid ranges
        let d1 = (1.0 - 2.0 * y * (n2 / n1)).max(0.0).min(1.0);
        let d2 = (2.0 - 3.0 * y * (n3 / n2)).max(0.0).min(2.0);
        let d3_plus = (3.0 - 4.0 * y * (n4 / n3)).max(0.0).min(3.0);

        Self { d1, d2, d3_plus, y }
    }

    /// Get the appropriate discount for a count.
    pub fn discount_for(&self, count: u64) -> f64 {
        match count {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3_plus,
        }
    }
}

/// Continuation counts for lower-order interpolation.
#[derive(Clone, Debug, Default)]
pub struct ContinuationCounts {
    /// N1+(•w): unique predecessors for each context.
    /// Key: context (e.g., "quick brown"), Value: count of unique predecessors.
    pub predecessor_counts: HashMap<String, u64>,

    /// N1+(w•): unique successors for each context.
    /// Key: context (e.g., "the quick"), Value: count of unique successors.
    pub successor_counts: HashMap<String, u64>,

    /// Total unique continuation contexts.
    pub total_contexts: u64,
}

impl ContinuationCounts {
    /// Merge another ContinuationCounts into this one.
    pub fn merge(&mut self, other: ContinuationCounts) {
        for (k, v) in other.predecessor_counts {
            *self.predecessor_counts.entry(k).or_default() += v;
        }
        for (k, v) in other.successor_counts {
            *self.successor_counts.entry(k).or_default() += v;
        }
        self.total_contexts += other.total_contexts;
    }
}

/// Complete MKN statistics across all orders.
#[derive(Clone, Debug)]
pub struct MknStats {
    /// Frequency counts per order (index 0 = unused, 1-5 = orders).
    pub frequency_counts: Vec<FrequencyCounts>,

    /// Discount parameters per order.
    pub discounts: Vec<DiscountParams>,

    /// Continuation counts per order (for interpolation).
    pub continuation_counts: Vec<ContinuationCounts>,

    /// Maximum order computed.
    pub max_order: u8,
}

impl MknStats {
    /// Create empty stats for a given max order.
    pub fn new(max_order: u8) -> Self {
        let size = (max_order + 1) as usize;
        Self {
            frequency_counts: vec![FrequencyCounts::default(); size],
            discounts: vec![DiscountParams::default(); size],
            continuation_counts: vec![ContinuationCounts::default(); size],
            max_order,
        }
    }

    /// Get discount for a specific order and count.
    pub fn get_discount(&self, order: u8, count: u64) -> f64 {
        if order as usize >= self.discounts.len() {
            return 0.0;
        }
        self.discounts[order as usize].discount_for(count)
    }
}

/// Aggregator for computing MKN statistics across shards.
pub struct MknAggregator<'a> {
    /// Reference to the coordinator.
    coordinator: &'a ShardCoordinator,

    /// Whether to compute continuation counts (more expensive).
    compute_continuations: bool,
}

impl<'a> MknAggregator<'a> {
    /// Create a new aggregator.
    pub fn new(coordinator: &'a ShardCoordinator) -> Self {
        Self {
            coordinator,
            compute_continuations: false,
        }
    }

    /// Enable continuation count computation.
    pub fn with_continuations(mut self) -> Self {
        self.compute_continuations = true;
        self
    }

    /// Compute frequency counts for all orders.
    ///
    /// This is a single-pass operation that iterates through all shards.
    ///
    /// # Note on Parallel Reduction
    ///
    /// `formal/rocq/FrequencyCountsMerge.v` proves that merge is associative
    /// and commutative, enabling parallel tree reduction. However, benchmarking
    /// showed that for typical dataset sizes, the atomic approach is faster
    /// due to lower overhead. Tree reduction would benefit with millions of
    /// n-grams across hundreds of shards causing significant cache contention.
    pub fn compute_frequency_counts(&self) -> MknResult<Vec<FrequencyCounts>> {
        let max_order = 5u8;
        let counts: Vec<AtomicFrequencyCounts> = (0..=max_order)
            .map(|_| AtomicFrequencyCounts::default())
            .collect();

        // Discover and iterate through all shards on disk (not just in-memory cached ones)
        let shard_files = self.coordinator.discover_shard_files()
            .map_err(|e| MknError::Coordinator(e))?;
        let shard_keys: Vec<_> = shard_files.into_iter().map(|(key, _)| key).collect();

        shard_keys.par_iter().for_each(|key| {
            if let Ok(shard) = self.coordinator.get_or_create_shard(key) {
                let guard = shard.read();
                match guard.iter_with_counts() {
                    Ok(iter) => {
                        for (ngram, count) in iter {
                            // Skip metadata keys (they start with \x00)
                            if ngram.starts_with('\x00') {
                                continue;
                            }
                            let order = ngram_order(&ngram);
                            if order <= max_order {
                                counts[order as usize].observe(count);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to iterate shard {}: {}", key, e);
                    }
                }
            }
        });

        Ok(counts.into_iter().map(|c| c.into_counts()).collect())
    }

    /// Compute discount parameters from frequency counts.
    pub fn compute_discounts(&self, freq_counts: &[FrequencyCounts]) -> Vec<DiscountParams> {
        freq_counts
            .iter()
            .map(DiscountParams::from_counts)
            .collect()
    }

    /// Compute continuation counts for interpolation.
    ///
    /// This requires a second pass through the data to track unique
    /// predecessors and successors.
    pub fn compute_continuation_counts(&self) -> MknResult<Vec<ContinuationCounts>> {
        let max_order = 5u8;
        let mut all_counts: Vec<ContinuationCounts> = (0..=max_order)
            .map(|_| ContinuationCounts::default())
            .collect();

        // For each order, track unique predecessor/successor sets
        // This is memory-intensive, so we process order by order
        for order in 2..=max_order {
            let counts = self.compute_continuation_counts_for_order(order)?;
            all_counts[order as usize] = counts;
        }

        Ok(all_counts)
    }

    /// Compute continuation counts for a specific order.
    ///
    /// N-grams are stored as varint-encoded Latin-1 strings (LEB128 encoding).
    /// This function properly decodes them to extract word indices for
    /// computing predecessor and successor contexts.
    fn compute_continuation_counts_for_order(&self, order: u8) -> MknResult<ContinuationCounts> {
        // Track unique predecessors/successors for each context
        // Context for order n is the (n-1)-gram prefix/suffix
        //
        // We store contexts as varint-encoded keys (same format as n-gram keys)
        // and predecessors/successors as u64 indices for efficient comparison.
        let mut predecessor_sets: HashMap<String, HashSet<u64>> = HashMap::new();
        let mut successor_sets: HashMap<String, HashSet<u64>> = HashMap::new();

        // Discover all shards on disk (not just in-memory cached ones)
        let shard_files = self.coordinator.discover_shard_files()
            .map_err(|e| MknError::Coordinator(e))?;
        let shard_keys: Vec<_> = shard_files.into_iter().map(|(key, _)| key).collect();

        for key in &shard_keys {
            if let Ok(shard) = self.coordinator.get_or_create_shard(key) {
                let guard = shard.read();
                let iter = match guard.iter_with_counts() {
                    Ok(iter) => iter,
                    Err(e) => {
                        log::warn!("Failed to iterate shard {}: {}", key, e);
                        continue;
                    }
                };
                for (ngram, _count) in iter {
                    // Skip metadata keys (they start with \x00)
                    if ngram.starts_with('\x00') {
                        continue;
                    }

                    // Decode varint-encoded key to word indices
                    let indices = decode_ngram_key(&ngram);
                    if indices.len() as u8 != order || indices.len() < 2 {
                        continue;
                    }

                    // Predecessor context: all but first index (varint-encoded)
                    // e.g., indices [0, 1, 2] → predecessor=0, context=encode([1, 2])
                    let predecessor = indices[0];
                    let pred_context = encode_indices_to_key(&indices[1..]);
                    predecessor_sets
                        .entry(pred_context)
                        .or_default()
                        .insert(predecessor);

                    // Successor context: all but last index (varint-encoded)
                    // e.g., indices [0, 1, 2] → context=encode([0, 1]), successor=2
                    let successor = indices[indices.len() - 1];
                    let succ_context = encode_indices_to_key(&indices[..indices.len() - 1]);
                    successor_sets
                        .entry(succ_context)
                        .or_default()
                        .insert(successor);
                }
            }
        }

        // Convert sets to counts
        let predecessor_counts: HashMap<String, u64> = predecessor_sets
            .into_iter()
            .map(|(k, v)| (k, v.len() as u64))
            .collect();

        let successor_counts: HashMap<String, u64> = successor_sets
            .into_iter()
            .map(|(k, v)| (k, v.len() as u64))
            .collect();

        let total_contexts = predecessor_counts.len() as u64 + successor_counts.len() as u64;

        Ok(ContinuationCounts {
            predecessor_counts,
            successor_counts,
            total_contexts,
        })
    }

    /// Compute all MKN statistics.
    ///
    /// This performs:
    /// 1. First pass: frequency counts
    /// 2. Compute discounts from counts
    /// 3. (Optional) Second pass: continuation counts
    pub fn compute_all(&self) -> MknResult<MknStats> {
        log::info!("Computing MKN frequency counts...");
        let frequency_counts = self.compute_frequency_counts()?;

        log::info!("Computing discount parameters...");
        let discounts = self.compute_discounts(&frequency_counts);

        let continuation_counts = if self.compute_continuations {
            log::info!("Computing continuation counts...");
            self.compute_continuation_counts()?
        } else {
            vec![ContinuationCounts::default(); 6]
        };

        let max_order = 5;

        Ok(MknStats {
            frequency_counts,
            discounts,
            continuation_counts,
            max_order,
        })
    }

    /// Compute frequency counts only (faster, for discount computation).
    pub fn compute_discounts_only(&self) -> MknResult<Vec<DiscountParams>> {
        let frequency_counts = self.compute_frequency_counts()?;
        Ok(self.compute_discounts(&frequency_counts))
    }
}

/// Summary statistics for logging/debugging.
#[derive(Clone, Debug)]
pub struct MknSummary {
    /// Per-order summaries.
    pub orders: Vec<OrderSummary>,
}

/// Summary for a single order.
#[derive(Clone, Debug)]
pub struct OrderSummary {
    /// N-gram order.
    pub order: u8,

    /// Total unique n-grams.
    pub unique_ngrams: u64,

    /// Total occurrences.
    pub total_count: u64,

    /// Discount parameters.
    pub discounts: DiscountParams,
}

impl MknStats {
    /// Generate a summary for logging.
    pub fn summary(&self) -> MknSummary {
        let orders = (1..=self.max_order)
            .map(|order| {
                let idx = order as usize;
                OrderSummary {
                    order,
                    unique_ngrams: self.frequency_counts[idx].total_unique,
                    total_count: self.frequency_counts[idx].total_count,
                    discounts: self.discounts[idx].clone(),
                }
            })
            .collect();

        MknSummary { orders }
    }

    /// Format as a table for display.
    pub fn format_table(&self) -> String {
        let mut lines = vec![
            "MKN Statistics Summary".to_string(),
            "======================".to_string(),
            format!(
                "{:>5} {:>12} {:>15} {:>8} {:>8} {:>8}",
                "Order", "Unique", "Total Count", "D1", "D2", "D3+"
            ),
            "-".repeat(60),
        ];

        for order in 1..=self.max_order {
            let idx = order as usize;
            let fc = &self.frequency_counts[idx];
            let d = &self.discounts[idx];
            lines.push(format!(
                "{:>5} {:>12} {:>15} {:>8.4} {:>8.4} {:>8.4}",
                order, fc.total_unique, fc.total_count, d.d1, d.d2, d.d3_plus
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::{ShardConfig, ShardGranularity};
    use crate::ngram::vocabulary::SharedVocabulary;
    use tempfile::TempDir;

    /// Create a test coordinator with varint-encoded n-grams.
    ///
    /// N-grams are properly encoded using the vocabulary, matching
    /// the production format (LEB128 varint-encoded Latin-1 strings).
    fn create_test_coordinator() -> (TempDir, ShardCoordinator, SharedVocabulary) {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let config = ShardConfig::new(dir.path().join("shards"))
            .with_granularity(ShardGranularity::TwoChar);

        let coordinator = ShardCoordinator::new(config).expect("Failed to create coordinator");

        // Create vocabulary for encoding
        let vocab_path = dir.path().join("vocab.artrie");
        let vocab = SharedVocabulary::create(&vocab_path).expect("Failed to create vocab");

        // Helper to encode n-gram using vocabulary
        let encode = |words: &[&str]| -> String {
            let mut buf = Vec::with_capacity(words.len() * 2);
            for word in words {
                let idx = vocab.get_or_insert(word);
                crate::ngram::vocabulary::encode_varint(idx, &mut buf);
            }
            buf.into_iter().map(|b| char::from(b)).collect()
        };

        // Add test data with various counts
        // 1-grams (encoded as single varint)
        coordinator.store_ngram(&encode(&["the"]), 100).expect("store");
        coordinator.store_ngram(&encode(&["a"]), 50).expect("store");
        coordinator.store_ngram(&encode(&["an"]), 1).expect("store"); // count=1
        coordinator.store_ngram(&encode(&["is"]), 2).expect("store"); // count=2
        coordinator.store_ngram(&encode(&["at"]), 3).expect("store"); // count=3
        coordinator.store_ngram(&encode(&["in"]), 4).expect("store"); // count=4

        // 2-grams (encoded as two varints)
        coordinator.store_ngram(&encode(&["the", "quick"]), 10).expect("store");
        coordinator.store_ngram(&encode(&["the", "slow"]), 5).expect("store");
        coordinator.store_ngram(&encode(&["a", "big"]), 1).expect("store"); // count=1
        coordinator.store_ngram(&encode(&["a", "small"]), 2).expect("store"); // count=2
        coordinator.store_ngram(&encode(&["is", "very"]), 3).expect("store"); // count=3
        coordinator.store_ngram(&encode(&["in", "the"]), 4).expect("store"); // count=4

        // 3-grams (encoded as three varints)
        coordinator
            .store_ngram(&encode(&["the", "quick", "brown"]), 5)
            .expect("store");
        coordinator.store_ngram(&encode(&["the", "quick", "red"]), 1).expect("store"); // count=1
        coordinator.store_ngram(&encode(&["the", "slow", "green"]), 2).expect("store"); // count=2

        (dir, coordinator, vocab)
    }

    #[test]
    fn test_frequency_counts() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let aggregator = MknAggregator::new(&coordinator);

        let counts = aggregator.compute_frequency_counts().expect("compute");

        // Check 1-gram counts
        assert_eq!(counts[1].total_unique, 6);
        assert!(counts[1].n1 >= 1); // "an" with count 1
        assert!(counts[1].n2 >= 1); // "is" with count 2

        // Check 2-gram counts
        assert_eq!(counts[2].total_unique, 6);

        // Check 3-gram counts
        assert_eq!(counts[3].total_unique, 3);
    }

    #[test]
    fn test_discount_computation() {
        let counts = FrequencyCounts {
            n1: 100,
            n2: 50,
            n3: 30,
            n4: 20,
            total_unique: 200,
            total_count: 1000,
        };

        let discounts = DiscountParams::from_counts(&counts);

        // Y = 100 / (100 + 2*50) = 0.5
        assert!((discounts.y - 0.5).abs() < 0.01);

        // D1 = 1 - 2*0.5*(50/100) = 1 - 0.5 = 0.5
        assert!((discounts.d1 - 0.5).abs() < 0.01);

        // D2 = 2 - 3*0.5*(30/50) = 2 - 0.9 = 1.1
        assert!((discounts.d2 - 1.1).abs() < 0.01);

        // D3+ = 3 - 4*0.5*(20/30) = 3 - 1.333... ≈ 1.667
        assert!((discounts.d3_plus - 1.667).abs() < 0.01);
    }

    #[test]
    fn test_discount_default_on_insufficient_data() {
        let counts = FrequencyCounts {
            n1: 0,
            n2: 0,
            n3: 0,
            n4: 0,
            total_unique: 0,
            total_count: 0,
        };

        let discounts = DiscountParams::from_counts(&counts);

        // Should return defaults
        assert_eq!(discounts.d1, 0.5);
        assert_eq!(discounts.d2, 0.75);
        assert_eq!(discounts.d3_plus, 0.9);
    }

    #[test]
    fn test_continuation_counts() {
        let (_dir, coordinator, vocab) = create_test_coordinator();
        let aggregator = MknAggregator::new(&coordinator).with_continuations();

        let counts = aggregator
            .compute_continuation_counts_for_order(2)
            .expect("compute");

        // "the quick" and "the slow" share predecessor "the" for contexts "quick" and "slow"
        // So we should have entries in predecessor_counts
        //
        // Note: contexts are now varint-encoded Latin-1 strings, not raw words

        // Get the index for "quick" to check predecessor counts
        let quick_idx = vocab.get("quick").expect("quick should be in vocab");
        let quick_key = encode_indices_to_key(&[quick_idx]);
        assert!(
            counts.predecessor_counts.contains_key(&quick_key),
            "Should have predecessor count for context 'quick' (encoded as {:?})",
            quick_key
        );

        // Get the index for "the" to check successor counts
        let the_idx = vocab.get("the").expect("the should be in vocab");
        let the_key = encode_indices_to_key(&[the_idx]);
        assert!(
            counts.successor_counts.contains_key(&the_key),
            "Should have successor count for context 'the' (encoded as {:?})",
            the_key
        );
        // "the" should have 2 unique successors: "quick" and "slow"
        assert_eq!(*counts.successor_counts.get(&the_key).unwrap(), 2);
    }

    #[test]
    fn test_compute_all() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let aggregator = MknAggregator::new(&coordinator);

        let stats = aggregator.compute_all().expect("compute");

        // Check we have stats for all orders
        assert_eq!(stats.max_order, 5);
        assert_eq!(stats.frequency_counts.len(), 6); // 0-5
        assert_eq!(stats.discounts.len(), 6);

        // Verify non-zero counts for orders 1-3
        assert!(stats.frequency_counts[1].total_unique > 0);
        assert!(stats.frequency_counts[2].total_unique > 0);
        assert!(stats.frequency_counts[3].total_unique > 0);
    }

    #[test]
    fn test_format_table() {
        let (_dir, coordinator, _vocab) = create_test_coordinator();
        let aggregator = MknAggregator::new(&coordinator);

        let stats = aggregator.compute_all().expect("compute");
        let table = stats.format_table();

        assert!(table.contains("MKN Statistics Summary"));
        assert!(table.contains("Order"));
        assert!(table.contains("Unique"));
    }

    #[test]
    fn test_discount_for_count() {
        let discounts = DiscountParams {
            d1: 0.5,
            d2: 1.1,
            d3_plus: 1.7,
            y: 0.5,
        };

        assert_eq!(discounts.discount_for(0), 0.0);
        assert_eq!(discounts.discount_for(1), 0.5);
        assert_eq!(discounts.discount_for(2), 1.1);
        assert_eq!(discounts.discount_for(3), 1.7);
        assert_eq!(discounts.discount_for(100), 1.7);
    }

    #[test]
    fn test_atomic_frequency_counts() {
        let atomic = AtomicFrequencyCounts::default();

        atomic.observe(1);
        atomic.observe(1);
        atomic.observe(2);
        atomic.observe(3);
        atomic.observe(4);
        atomic.observe(100);

        let counts = atomic.into_counts();

        assert_eq!(counts.n1, 2);
        assert_eq!(counts.n2, 1);
        assert_eq!(counts.n3, 1);
        assert_eq!(counts.n4, 1);
        assert_eq!(counts.total_unique, 6);
        assert_eq!(counts.total_count, 1 + 1 + 2 + 3 + 4 + 100);
    }
}
