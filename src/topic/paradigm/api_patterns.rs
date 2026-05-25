//! API pattern mining using PrefixSpan algorithm.
//!
//! This module implements sequential pattern mining to discover
//! frequent API call sequences in code. The PrefixSpan algorithm
//! efficiently mines sequential patterns by recursively building
//! prefix-projected databases.
//!
//! # Algorithm
//!
//! PrefixSpan works by:
//! 1. Finding frequent 1-sequences (single API calls)
//! 2. For each frequent 1-sequence, projecting the database
//! 3. Recursively mining the projected database for longer patterns
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::topic::paradigm::{ApiPatternMiner, ApiPatternConfig};
//!
//! let config = ApiPatternConfig::default();
//! let miner = ApiPatternMiner::new(config);
//!
//! // Sequences of API calls
//! let sequences = vec![
//!     vec!["open", "read", "close"],
//!     vec!["open", "write", "close"],
//!     vec!["open", "read", "seek", "read", "close"],
//! ];
//!
//! let patterns = miner.mine(&sequences);
//! // Finds: ["open", "close"] with support 3
//! ```

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "serde-extras")]
use serde::{Deserialize, Serialize};

/// Configuration for API pattern mining.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct ApiPatternConfig {
    /// Minimum support threshold (fraction of sequences containing the pattern).
    pub min_support: f64,
    /// Minimum absolute support count.
    pub min_support_count: usize,
    /// Maximum pattern length to mine.
    pub max_pattern_length: usize,
    /// Minimum pattern length to report.
    pub min_pattern_length: usize,
    /// Whether to mine closed patterns only (no superpatterns with same support).
    pub closed_only: bool,
    /// Maximum number of patterns to return.
    pub max_patterns: usize,
    /// Whether to enable gap constraints (allow non-consecutive matches).
    pub allow_gaps: bool,
    /// Maximum gap size if gaps are allowed.
    pub max_gap: usize,
}

impl Default for ApiPatternConfig {
    fn default() -> Self {
        Self {
            min_support: 0.1,
            min_support_count: 2,
            max_pattern_length: 10,
            min_pattern_length: 2,
            closed_only: false,
            max_patterns: 1000,
            allow_gaps: true,
            max_gap: 3,
        }
    }
}

impl ApiPatternConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum support threshold.
    pub fn with_min_support(mut self, support: f64) -> Self {
        self.min_support = support;
        self
    }

    /// Set minimum support count.
    pub fn with_min_support_count(mut self, count: usize) -> Self {
        self.min_support_count = count;
        self
    }

    /// Set maximum pattern length.
    pub fn with_max_pattern_length(mut self, length: usize) -> Self {
        self.max_pattern_length = length;
        self
    }

    /// Set minimum pattern length.
    pub fn with_min_pattern_length(mut self, length: usize) -> Self {
        self.min_pattern_length = length;
        self
    }

    /// Enable/disable closed pattern mining.
    pub fn with_closed_only(mut self, closed: bool) -> Self {
        self.closed_only = closed;
        self
    }

    /// Set maximum gap size.
    pub fn with_max_gap(mut self, gap: usize) -> Self {
        self.max_gap = gap;
        self.allow_gaps = gap > 0;
        self
    }

    /// Configuration for strict (no gaps) mining.
    pub fn strict() -> Self {
        Self {
            allow_gaps: false,
            max_gap: 0,
            ..Default::default()
        }
    }

    /// Configuration for lenient mining (large gaps allowed).
    pub fn lenient() -> Self {
        Self {
            max_gap: 10,
            allow_gaps: true,
            min_support: 0.05,
            ..Default::default()
        }
    }
}

/// A mined API call pattern.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct ApiPattern {
    /// The sequence of API calls in the pattern.
    pub sequence: Vec<Arc<str>>,
    /// Number of sequences containing this pattern.
    pub support: usize,
    /// Support as a fraction of total sequences.
    pub support_ratio: f64,
    /// Whether this is a closed pattern.
    pub is_closed: bool,
    /// Average position in sequences (0.0 = start, 1.0 = end).
    pub avg_position: f64,
    /// Confidence if used as an association rule (A -> B).
    pub confidence: Option<f64>,
}

impl ApiPattern {
    /// Create a new pattern.
    pub fn new(sequence: Vec<Arc<str>>, support: usize, total_sequences: usize) -> Self {
        Self {
            sequence,
            support,
            support_ratio: support as f64 / total_sequences.max(1) as f64,
            is_closed: false,
            avg_position: 0.5,
            confidence: None,
        }
    }

    /// Get the pattern length.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if pattern is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    /// Get pattern as string representation.
    pub fn to_string_pattern(&self) -> String {
        self.sequence
            .iter()
            .map(|s| s.as_ref())
            .collect::<Vec<_>>()
            .join(" -> ")
    }
}

/// Internal representation of a sequence position.
#[derive(Clone, Debug)]
struct SequencePosition {
    /// Index of the sequence in the database.
    sequence_idx: usize,
    /// Position within the sequence.
    position: usize,
}

/// Projected database for PrefixSpan.
#[derive(Clone)]
struct ProjectedDatabase {
    /// Positions in the original database.
    positions: Vec<SequencePosition>,
    /// The prefix that created this projection.
    prefix: Vec<Arc<str>>,
}

/// API pattern miner using PrefixSpan algorithm.
#[derive(Clone)]
pub struct ApiPatternMiner {
    config: ApiPatternConfig,
    /// Interned strings for efficiency.
    string_cache: HashMap<String, Arc<str>>,
}

impl ApiPatternMiner {
    /// Create a new miner with the given configuration.
    pub fn new(config: ApiPatternConfig) -> Self {
        Self {
            config,
            string_cache: HashMap::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ApiPatternConfig::default())
    }

    /// Get or intern a string.
    fn intern(&mut self, s: &str) -> Arc<str> {
        if let Some(cached) = self.string_cache.get(s) {
            Arc::clone(cached)
        } else {
            let arc: Arc<str> = Arc::from(s);
            self.string_cache.insert(s.to_string(), Arc::clone(&arc));
            arc
        }
    }

    /// Mine patterns from API call sequences.
    pub fn mine<S, T>(&mut self, sequences: &[S]) -> Vec<ApiPattern>
    where
        S: AsRef<[T]>,
        T: AsRef<str>,
    {
        if sequences.is_empty() {
            return Vec::new();
        }

        let total_sequences = sequences.len();
        let min_support = ((self.config.min_support * total_sequences as f64).ceil() as usize)
            .max(self.config.min_support_count);

        // Convert sequences to interned form
        let interned_sequences: Vec<Vec<Arc<str>>> = sequences
            .iter()
            .map(|seq| {
                seq.as_ref()
                    .iter()
                    .map(|s| self.intern(s.as_ref()))
                    .collect()
            })
            .collect();

        // Find frequent 1-sequences
        let mut item_counts: HashMap<Arc<str>, usize> = HashMap::new();
        for seq in &interned_sequences {
            let mut seen = std::collections::HashSet::new();
            for item in seq {
                if seen.insert(Arc::clone(item)) {
                    *item_counts.entry(Arc::clone(item)).or_insert(0) += 1;
                }
            }
        }

        // Filter frequent items
        let frequent_items: Vec<Arc<str>> = item_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_support)
            .map(|(item, _)| item)
            .collect();

        if frequent_items.is_empty() {
            return Vec::new();
        }

        // Mine patterns recursively
        let mut patterns = Vec::new();

        for item in &frequent_items {
            // Create initial projected database
            let mut positions = Vec::new();
            for (seq_idx, seq) in interned_sequences.iter().enumerate() {
                for (pos, s) in seq.iter().enumerate() {
                    if s == item {
                        positions.push(SequencePosition {
                            sequence_idx: seq_idx,
                            position: pos + 1, // Start after this item
                        });
                        break; // Only first occurrence per sequence
                    }
                }
            }

            if positions.len() >= min_support {
                let prefix = vec![Arc::clone(item)];
                let projected = ProjectedDatabase { positions, prefix };

                self.prefix_span_recursive(
                    &interned_sequences,
                    projected,
                    min_support,
                    total_sequences,
                    &mut patterns,
                );
            }
        }

        // Sort by support (descending) then length (descending)
        patterns.sort_by(|a, b| {
            b.support
                .cmp(&a.support)
                .then_with(|| b.len().cmp(&a.len()))
        });

        // Limit number of patterns
        patterns.truncate(self.config.max_patterns);

        // Mark closed patterns if requested
        if self.config.closed_only {
            patterns = self.filter_closed_patterns(patterns);
        }

        patterns
    }

    /// Recursive PrefixSpan mining.
    fn prefix_span_recursive(
        &self,
        sequences: &[Vec<Arc<str>>],
        projected: ProjectedDatabase,
        min_support: usize,
        total_sequences: usize,
        patterns: &mut Vec<ApiPattern>,
    ) {
        // Add current prefix as a pattern if it meets length requirement
        if projected.prefix.len() >= self.config.min_pattern_length {
            let support = self.count_unique_sequences(&projected.positions);
            let pattern = ApiPattern::new(projected.prefix.clone(), support, total_sequences);
            patterns.push(pattern);
        }

        // Stop if max length reached
        if projected.prefix.len() >= self.config.max_pattern_length {
            return;
        }

        // Find frequent items in projected database
        let mut item_counts: HashMap<Arc<str>, Vec<SequencePosition>> = HashMap::new();

        for pos in &projected.positions {
            let seq = &sequences[pos.sequence_idx];
            let max_pos = if self.config.allow_gaps {
                (pos.position + self.config.max_gap + 1).min(seq.len())
            } else {
                (pos.position + 1).min(seq.len())
            };

            let mut seen_in_seq = std::collections::HashSet::new();
            for i in pos.position..max_pos {
                let item = &seq[i];
                if seen_in_seq.insert(Arc::clone(item)) {
                    item_counts
                        .entry(Arc::clone(item))
                        .or_default()
                        .push(SequencePosition {
                            sequence_idx: pos.sequence_idx,
                            position: i + 1,
                        });
                }
            }
        }

        // Recursively mine each frequent extension
        for (item, positions) in item_counts {
            let support = self.count_unique_sequences(&positions);
            if support >= min_support {
                let mut new_prefix = projected.prefix.clone();
                new_prefix.push(item);

                let new_projected = ProjectedDatabase {
                    positions,
                    prefix: new_prefix,
                };

                self.prefix_span_recursive(
                    sequences,
                    new_projected,
                    min_support,
                    total_sequences,
                    patterns,
                );
            }
        }
    }

    /// Count unique sequences in positions.
    fn count_unique_sequences(&self, positions: &[SequencePosition]) -> usize {
        let mut seen = std::collections::HashSet::new();
        for pos in positions {
            seen.insert(pos.sequence_idx);
        }
        seen.len()
    }

    /// Filter to only closed patterns.
    fn filter_closed_patterns(&self, patterns: Vec<ApiPattern>) -> Vec<ApiPattern> {
        let mut closed: Vec<ApiPattern> = Vec::new();

        'outer: for pattern in patterns {
            // Check if any superpattern has the same support
            for other in &closed {
                if other.support == pattern.support
                    && other.len() > pattern.len()
                    && self.is_subsequence(&pattern.sequence, &other.sequence)
                {
                    continue 'outer;
                }
            }

            // Also check upcoming patterns
            let support = pattern.support;
            let len = pattern.len();
            let seq = &pattern.sequence;

            closed.retain(|p: &ApiPattern| {
                !(p.support == support && p.len() < len && self.is_subsequence(&p.sequence, seq))
            });

            closed.push(pattern);
        }

        closed
    }

    /// Check if `sub` is a subsequence of `super_seq`.
    fn is_subsequence(&self, sub: &[Arc<str>], super_seq: &[Arc<str>]) -> bool {
        if sub.len() > super_seq.len() {
            return false;
        }

        let mut sub_idx = 0;
        for item in super_seq {
            if sub_idx < sub.len() && item == &sub[sub_idx] {
                sub_idx += 1;
            }
        }

        sub_idx == sub.len()
    }
}

impl std::fmt::Debug for ApiPatternMiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiPatternMiner")
            .field("config", &self.config)
            .field("cache_size", &self.string_cache.len())
            .finish()
    }
}

/// Mining statistics.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde-extras", derive(Serialize, Deserialize))]
pub struct MiningStats {
    /// Total sequences processed.
    pub sequences_processed: usize,
    /// Total items processed.
    pub items_processed: usize,
    /// Patterns found.
    pub patterns_found: usize,
    /// Mining time in microseconds.
    pub time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ApiPatternConfig::default();
        assert!(config.min_support > 0.0);
        assert!(config.max_pattern_length > 0);
    }

    #[test]
    fn test_config_builder() {
        let config = ApiPatternConfig::new()
            .with_min_support(0.2)
            .with_max_pattern_length(5)
            .with_max_gap(2);

        assert_eq!(config.min_support, 0.2);
        assert_eq!(config.max_pattern_length, 5);
        assert_eq!(config.max_gap, 2);
    }

    #[test]
    fn test_empty_sequences() {
        let mut miner = ApiPatternMiner::with_defaults();
        let sequences: Vec<Vec<&str>> = vec![];
        let patterns = miner.mine(&sequences);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_simple_pattern() {
        let mut miner = ApiPatternMiner::new(ApiPatternConfig {
            min_support: 0.5,
            min_support_count: 2,
            min_pattern_length: 2,
            ..Default::default()
        });

        let sequences = vec![
            vec!["open", "read", "close"],
            vec!["open", "write", "close"],
            vec!["open", "read", "seek", "close"],
        ];

        let patterns = miner.mine(&sequences);

        // Should find ["open", "close"] with support 3
        assert!(!patterns.is_empty());

        let open_close = patterns.iter().find(|p| {
            p.sequence.len() == 2
                && p.sequence[0].as_ref() == "open"
                && p.sequence[1].as_ref() == "close"
        });

        assert!(open_close.is_some());
        assert_eq!(open_close.unwrap().support, 3);
    }

    #[test]
    fn test_longer_patterns() {
        let mut miner = ApiPatternMiner::new(ApiPatternConfig {
            min_support: 0.5,
            min_support_count: 2,
            min_pattern_length: 3,
            max_pattern_length: 5,
            ..Default::default()
        });

        let sequences = vec![
            vec!["connect", "query", "fetch", "close"],
            vec!["connect", "query", "fetch", "close"],
            vec!["connect", "query", "fetch", "process", "close"],
        ];

        let patterns = miner.mine(&sequences);

        // Should find ["connect", "query", "fetch", "close"]
        let long_pattern = patterns.iter().find(|p| p.sequence.len() >= 4);

        assert!(long_pattern.is_some());
    }

    #[test]
    fn test_closed_patterns() {
        let mut miner = ApiPatternMiner::new(ApiPatternConfig {
            min_support: 0.5,
            min_support_count: 2,
            closed_only: true,
            ..Default::default()
        });

        let sequences = vec![
            vec!["a", "b", "c"],
            vec!["a", "b", "c"],
            vec!["a", "b", "c"],
        ];

        let patterns = miner.mine(&sequences);

        // With closed_only, should prefer longer pattern ["a", "b", "c"]
        // over shorter ones with same support
        let has_abc = patterns
            .iter()
            .any(|p| p.sequence.len() == 3 && p.sequence[0].as_ref() == "a");

        let has_ab = patterns.iter().any(|p| {
            p.sequence.len() == 2 && p.sequence[0].as_ref() == "a" && p.sequence[1].as_ref() == "b"
        });

        // The closed pattern is abc (length 3), not ab (length 2)
        assert!(has_abc);
        // ab should not be present as it's not closed (abc has same support)
        assert!(!has_ab);
    }

    #[test]
    fn test_strict_mining() {
        let mut miner = ApiPatternMiner::new(
            ApiPatternConfig::strict()
                .with_min_support(0.5)
                .with_min_support_count(2),
        );

        let sequences = vec![
            vec!["a", "x", "b"], // a-b not consecutive
            vec!["a", "b"],      // a-b consecutive
            vec!["a", "b"],      // a-b consecutive
        ];

        let patterns = miner.mine(&sequences);

        // With strict (no gaps), ["a", "b"] appears in 2 sequences
        let ab_pattern = patterns.iter().find(|p| {
            p.sequence.len() == 2 && p.sequence[0].as_ref() == "a" && p.sequence[1].as_ref() == "b"
        });

        // Should find it with support 2 (only consecutive matches)
        assert!(ab_pattern.is_some());
        assert_eq!(ab_pattern.unwrap().support, 2);
    }

    #[test]
    fn test_gap_mining() {
        let mut miner = ApiPatternMiner::new(ApiPatternConfig {
            min_support: 0.5,
            min_support_count: 2,
            allow_gaps: true,
            max_gap: 2,
            ..Default::default()
        });

        let sequences = vec![
            vec!["a", "x", "b"],      // a-b with gap 1
            vec!["a", "x", "y", "b"], // a-b with gap 2
            vec!["a", "b"],           // a-b consecutive
        ];

        let patterns = miner.mine(&sequences);

        // With gaps allowed, ["a", "b"] appears in all 3 sequences
        let ab_pattern = patterns.iter().find(|p| {
            p.sequence.len() == 2 && p.sequence[0].as_ref() == "a" && p.sequence[1].as_ref() == "b"
        });

        assert!(ab_pattern.is_some());
        assert_eq!(ab_pattern.unwrap().support, 3);
    }

    #[test]
    fn test_pattern_to_string() {
        let pattern = ApiPattern::new(
            vec![Arc::from("open"), Arc::from("read"), Arc::from("close")],
            10,
            100,
        );

        assert_eq!(pattern.to_string_pattern(), "open -> read -> close");
    }

    #[test]
    fn test_is_subsequence() {
        let miner = ApiPatternMiner::with_defaults();

        let sub: Vec<Arc<str>> = vec![Arc::from("a"), Arc::from("c")];
        let super_seq: Vec<Arc<str>> = vec![Arc::from("a"), Arc::from("b"), Arc::from("c")];

        assert!(miner.is_subsequence(&sub, &super_seq));
        assert!(!miner.is_subsequence(&super_seq, &sub));
    }

    #[test]
    fn test_real_api_sequences() {
        let mut miner = ApiPatternMiner::new(ApiPatternConfig {
            min_support: 0.3,
            min_support_count: 2,
            min_pattern_length: 2,
            ..Default::default()
        });

        // Simulated file I/O API sequences
        let sequences = vec![
            vec!["fopen", "fread", "fclose"],
            vec!["fopen", "fwrite", "fflush", "fclose"],
            vec!["fopen", "fread", "fseek", "fread", "fclose"],
            vec!["fopen", "fgets", "fclose"],
            vec!["fopen", "fprintf", "fclose"],
        ];

        let patterns = miner.mine(&sequences);

        // Should find ["fopen", "fclose"] in all sequences
        let open_close = patterns.iter().find(|p| {
            p.sequence.len() == 2
                && p.sequence[0].as_ref() == "fopen"
                && p.sequence[1].as_ref() == "fclose"
        });

        assert!(open_close.is_some());
        assert_eq!(open_close.unwrap().support, 5);
    }
}
