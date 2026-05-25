//! Deduplication for corpus sentences.
//!
//! This module provides configurable deduplication to remove duplicate
//! sentences from training corpora. Three deduplication modes are supported:
//!
//! - **Exact**: Only removes exact duplicates
//! - **Normalized**: Removes duplicates after normalization (lowercase, no punctuation)
//! - **MinHash**: Fuzzy deduplication using locality-sensitive hashing
//!
//! # Example
//!
//! ```ignore
//! use libgrammstein::corpus::Deduplicator;
//!
//! let mut dedup = Deduplicator::new(DeduplicationMode::Normalized);
//!
//! // First occurrence is kept
//! assert!(dedup.is_unique("Hello world!"));
//!
//! // Exact duplicate
//! assert!(!dedup.is_unique("Hello world!"));
//!
//! // Normalized duplicate (same after lowercasing)
//! assert!(!dedup.is_unique("HELLO WORLD!"));
//! ```

use std::collections::HashSet;

use gxhash::GxBuildHasher;

/// Deduplication mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeduplicationMode {
    /// Only remove exact duplicates.
    Exact,
    /// Remove duplicates after normalization (lowercase, remove punctuation).
    Normalized,
    /// Fuzzy deduplication using MinHash (Jaccard similarity threshold).
    MinHash {
        /// Number of hash functions to use.
        num_hashes: usize,
        /// Minimum Jaccard similarity to consider as duplicate.
        threshold: f32,
        /// N-gram size for shingling.
        shingle_size: usize,
    },
}

impl Default for DeduplicationMode {
    fn default() -> Self {
        Self::Normalized
    }
}

/// Deduplicator for removing duplicate sentences from a corpus.
///
/// The deduplicator maintains a set of seen sentence hashes and can operate
/// in different modes depending on how strict the deduplication should be.
pub struct Deduplicator {
    /// Set of seen sentence hashes.
    seen: HashSet<u64, GxBuildHasher>,
    /// Deduplication mode.
    mode: DeduplicationMode,
    /// MinHash signatures for fuzzy deduplication (when mode is MinHash).
    minhash_signatures: Vec<MinHashSignature>,
    /// Statistics.
    stats: DeduplicationStats,
}

/// MinHash signature for a sentence.
struct MinHashSignature {
    /// The minimum hash values for each hash function.
    hashes: Vec<u64>,
}

impl MinHashSignature {
    /// Create a new MinHash signature from shingles.
    fn from_shingles(shingles: &[u64], num_hashes: usize) -> Self {
        let mut hashes = vec![u64::MAX; num_hashes];

        for &shingle in shingles {
            for (i, hash) in hashes.iter_mut().enumerate() {
                // Create different hash functions by XORing with the function index
                let h = Self::hash_with_seed(shingle, i as u64);
                if h < *hash {
                    *hash = h;
                }
            }
        }

        Self { hashes }
    }

    /// Hash a value with a seed to create different hash functions.
    ///
    /// Uses standard library hasher to avoid gxhash's SIMD buffer overflows
    /// when hashing small inputs (u64 is only 8 bytes).
    fn hash_with_seed(value: u64, seed: u64) -> u64 {
        use std::hash::{DefaultHasher, Hasher};
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(seed);
        hasher.write_u64(value);
        hasher.finish()
    }

    /// Compute approximate Jaccard similarity between two signatures.
    fn jaccard_similarity(&self, other: &MinHashSignature) -> f32 {
        if self.hashes.len() != other.hashes.len() {
            return 0.0;
        }

        let matching = self
            .hashes
            .iter()
            .zip(other.hashes.iter())
            .filter(|(a, b)| a == b)
            .count();

        matching as f32 / self.hashes.len() as f32
    }
}

/// Statistics about deduplication.
#[derive(Debug, Clone, Default)]
pub struct DeduplicationStats {
    /// Total sentences processed.
    pub total: usize,
    /// Unique sentences kept.
    pub unique: usize,
    /// Duplicate sentences removed.
    pub duplicates: usize,
}

impl DeduplicationStats {
    /// Get the deduplication rate as a percentage.
    pub fn dedup_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.duplicates as f64 / self.total as f64
        }
    }
}

impl std::fmt::Display for DeduplicationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Deduplication Statistics:")?;
        writeln!(f, "  Total:      {}", self.total)?;
        writeln!(f, "  Unique:     {}", self.unique)?;
        writeln!(
            f,
            "  Duplicates: {} ({:.1}%)",
            self.duplicates,
            self.dedup_rate()
        )?;
        Ok(())
    }
}

impl Deduplicator {
    /// Create a new deduplicator with the given mode.
    pub fn new(mode: DeduplicationMode) -> Self {
        Self {
            seen: HashSet::with_hasher(GxBuildHasher::default()),
            mode,
            minhash_signatures: Vec::new(),
            stats: DeduplicationStats::default(),
        }
    }

    /// Create a deduplicator that removes exact duplicates only.
    pub fn exact() -> Self {
        Self::new(DeduplicationMode::Exact)
    }

    /// Create a deduplicator that normalizes before comparing.
    pub fn normalized() -> Self {
        Self::new(DeduplicationMode::Normalized)
    }

    /// Create a deduplicator that uses MinHash for fuzzy deduplication.
    ///
    /// # Arguments
    ///
    /// * `num_hashes` - Number of hash functions (more = more accurate, but slower)
    /// * `threshold` - Jaccard similarity threshold (0.0-1.0, higher = stricter)
    /// * `shingle_size` - Size of character n-grams for shingling
    pub fn minhash(num_hashes: usize, threshold: f32, shingle_size: usize) -> Self {
        Self::new(DeduplicationMode::MinHash {
            num_hashes,
            threshold,
            shingle_size,
        })
    }

    /// Create a deduplicator with default MinHash settings.
    ///
    /// Uses 128 hash functions, 0.8 threshold, and 3-gram shingles.
    pub fn minhash_default() -> Self {
        Self::minhash(128, 0.8, 3)
    }

    /// Check if a sentence is unique (not seen before).
    ///
    /// Returns `true` if the sentence is unique and should be kept,
    /// `false` if it's a duplicate.
    pub fn is_unique(&mut self, sentence: &str) -> bool {
        self.stats.total += 1;

        let is_unique = match &self.mode {
            DeduplicationMode::Exact => self.check_exact(sentence),
            DeduplicationMode::Normalized => self.check_normalized(sentence),
            DeduplicationMode::MinHash {
                num_hashes,
                threshold,
                shingle_size,
            } => self.check_minhash(sentence, *num_hashes, *threshold, *shingle_size),
        };

        if is_unique {
            self.stats.unique += 1;
        } else {
            self.stats.duplicates += 1;
        }

        is_unique
    }

    /// Check for exact duplicate.
    fn check_exact(&mut self, sentence: &str) -> bool {
        let hash = Self::hash_string(sentence);
        self.seen.insert(hash)
    }

    /// Check for normalized duplicate.
    fn check_normalized(&mut self, sentence: &str) -> bool {
        let normalized = Self::normalize(sentence);
        let hash = Self::hash_string(&normalized);
        self.seen.insert(hash)
    }

    /// Check for MinHash duplicate.
    fn check_minhash(
        &mut self,
        sentence: &str,
        num_hashes: usize,
        threshold: f32,
        shingle_size: usize,
    ) -> bool {
        let shingles = Self::create_shingles(sentence, shingle_size);
        if shingles.is_empty() {
            return false;
        }

        let signature = MinHashSignature::from_shingles(&shingles, num_hashes);

        // Check against existing signatures
        for existing in &self.minhash_signatures {
            if signature.jaccard_similarity(existing) >= threshold {
                return false; // Found a near-duplicate
            }
        }

        // No duplicate found, add this signature
        self.minhash_signatures.push(signature);
        true
    }

    /// Hash a string using safe_hash (gxhash for long strings, FNV-1a for short).
    fn hash_string(s: &str) -> u64 {
        crate::util::hash::safe_hash(s.as_bytes())
    }

    /// Normalize a sentence for comparison.
    fn normalize(sentence: &str) -> String {
        sentence
            .chars()
            .filter_map(|c| {
                if c.is_alphabetic() {
                    Some(c.to_lowercase().next().unwrap_or(c))
                } else if c.is_whitespace() {
                    Some(' ')
                } else if c.is_numeric() {
                    Some(c)
                } else {
                    None
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Create character shingles (n-grams) from a sentence.
    fn create_shingles(sentence: &str, n: usize) -> Vec<u64> {
        let normalized = Self::normalize(sentence);
        let chars: Vec<char> = normalized.chars().collect();

        if chars.len() < n {
            return vec![];
        }

        let mut shingles = Vec::with_capacity(chars.len() - n + 1);

        for window in chars.windows(n) {
            let shingle: String = window.iter().collect();
            shingles.push(Self::hash_string(&shingle));
        }

        shingles
    }

    /// Filter an iterator of sentences, keeping only unique ones.
    pub fn filter<'a, I>(&'a mut self, sentences: I) -> impl Iterator<Item = String> + 'a
    where
        I: Iterator<Item = String> + 'a,
    {
        sentences.filter(move |s| self.is_unique(s))
    }

    /// Get deduplication statistics.
    pub fn stats(&self) -> &DeduplicationStats {
        &self.stats
    }

    /// Reset the deduplicator, clearing all seen sentences.
    pub fn reset(&mut self) {
        self.seen.clear();
        self.minhash_signatures.clear();
        self.stats = DeduplicationStats::default();
    }

    /// Get the current memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let seen_size = self.seen.capacity() * std::mem::size_of::<u64>();
        let minhash_size = self
            .minhash_signatures
            .iter()
            .map(|s| s.hashes.len() * 8)
            .sum::<usize>();
        seen_size + minhash_size
    }
}

/// Builder for creating a Deduplicator with custom configuration.
#[derive(Debug, Clone)]
pub struct DeduplicatorBuilder {
    mode: DeduplicationMode,
    initial_capacity: Option<usize>,
}

impl DeduplicatorBuilder {
    /// Create a new builder with normalized mode.
    pub fn new() -> Self {
        Self {
            mode: DeduplicationMode::Normalized,
            initial_capacity: None,
        }
    }

    /// Set exact deduplication mode.
    pub fn exact(mut self) -> Self {
        self.mode = DeduplicationMode::Exact;
        self
    }

    /// Set normalized deduplication mode.
    pub fn normalized(mut self) -> Self {
        self.mode = DeduplicationMode::Normalized;
        self
    }

    /// Set MinHash deduplication mode with custom parameters.
    pub fn minhash(mut self, num_hashes: usize, threshold: f32, shingle_size: usize) -> Self {
        self.mode = DeduplicationMode::MinHash {
            num_hashes,
            threshold,
            shingle_size,
        };
        self
    }

    /// Set initial capacity for the hash set.
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = Some(capacity);
        self
    }

    /// Build the deduplicator.
    pub fn build(self) -> Deduplicator {
        let mut dedup = Deduplicator::new(self.mode);
        if let Some(cap) = self.initial_capacity {
            dedup.seen.reserve(cap);
        }
        dedup
    }
}

impl Default for DeduplicatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_dedup() {
        let mut dedup = Deduplicator::exact();

        // First occurrence should be unique
        assert!(dedup.is_unique("Hello world!"));

        // Exact duplicate should not be unique
        assert!(!dedup.is_unique("Hello world!"));

        // Different case should be unique (exact mode)
        assert!(dedup.is_unique("HELLO WORLD!"));

        // Different text should be unique
        assert!(dedup.is_unique("Goodbye world!"));
    }

    #[test]
    fn test_normalized_dedup() {
        let mut dedup = Deduplicator::normalized();

        // First occurrence should be unique
        assert!(dedup.is_unique("Hello world!"));

        // Exact duplicate should not be unique
        assert!(!dedup.is_unique("Hello world!"));

        // Different case should NOT be unique (normalized mode)
        assert!(!dedup.is_unique("HELLO WORLD!"));

        // Different punctuation should NOT be unique
        assert!(!dedup.is_unique("Hello, world."));

        // Different text should be unique
        assert!(dedup.is_unique("Goodbye world!"));
    }

    #[test]
    fn test_minhash_dedup() {
        let mut dedup = Deduplicator::minhash(64, 0.7, 3);

        // First occurrence should be unique
        assert!(dedup.is_unique("The quick brown fox jumps over the lazy dog."));

        // Very similar sentence should not be unique
        assert!(!dedup.is_unique("The quick brown fox jumps over a lazy dog."));

        // Different sentence should be unique
        assert!(dedup.is_unique("A completely different sentence about cats."));
    }

    #[test]
    fn test_normalization() {
        assert_eq!(Deduplicator::normalize("Hello World!"), "hello world");
        assert_eq!(
            Deduplicator::normalize("  Multiple   spaces  "),
            "multiple spaces"
        );
        assert_eq!(Deduplicator::normalize("Numbers123Here"), "numbers123here");
        assert_eq!(Deduplicator::normalize("!@#$%^&*()"), "");
    }

    #[test]
    fn test_shingle_creation() {
        let shingles = Deduplicator::create_shingles("hello", 3);
        assert_eq!(shingles.len(), 3); // "hel", "ell", "llo"

        let shingles = Deduplicator::create_shingles("hi", 3);
        assert!(shingles.is_empty()); // Too short
    }

    #[test]
    fn test_stats() {
        let mut dedup = Deduplicator::normalized();

        dedup.is_unique("Sentence one.");
        dedup.is_unique("Sentence two.");
        dedup.is_unique("Sentence one."); // Duplicate

        let stats = dedup.stats();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.unique, 2);
        assert_eq!(stats.duplicates, 1);
    }

    #[test]
    fn test_filter() {
        let mut dedup = Deduplicator::exact();

        let sentences = vec![
            "First".to_string(),
            "Second".to_string(),
            "First".to_string(), // Duplicate
            "Third".to_string(),
        ];

        let unique: Vec<String> = dedup.filter(sentences.into_iter()).collect();
        assert_eq!(unique, vec!["First", "Second", "Third"]);
    }

    #[test]
    fn test_reset() {
        let mut dedup = Deduplicator::normalized();

        dedup.is_unique("Hello");
        assert!(!dedup.is_unique("Hello"));

        dedup.reset();

        // After reset, same sentence should be unique again
        assert!(dedup.is_unique("Hello"));
    }

    #[test]
    fn test_builder() {
        let dedup = DeduplicatorBuilder::new()
            .normalized()
            .capacity(1000)
            .build();

        assert!(matches!(dedup.mode, DeduplicationMode::Normalized));
    }

    #[test]
    fn test_unicode_dedup() {
        let mut dedup = Deduplicator::normalized();

        // Chinese text
        assert!(dedup.is_unique("你好世界"));
        assert!(!dedup.is_unique("你好世界"));

        // Japanese text
        assert!(dedup.is_unique("こんにちは"));
        assert!(!dedup.is_unique("こんにちは"));

        // Mixed scripts
        assert!(dedup.is_unique("Hello 世界!"));
        assert!(!dedup.is_unique("HELLO 世界!"));
    }
}
