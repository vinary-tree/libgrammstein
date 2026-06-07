//! Shard routing logic for distributing n-grams across shards.
//!
//! N-grams are routed to shards using one of two strategies:
//!
//! 1. **Prefix-based routing** (FirstChar, TwoChar, Adaptive, Custom):
//!    Routes based on the first character(s) of the first word.
//!    Matches Google Books file partitioning.
//!
//! 2. **Hash-based routing** (CpuProportional):
//!    Routes using consistent hashing for even distribution across
//!    a CPU-proportional number of shards.

use super::config::ShardGranularity;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Unique identifier for a shard.
///
/// Shards are identified by their prefix (lowercase letters) and optionally
/// by n-gram order for order-specific sharding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardKey {
    /// Lowercase prefix (e.g., "th", "a", "zz").
    pub prefix: String,

    /// Optional n-gram order (1-5) for order-specific sharding.
    /// If None, the shard contains all orders for this prefix.
    pub order: Option<u8>,
}

impl ShardKey {
    /// Create a new shard key with the given prefix.
    ///
    /// Used for prefix-based routing (FirstChar, TwoChar, Adaptive, Custom).
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            order: None,
        }
    }

    /// Create a new shard key with prefix and order.
    pub fn with_order(prefix: impl Into<String>, order: u8) -> Self {
        Self {
            prefix: prefix.into(),
            order: Some(order),
        }
    }

    /// Create a shard key from a numeric index.
    ///
    /// Used for hash-based routing (CpuProportional).
    /// The index is zero-padded to 4 digits for consistent sorting.
    ///
    /// # Arguments
    ///
    /// * `index` - The shard index (0..num_shards)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let key = ShardKey::from_index(0);
    /// assert_eq!(key.prefix, "0000");
    ///
    /// let key = ShardKey::from_index(42);
    /// assert_eq!(key.prefix, "0042");
    /// ```
    pub fn from_index(index: usize) -> Self {
        Self {
            prefix: format!("{:04}", index),
            order: None,
        }
    }

    /// Check if this is an index-based shard key (from hash routing).
    pub fn is_index_based(&self) -> bool {
        self.prefix.chars().all(|c| c.is_ascii_digit())
    }

    /// Get the index if this is an index-based shard key.
    ///
    /// Returns `None` for prefix-based keys.
    pub fn as_index(&self) -> Option<usize> {
        if self.is_index_based() {
            self.prefix.parse().ok()
        } else {
            None
        }
    }

    /// Get a canonical string representation for file naming.
    pub fn as_file_stem(&self) -> String {
        match self.order {
            Some(order) => format!("{}_{}", self.prefix, order),
            None => self.prefix.clone(),
        }
    }
}

impl Hash for ShardKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prefix.hash(state);
        self.order.hash(state);
    }
}

impl fmt::Display for ShardKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.order {
            Some(order) => write!(f, "{}:{}", self.prefix, order),
            None => write!(f, "{}", self.prefix),
        }
    }
}

/// Compute the shard key for an n-gram.
///
/// For prefix-based granularities (FirstChar, TwoChar, Adaptive, Custom):
///   Extracts the first word from the n-gram and takes the first N characters
///   as the shard prefix.
///
/// For hash-based granularities (CpuProportional):
///   Uses consistent hashing to distribute n-grams evenly across shards.
///
/// # Arguments
///
/// * `ngram` - The n-gram string (pipe-separated tokens, e.g., "the|quick|brown")
/// * `order` - The n-gram order (1-5), used for Adaptive granularity
/// * `granularity` - The sharding granularity configuration
///
/// # Returns
///
/// A `ShardKey` identifying which shard should store this n-gram.
///
/// # Examples
///
/// ```ignore
/// use libgrammstein::sources::google_books::sharding::routing::compute_shard_key;
/// use libgrammstein::sources::google_books::sharding::config::ShardGranularity;
///
/// // Prefix-based routing
/// let key = compute_shard_key("the|quick|brown", 3, &ShardGranularity::TwoChar);
/// assert_eq!(key.prefix, "th");
///
/// // Hash-based routing (index-based key)
/// let g = ShardGranularity::CpuProportional { multiplier: 2, minimum: 8 };
/// let key = compute_shard_key("the|quick|brown", 3, &g);
/// assert!(key.is_index_based());
/// ```
pub fn compute_shard_key(ngram: &str, order: u8, granularity: &ShardGranularity) -> ShardKey {
    // Handle hash-based routing
    if let ShardGranularity::CpuProportional {
        multiplier: _,
        minimum: _,
    } = granularity
    {
        let num_shards = granularity.num_shards();
        let index = hash_to_shard(ngram, num_shards);
        return ShardKey::from_index(index);
    }

    // Prefix-based routing
    let prefix_len = granularity.prefix_len_for_order(order);

    // Extract the first word (before first pipe separator)
    let first_word = ngram.split('|').next().unwrap_or("");

    // Get prefix characters, lowercase
    let prefix: String = first_word
        .chars()
        .filter(|c| c.is_alphabetic())
        .take(prefix_len)
        .flat_map(|c| c.to_lowercase())
        .collect();

    // Handle edge cases: non-alphabetic or short words
    let prefix = if prefix.is_empty() {
        // Non-alphabetic first word (numbers, symbols) → use special shard
        "_".repeat(prefix_len)
    } else if prefix.len() < prefix_len {
        // Short word → pad with 'a' (keeps lexicographic ordering)
        format!("{:a<width$}", prefix, width = prefix_len)
    } else {
        prefix
    };

    ShardKey::new(prefix)
}

/// Hash an n-gram to a shard index using consistent hashing.
///
/// Uses the default hasher for deterministic distribution.
///
/// # Arguments
///
/// * `ngram` - The n-gram string to hash
/// * `num_shards` - Total number of shards
///
/// # Returns
///
/// The shard index (0..num_shards) for this n-gram.
fn hash_to_shard(ngram: &str, num_shards: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    ngram.hash(&mut hasher);
    (hasher.finish() as usize) % num_shards
}

/// Compute the shard key for a Google Books prefix file.
///
/// Google Books files are already partitioned by prefix (e.g., "th", "aa").
/// This function converts a file prefix to a shard key.
///
/// **Note**: For hash-based granularities (CpuProportional), this function
/// cannot determine a single shard key since n-grams from one file may be
/// distributed across multiple shards. In that case, use `compute_shard_key`
/// for each individual n-gram.
///
/// # Arguments
///
/// * `file_prefix` - The prefix from the Google Books file name (e.g., "th", "a")
/// * `order` - The n-gram order (1-5)
/// * `granularity` - The sharding granularity configuration
///
/// # Returns
///
/// A `ShardKey` for the shard that should store n-grams from this file.
/// For hash-based granularities, returns a key based on the file prefix itself.
pub fn shard_key_for_file_prefix(
    file_prefix: &str,
    order: u8,
    granularity: &ShardGranularity,
) -> ShardKey {
    // For hash-based routing, we can't map a file prefix to a single shard
    // since n-grams will be distributed by hash. Use the prefix directly.
    if granularity.is_hash_based() {
        // Hash the file prefix to get a representative shard
        let num_shards = granularity.num_shards();
        let index = hash_to_shard(file_prefix, num_shards);
        return ShardKey::from_index(index);
    }

    let target_len = granularity.prefix_len_for_order(order);
    let prefix = file_prefix.to_lowercase();

    // Normalize to target length
    let prefix = if prefix.len() < target_len {
        format!("{:a<width$}", prefix, width = target_len)
    } else if prefix.len() > target_len {
        prefix[..target_len].to_string()
    } else {
        prefix
    };

    ShardKey::new(prefix)
}

/// Generate all possible shard keys for a given granularity and order.
///
/// This is useful for pre-creating shards or iterating over all shards.
///
/// # Arguments
///
/// * `granularity` - The sharding granularity configuration
/// * `order` - The n-gram order (1-5)
///
/// # Returns
///
/// A vector of all valid shard keys.
pub fn all_shard_keys(granularity: &ShardGranularity, order: u8) -> Vec<ShardKey> {
    // For hash-based granularities, generate index-based keys
    if granularity.is_hash_based() {
        let num_shards = granularity.num_shards();
        return (0..num_shards).map(ShardKey::from_index).collect();
    }

    // For prefix-based granularities, generate all letter combinations
    let prefix_len = granularity.prefix_len_for_order(order);

    AllPrefixIter {
        prefix_len,
        current: None,
    }
    .map(|prefix| ShardKey::new(prefix))
    .collect()
}

/// Iterator that generates all possible lowercase letter prefixes of a given length.
struct AllPrefixIter {
    prefix_len: usize,
    current: Option<Vec<u8>>,
}

impl Iterator for AllPrefixIter {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.current {
            None => {
                // First iteration: start with "aaa..." (all 'a's)
                self.current = Some(vec![b'a'; self.prefix_len]);
                Some(String::from_utf8(vec![b'a'; self.prefix_len]).unwrap())
            }
            Some(chars) => {
                // Increment like a base-26 counter
                let mut i = self.prefix_len;
                while i > 0 {
                    i -= 1;
                    if chars[i] < b'z' {
                        chars[i] += 1;
                        // Reset all following positions to 'a'
                        for j in (i + 1)..self.prefix_len {
                            chars[j] = b'a';
                        }
                        return Some(String::from_utf8(chars.clone()).unwrap());
                    }
                }
                // All positions are 'z', we're done
                None
            }
        }
    }
}

/// Get the n-gram order from a pipe-separated n-gram string.
///
/// # Arguments
///
/// * `ngram` - The n-gram string (pipe-separated tokens)
///
/// # Returns
///
/// The number of tokens (order) in the n-gram.
pub fn ngram_order(ngram: &str) -> u8 {
    ngram.split('|').count() as u8
}

/// Compute the shard key from the first token of an n-gram.
///
/// This is used for vocabulary-indexed encoding where the n-gram key is a
/// sequence of PUA characters (not pipe-separated). Routing is based on
/// the original first token before encoding.
///
/// # Arguments
///
/// * `first_token` - The first word of the n-gram (e.g., "the")
/// * `order` - The n-gram order (1-5)
/// * `granularity` - The sharding granularity configuration
///
/// # Returns
///
/// A `ShardKey` identifying which shard should store this n-gram.
///
/// # Examples
///
/// ```ignore
/// use libgrammstein::sources::google_books::sharding::routing::compute_shard_key_from_token;
/// use libgrammstein::sources::google_books::sharding::config::ShardGranularity;
///
/// let key = compute_shard_key_from_token("the", 3, &ShardGranularity::TwoChar);
/// assert_eq!(key.prefix, "th");
/// ```
pub fn compute_shard_key_from_token(
    first_token: &str,
    order: u8,
    granularity: &ShardGranularity,
) -> ShardKey {
    // Handle hash-based routing: hash the first token
    if let ShardGranularity::CpuProportional { .. } = granularity {
        let num_shards = granularity.num_shards();
        let index = hash_to_shard(first_token, num_shards);
        return ShardKey::from_index(index);
    }

    // Prefix-based routing
    let prefix_len = granularity.prefix_len_for_order(order);

    // Get prefix characters, lowercase
    let prefix: String = first_token
        .chars()
        .filter(|c| c.is_alphabetic())
        .take(prefix_len)
        .flat_map(|c| c.to_lowercase())
        .collect();

    // Handle edge cases: non-alphabetic or short words
    let prefix = if prefix.is_empty() {
        // Non-alphabetic first word (numbers, symbols) → use special shard
        "_".repeat(prefix_len)
    } else if prefix.len() < prefix_len {
        // Short word → pad with 'a' (keeps lexicographic ordering)
        format!("{:a<width$}", prefix, width = prefix_len)
    } else {
        prefix
    };

    ShardKey::new(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_shard_key_first_char() {
        let g = ShardGranularity::FirstChar;

        let key = compute_shard_key("the|quick|brown", 3, &g);
        assert_eq!(key.prefix, "t");

        let key = compute_shard_key("APPLE", 1, &g);
        assert_eq!(key.prefix, "a"); // Lowercase

        let key = compute_shard_key("123", 1, &g);
        assert_eq!(key.prefix, "_"); // Non-alphabetic
    }

    #[test]
    fn test_compute_shard_key_two_char() {
        let g = ShardGranularity::TwoChar;

        let key = compute_shard_key("the|quick|brown", 3, &g);
        assert_eq!(key.prefix, "th");

        let key = compute_shard_key("a|b|c", 3, &g);
        assert_eq!(key.prefix, "aa"); // Padded

        let key = compute_shard_key("ZEBRA", 1, &g);
        assert_eq!(key.prefix, "ze"); // Lowercase
    }

    #[test]
    fn test_compute_shard_key_adaptive() {
        let g = ShardGranularity::Adaptive;

        // 1-grams: single char
        let key = compute_shard_key("apple", 1, &g);
        assert_eq!(key.prefix, "a");

        // 2-grams: two chars
        let key = compute_shard_key("apple|pie", 2, &g);
        assert_eq!(key.prefix, "ap");

        // 5-grams: two chars
        let key = compute_shard_key("the|quick|brown|fox|jumps", 5, &g);
        assert_eq!(key.prefix, "th");
    }

    #[test]
    fn test_compute_shard_key_cpu_proportional() {
        let g = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };

        // Hash-based routing should produce index-based keys
        let key = compute_shard_key("the|quick|brown", 3, &g);
        assert!(key.is_index_based());

        // Same n-gram should always route to same shard (deterministic)
        let key2 = compute_shard_key("the|quick|brown", 3, &g);
        assert_eq!(key.prefix, key2.prefix);

        // Different n-grams may route to different shards
        let key3 = compute_shard_key("apple|pie", 2, &g);
        assert!(key3.is_index_based());
    }

    #[test]
    fn test_hash_distribution() {
        let g = ShardGranularity::CpuProportional {
            multiplier: 1,
            minimum: 16, // Fixed 16 shards for testing
        };

        let num_shards = g.num_shards();
        let mut shard_counts = vec![0usize; num_shards];

        // Hash many n-grams and check distribution
        let test_ngrams = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "hello|world",
            "foo|bar",
            "test|data",
            "n-gram|model",
            "machine|learning",
        ];

        for ngram in &test_ngrams {
            let key = compute_shard_key(ngram, 1, &g);
            let index = key.as_index().expect("Should be index-based");
            assert!(
                index < num_shards,
                "Index {} out of range for {} shards",
                index,
                num_shards
            );
            shard_counts[index] += 1;
        }

        // At least some shards should have entries (not all in one)
        let non_empty = shard_counts.iter().filter(|&&c| c > 0).count();
        assert!(
            non_empty >= 2,
            "Hash distribution too skewed: only {} non-empty shards",
            non_empty
        );
    }

    #[test]
    fn test_shard_key_from_index() {
        let key = ShardKey::from_index(0);
        assert_eq!(key.prefix, "0000");
        assert!(key.is_index_based());
        assert_eq!(key.as_index(), Some(0));

        let key = ShardKey::from_index(42);
        assert_eq!(key.prefix, "0042");
        assert_eq!(key.as_index(), Some(42));

        let key = ShardKey::from_index(9999);
        assert_eq!(key.prefix, "9999");
        assert_eq!(key.as_index(), Some(9999));
    }

    #[test]
    fn test_shard_key_is_index_based() {
        // Index-based keys
        assert!(ShardKey::from_index(0).is_index_based());
        assert!(ShardKey::new("0000").is_index_based());
        assert!(ShardKey::new("1234").is_index_based());

        // Prefix-based keys
        assert!(!ShardKey::new("th").is_index_based());
        assert!(!ShardKey::new("apple").is_index_based());
        assert!(!ShardKey::new("_").is_index_based());
    }

    #[test]
    fn test_shard_key_for_file_prefix() {
        let g = ShardGranularity::TwoChar;

        let key = shard_key_for_file_prefix("th", 2, &g);
        assert_eq!(key.prefix, "th");

        // Pad short prefix
        let key = shard_key_for_file_prefix("a", 2, &g);
        assert_eq!(key.prefix, "aa");

        // Truncate long prefix
        let key = shard_key_for_file_prefix("the", 2, &g);
        assert_eq!(key.prefix, "th");
    }

    #[test]
    fn test_shard_key_for_file_prefix_cpu_proportional() {
        let g = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };

        // For hash-based, returns an index-based key
        let key = shard_key_for_file_prefix("th", 2, &g);
        assert!(key.is_index_based());
    }

    #[test]
    fn test_all_shard_keys_first_char() {
        let g = ShardGranularity::FirstChar;
        let keys = all_shard_keys(&g, 1);

        assert_eq!(keys.len(), 26);
        assert_eq!(keys[0].prefix, "a");
        assert_eq!(keys[25].prefix, "z");
    }

    #[test]
    fn test_all_shard_keys_two_char() {
        let g = ShardGranularity::TwoChar;
        let keys = all_shard_keys(&g, 2);

        assert_eq!(keys.len(), 676); // 26 * 26
        assert_eq!(keys[0].prefix, "aa");
        assert_eq!(keys[675].prefix, "zz");
    }

    #[test]
    fn test_all_shard_keys_cpu_proportional() {
        let g = ShardGranularity::CpuProportional {
            multiplier: 1,
            minimum: 16,
        };

        let keys = all_shard_keys(&g, 1);
        let num_shards = g.num_shards();

        assert_eq!(keys.len(), num_shards);

        // All keys should be index-based
        for (i, key) in keys.iter().enumerate() {
            assert!(key.is_index_based());
            assert_eq!(key.as_index(), Some(i));
        }
    }

    #[test]
    fn test_shard_key_display() {
        let key = ShardKey::new("th");
        assert_eq!(format!("{}", key), "th");

        let key = ShardKey::with_order("th", 3);
        assert_eq!(format!("{}", key), "th:3");

        let key = ShardKey::from_index(42);
        assert_eq!(format!("{}", key), "0042");
    }

    #[test]
    fn test_shard_key_file_stem() {
        let key = ShardKey::new("th");
        assert_eq!(key.as_file_stem(), "th");

        let key = ShardKey::with_order("th", 3);
        assert_eq!(key.as_file_stem(), "th_3");

        let key = ShardKey::from_index(42);
        assert_eq!(key.as_file_stem(), "0042");
    }

    #[test]
    fn test_ngram_order() {
        assert_eq!(ngram_order("apple"), 1);
        assert_eq!(ngram_order("apple|pie"), 2);
        assert_eq!(ngram_order("the|quick|brown|fox|jumps"), 5);
    }

    #[test]
    fn test_compute_shard_key_from_token_two_char() {
        let g = ShardGranularity::TwoChar;

        // Standard cases
        let key = compute_shard_key_from_token("the", 3, &g);
        assert_eq!(key.prefix, "th");

        let key = compute_shard_key_from_token("apple", 2, &g);
        assert_eq!(key.prefix, "ap");

        // Uppercase (should lowercase)
        let key = compute_shard_key_from_token("ZEBRA", 1, &g);
        assert_eq!(key.prefix, "ze");

        // Short word (should pad)
        let key = compute_shard_key_from_token("a", 2, &g);
        assert_eq!(key.prefix, "aa");

        // Non-alphabetic (should use underscore shard)
        let key = compute_shard_key_from_token("123", 1, &g);
        assert_eq!(key.prefix, "__");
    }

    #[test]
    fn test_compute_shard_key_from_token_adaptive() {
        let g = ShardGranularity::Adaptive;

        // 1-gram: single char
        let key = compute_shard_key_from_token("apple", 1, &g);
        assert_eq!(key.prefix, "a");

        // 2-gram: two chars
        let key = compute_shard_key_from_token("the", 2, &g);
        assert_eq!(key.prefix, "th");

        // 3-gram: two chars
        let key = compute_shard_key_from_token("quick", 3, &g);
        assert_eq!(key.prefix, "qu");
    }

    #[test]
    fn test_compute_shard_key_from_token_cpu_proportional() {
        let g = ShardGranularity::CpuProportional {
            multiplier: 2,
            minimum: 8,
        };

        // Hash-based routing should produce index-based keys
        let key = compute_shard_key_from_token("the", 3, &g);
        assert!(key.is_index_based());

        // Same token should always route to same shard (deterministic)
        let key2 = compute_shard_key_from_token("the", 3, &g);
        assert_eq!(key.prefix, key2.prefix);

        // Different tokens may route to different shards
        let key3 = compute_shard_key_from_token("apple", 2, &g);
        assert!(key3.is_index_based());
    }

    #[test]
    fn test_compute_shard_key_from_token_matches_compute_shard_key() {
        // Verify that compute_shard_key_from_token gives the same result as
        // compute_shard_key when given the first token of an n-gram
        let granularities = [
            ShardGranularity::FirstChar,
            ShardGranularity::TwoChar,
            ShardGranularity::Adaptive,
        ];

        for g in &granularities {
            // "the|quick|brown" -> first token is "the"
            let key_from_ngram = compute_shard_key("the|quick|brown", 3, g);
            let key_from_token = compute_shard_key_from_token("the", 3, g);
            assert_eq!(key_from_ngram, key_from_token, "Mismatch for {:?}", g);

            // "apple|pie" -> first token is "apple"
            let key_from_ngram = compute_shard_key("apple|pie", 2, g);
            let key_from_token = compute_shard_key_from_token("apple", 2, g);
            assert_eq!(key_from_ngram, key_from_token, "Mismatch for {:?}", g);
        }
    }
}
