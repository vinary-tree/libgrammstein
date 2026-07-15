//! N-gram entry type for dictionary storage.
//!
//! This module defines the `NgramEntry` struct that stores n-gram statistics
//! in libdictenstein dictionary backends.

use libdictenstein::value::DictionaryValue;
use serde::{Deserialize, Serialize};

/// Entry stored for each n-gram in the dictionary.
///
/// Contains the raw count and statistics needed for Modified Kneser-Ney smoothing.
///
/// # Modified Kneser-Ney Statistics
///
/// For MKN smoothing, we need:
/// - `count`: Raw corpus count of this n-gram
/// - `continuation_count`: Number of unique contexts that precede this n-gram
///   (used for lower-order probability estimation)
/// - `unique_continuations`: Number of unique words that follow this n-gram
///   (used for computing interpolation weights)
///
/// # Concurrency
///
/// `NgramEntry` is a plain, `Copy` value with no interior atomics. Concurrent
/// updates are the *store's* responsibility: the byte-keyed backends apply an
/// update closure to a private clone and publish it with a lock-free
/// compare-and-swap (`update_or_insert_bytes`), so the entry itself carries no
/// synchronization. This mirrors the value-CAS model exactly — an interior
/// atomic would be snapshotted by the per-attempt clone and buy nothing.
///
/// # Example
///
/// ```
/// use libgrammstein::ngram::NgramEntry;
///
/// let mut entry = NgramEntry::new(5);
/// assert_eq!(entry.count(), 5);
///
/// entry.increment();
/// assert_eq!(entry.count(), 6);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NgramEntry {
    /// Raw corpus count of this n-gram.
    count: u64,

    /// Number of unique preceding contexts (for continuation probability).
    /// Used in Modified Kneser-Ney for lower-order probability estimation.
    continuation_count: u32,

    /// Number of unique following words.
    /// Used to compute interpolation weights in MKN.
    unique_continuations: u32,
}

impl NgramEntry {
    /// Create a new n-gram entry with the given initial count.
    #[inline]
    pub fn new(count: u64) -> Self {
        Self {
            count,
            continuation_count: 0,
            unique_continuations: 0,
        }
    }

    /// Create a new entry with all statistics initialized.
    #[inline]
    pub fn with_stats(count: u64, continuation_count: u32, unique_continuations: u32) -> Self {
        Self {
            count,
            continuation_count,
            unique_continuations,
        }
    }

    /// Get the raw corpus count.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the continuation count (unique preceding contexts).
    #[inline]
    pub fn continuation_count(&self) -> u32 {
        self.continuation_count
    }

    /// Get the number of unique continuations (following words).
    #[inline]
    pub fn unique_continuations(&self) -> u32 {
        self.unique_continuations
    }

    /// Increment the count by 1.
    #[inline]
    pub fn increment(&mut self) {
        self.count += 1;
    }

    /// Increment the count by a given amount.
    #[inline]
    pub fn increment_by(&mut self, amount: u64) {
        self.count += amount;
    }

    /// Increment the continuation count by 1.
    #[inline]
    pub fn increment_continuation(&mut self) {
        self.continuation_count += 1;
    }

    /// Increment the unique continuations count by 1.
    #[inline]
    pub fn increment_unique_continuations(&mut self) {
        self.unique_continuations += 1;
    }

    /// Set the continuation count (typically done after the initial counting pass).
    #[inline]
    pub fn set_continuation_count(&mut self, value: u32) {
        self.continuation_count = value;
    }

    /// Set the unique continuations count.
    #[inline]
    pub fn set_unique_continuations(&mut self, value: u32) {
        self.unique_continuations = value;
    }
}

// Implement DictionaryValue for storage in libdictenstein dictionaries.
// DictionaryValue requires Clone + Send + Sync + Unpin + Serialize + DeserializeOwned + 'static.
impl DictionaryValue for NgramEntry {}

/// Snapshot of `NgramEntry` for the portable serialization format.
///
/// Structurally identical to `NgramEntry`; kept as a distinct name so the
/// on-disk portable model schema (`PortableNgramModel::entries`) stays stable
/// and self-documenting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NgramEntrySnapshot {
    /// Raw corpus count.
    pub count: u64,
    /// Continuation count.
    pub continuation_count: u32,
    /// Unique continuations.
    pub unique_continuations: u32,
}

impl From<&NgramEntry> for NgramEntrySnapshot {
    fn from(entry: &NgramEntry) -> Self {
        Self {
            count: entry.count(),
            continuation_count: entry.continuation_count(),
            unique_continuations: entry.unique_continuations(),
        }
    }
}

impl From<NgramEntrySnapshot> for NgramEntry {
    fn from(snapshot: NgramEntrySnapshot) -> Self {
        Self::with_stats(
            snapshot.count,
            snapshot.continuation_count,
            snapshot.unique_continuations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_entry() {
        let entry = NgramEntry::new(42);
        assert_eq!(entry.count(), 42);
        assert_eq!(entry.continuation_count(), 0);
        assert_eq!(entry.unique_continuations(), 0);
    }

    #[test]
    fn test_with_stats() {
        let entry = NgramEntry::with_stats(100, 10, 5);
        assert_eq!(entry.count(), 100);
        assert_eq!(entry.continuation_count(), 10);
        assert_eq!(entry.unique_continuations(), 5);
    }

    #[test]
    fn test_increment() {
        let mut entry = NgramEntry::new(0);
        entry.increment();
        entry.increment();
        entry.increment();
        assert_eq!(entry.count(), 3);
    }

    #[test]
    fn test_increment_by() {
        let mut entry = NgramEntry::new(10);
        entry.increment_by(5);
        assert_eq!(entry.count(), 15);
    }

    #[test]
    fn test_continuation_stats_mutation() {
        let mut entry = NgramEntry::new(7);
        entry.set_continuation_count(4);
        entry.increment_unique_continuations();
        entry.increment_unique_continuations();
        assert_eq!(entry.count(), 7);
        assert_eq!(entry.continuation_count(), 4);
        assert_eq!(entry.unique_continuations(), 2);
    }

    #[test]
    fn test_clone_is_copy() {
        let entry = NgramEntry::with_stats(50, 8, 3);
        let copied = entry; // Copy
        assert_eq!(copied.count(), 50);
        assert_eq!(copied.continuation_count(), 8);
        assert_eq!(copied.unique_continuations(), 3);
        // original still usable (Copy, not moved)
        assert_eq!(entry.count(), 50);
    }

    #[test]
    fn test_snapshot_conversion() {
        let entry = NgramEntry::with_stats(100, 20, 10);
        let snapshot = NgramEntrySnapshot::from(&entry);

        assert_eq!(snapshot.count, 100);
        assert_eq!(snapshot.continuation_count, 20);
        assert_eq!(snapshot.unique_continuations, 10);

        let restored = NgramEntry::from(snapshot);
        assert_eq!(restored.count(), 100);
        assert_eq!(restored.continuation_count(), 20);
        assert_eq!(restored.unique_continuations(), 10);
    }
}
