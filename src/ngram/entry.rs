//! N-gram entry type for dictionary storage.
//!
//! This module defines the `NgramEntry` struct that stores n-gram statistics
//! in liblevenshtein-rust dictionary backends.

use liblevenshtein::dictionary::value::DictionaryValue;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

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
/// # Thread Safety
///
/// `NgramEntry` uses atomic operations for thread-safe counting during parallel
/// corpus processing. The `Sync` and `Send` traits are automatically derived.
///
/// # Example
///
/// ```
/// use libgrammstein::ngram::NgramEntry;
///
/// let entry = NgramEntry::new(5);
/// assert_eq!(entry.count(), 5);
///
/// entry.increment();
/// assert_eq!(entry.count(), 6);
/// ```
#[derive(Debug, Default)]
pub struct NgramEntry {
    /// Raw corpus count of this n-gram.
    count: AtomicU64,

    /// Number of unique preceding contexts (for continuation probability).
    /// Used in Modified Kneser-Ney for lower-order probability estimation.
    continuation_count: AtomicU32,

    /// Number of unique following words.
    /// Used to compute interpolation weights in MKN.
    unique_continuations: AtomicU32,
}

impl NgramEntry {
    /// Create a new n-gram entry with the given initial count.
    #[inline]
    pub fn new(count: u64) -> Self {
        Self {
            count: AtomicU64::new(count),
            continuation_count: AtomicU32::new(0),
            unique_continuations: AtomicU32::new(0),
        }
    }

    /// Create a new entry with all statistics initialized.
    #[inline]
    pub fn with_stats(count: u64, continuation_count: u32, unique_continuations: u32) -> Self {
        Self {
            count: AtomicU64::new(count),
            continuation_count: AtomicU32::new(continuation_count),
            unique_continuations: AtomicU32::new(unique_continuations),
        }
    }

    /// Get the raw corpus count.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the continuation count (unique preceding contexts).
    #[inline]
    pub fn continuation_count(&self) -> u32 {
        self.continuation_count.load(Ordering::Relaxed)
    }

    /// Get the number of unique continuations (following words).
    #[inline]
    pub fn unique_continuations(&self) -> u32 {
        self.unique_continuations.load(Ordering::Relaxed)
    }

    /// Atomically increment the count by 1.
    #[inline]
    pub fn increment(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically increment the count by a given amount.
    #[inline]
    pub fn increment_by(&self, amount: u64) {
        self.count.fetch_add(amount, Ordering::Relaxed);
    }

    /// Atomically increment the continuation count by 1.
    #[inline]
    pub fn increment_continuation(&self) {
        self.continuation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically increment the unique continuations count by 1.
    #[inline]
    pub fn increment_unique_continuations(&self) {
        self.unique_continuations.fetch_add(1, Ordering::Relaxed);
    }

    /// Set the continuation count (typically done after initial counting pass).
    #[inline]
    pub fn set_continuation_count(&self, value: u32) {
        self.continuation_count.store(value, Ordering::Relaxed);
    }

    /// Set the unique continuations count.
    #[inline]
    pub fn set_unique_continuations(&self, value: u32) {
        self.unique_continuations.store(value, Ordering::Relaxed);
    }
}

impl Clone for NgramEntry {
    fn clone(&self) -> Self {
        Self {
            count: AtomicU64::new(self.count.load(Ordering::Relaxed)),
            continuation_count: AtomicU32::new(self.continuation_count.load(Ordering::Relaxed)),
            unique_continuations: AtomicU32::new(self.unique_continuations.load(Ordering::Relaxed)),
        }
    }
}

// Implement DictionaryValue for storage in liblevenshtein dictionaries.
// Note: DictionaryValue requires Clone + Send + Sync + Unpin + 'static.
// NgramEntry satisfies all these via atomic types.
impl DictionaryValue for NgramEntry {}

/// Snapshot of NgramEntry for non-atomic access.
///
/// Used when we need to pass n-gram data across thread boundaries
/// or when atomic operations are not needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

#[cfg(feature = "serde")]
impl serde::Serialize for NgramEntry {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NgramEntry", 3)?;
        state.serialize_field("count", &self.count())?;
        state.serialize_field("continuation_count", &self.continuation_count())?;
        state.serialize_field("unique_continuations", &self.unique_continuations())?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for NgramEntry {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            count: u64,
            continuation_count: u32,
            unique_continuations: u32,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(Self::with_stats(
            helper.count,
            helper.continuation_count,
            helper.unique_continuations,
        ))
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
        let entry = NgramEntry::new(0);
        entry.increment();
        entry.increment();
        entry.increment();
        assert_eq!(entry.count(), 3);
    }

    #[test]
    fn test_increment_by() {
        let entry = NgramEntry::new(10);
        entry.increment_by(5);
        assert_eq!(entry.count(), 15);
    }

    #[test]
    fn test_clone() {
        let entry = NgramEntry::with_stats(50, 8, 3);
        let cloned = entry.clone();
        assert_eq!(cloned.count(), 50);
        assert_eq!(cloned.continuation_count(), 8);
        assert_eq!(cloned.unique_continuations(), 3);
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

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let entry = Arc::new(NgramEntry::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let entry = Arc::clone(&entry);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    entry.increment();
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        assert_eq!(entry.count(), 10000);
    }
}
