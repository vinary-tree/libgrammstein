//! Metadata-filtering zipper wrapper.
//!
//! Wraps any `DictZipper` and filters out `\x00` prefixed metadata paths.
//! This provides a generic reusable wrapper for any char-based zipper that
//! needs to exclude metadata entries from iteration and navigation.
//!
//! # Design
//!
//! The zipper:
//! - Filters `\x00` edges only at the root level (metadata convention)
//! - `children()` skips edges with label `\x00` when at root
//! - `descend('\x00')` returns `None` when at root
//! - Non-root nodes pass through to the wrapped zipper unchanged
//!
//! # Why Filter Only at Root?
//!
//! Metadata entries in this codebase use `\x00` prefixed keys that exist only
//! at the trie root level (e.g., `\x00__meta__:version`). Non-root nodes cannot
//! have metadata entries, so filtering at every level would be unnecessary overhead.

use liblevenshtein::dictionary::value::DictionaryValue;
use liblevenshtein::dictionary::zipper::{DictZipper, ValuedDictZipper};

/// Metadata key prefix (null byte).
///
/// Keys starting with this character are reserved for internal metadata
/// (vocabulary version, checkpoints, etc.) and should not be exposed
/// through normal iteration.
pub const METADATA_PREFIX: char = '\x00';

/// Zipper wrapper that filters out metadata at root level.
///
/// This is a generic wrapper that can be composed with any `DictZipper<Unit = char>`
/// to automatically exclude `\x00` prefixed metadata from navigation and iteration.
///
/// # Type Parameters
///
/// * `Z` - The underlying zipper type (must implement `DictZipper<Unit = char>`)
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::metadata_filtering_zipper::MetadataFilteringZipper;
/// use libdictenstein::dynamic_dawg_char_zipper::DynamicDawgCharZipper;
///
/// let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
/// dict.insert_with_value("\x00meta", 1);
/// dict.insert_with_value("normal", 2);
///
/// let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
/// let zipper = MetadataFilteringZipper::new(backend_zipper);
///
/// // Metadata is filtered:
/// assert!(zipper.descend('\x00').is_none());
/// // Normal data is accessible:
/// assert!(zipper.descend('n').is_some());
/// ```
#[derive(Clone, Debug)]
pub struct MetadataFilteringZipper<Z> {
    /// The wrapped zipper.
    inner: Z,
    /// Whether we're at the root level.
    /// Metadata filtering only applies at root.
    at_root: bool,
}

impl<Z: DictZipper<Unit = char>> MetadataFilteringZipper<Z> {
    /// Create a new metadata-filtering zipper wrapping the given inner zipper.
    ///
    /// The inner zipper should be positioned at the root.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying zipper to wrap
    pub fn new(inner: Z) -> Self {
        Self {
            inner,
            at_root: true,
        }
    }

    /// Get a reference to the inner zipper.
    pub fn inner(&self) -> &Z {
        &self.inner
    }

    /// Check if we're currently at the root level.
    pub fn is_at_root(&self) -> bool {
        self.at_root
    }
}

impl<Z: DictZipper<Unit = char>> DictZipper for MetadataFilteringZipper<Z> {
    type Unit = char;

    fn is_final(&self) -> bool {
        self.inner.is_final()
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        // Block descend to metadata at root level
        if self.at_root && label == METADATA_PREFIX {
            return None;
        }

        self.inner.descend(label).map(|inner| Self {
            inner,
            at_root: false,
        })
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        let at_root = self.at_root;
        self.inner
            .children()
            .filter(move |(label, _)| !at_root || *label != METADATA_PREFIX)
            .map(|(label, inner)| {
                (
                    label,
                    Self {
                        inner,
                        at_root: false,
                    },
                )
            })
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.inner.path()
    }
}

impl<Z: ValuedDictZipper<Unit = char>> ValuedDictZipper for MetadataFilteringZipper<Z>
where
    Z::Value: DictionaryValue,
{
    type Value = Z::Value;

    fn value(&self) -> Option<Self::Value> {
        self.inner.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;
    use liblevenshtein::dictionary::dynamic_dawg_char_zipper::DynamicDawgCharZipper;

    #[test]
    fn test_metadata_filtering_zipper_blocks_descend() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        dict.insert_with_value("\x00meta", 1);
        dict.insert_with_value("normal", 2);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        // Should block descend to metadata at root
        assert!(zipper.descend('\x00').is_none());

        // Should allow descend to normal data
        assert!(zipper.descend('n').is_some());
    }

    #[test]
    fn test_metadata_filtering_zipper_filters_children() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        dict.insert_with_value("\x00meta", 1);
        dict.insert_with_value("abc", 2);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        let children: Vec<char> = zipper.children().map(|(c, _)| c).collect();
        assert!(!children.contains(&'\x00'));
        assert!(children.contains(&'a'));
    }

    #[test]
    fn test_metadata_filtering_allows_null_after_root() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        // Insert a term that has \x00 NOT at the start
        dict.insert_with_value("a\x00b", 42);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        // Navigate to 'a'
        let a = zipper.descend('a').expect("should descend to 'a'");
        assert!(!a.is_at_root());

        // Should be able to descend to '\x00' since we're not at root
        let null = a
            .descend('\x00')
            .expect("should descend to '\\x00' after root");
        assert!(null.descend('b').is_some());
    }

    #[test]
    fn test_metadata_filtered_iteration() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        dict.insert_with_value("\x00__meta__", 999);
        dict.insert_with_value("word1", 1);
        dict.insert_with_value("word2", 2);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        // Count children at root - should not include metadata
        let root_children: Vec<char> = zipper.children().map(|(c, _)| c).collect();
        assert!(!root_children.contains(&'\x00'));
        assert!(root_children.contains(&'w')); // Both word1 and word2 start with 'w'

        // The metadata key should not be reachable via descend
        assert!(zipper.descend('\x00').is_none());
    }

    #[test]
    fn test_valued_zipper() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        dict.insert_with_value("test", 42);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        let t = zipper.descend('t').unwrap();
        let e = t.descend('e').unwrap();
        let s = e.descend('s').unwrap();
        let final_t = s.descend('t').unwrap();

        assert!(final_t.is_final());
        assert_eq!(final_t.value(), Some(42));
    }

    #[test]
    fn test_path_tracking() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();
        dict.insert_with_value("abc", 1);

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        let a = zipper.descend('a').unwrap();
        assert_eq!(a.path(), vec!['a']);

        let b = a.descend('b').unwrap();
        assert_eq!(b.path(), vec!['a', 'b']);

        let c = b.descend('c').unwrap();
        assert_eq!(c.path(), vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_empty_dict() {
        let dict: DynamicDawgChar<u64> = DynamicDawgChar::new();

        let backend_zipper = DynamicDawgCharZipper::new_from_dict(&dict);
        let zipper = MetadataFilteringZipper::new(backend_zipper);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }
}
