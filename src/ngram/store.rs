//! Byte-native n-gram store — the term-id (LEB128) storage seam for the MKN model.
//!
//! [`NgramLookup`] is the read surface the Modified Kneser-Ney estimator needs;
//! [`MutableNgramStore`] adds the training surface. [`TermIdStore`] implements
//! both over a byte-unit libdictenstein backend keyed on **raw LEB128 varint
//! term-id sequences** — the same self-delimiting encoding the Google-Books
//! count store uses — so [`NgramEntry`] values are looked up, incremented, and
//! iterated by term-id, never by a `'|'`-delimited string, and the
//! delimiter-collision class cannot occur.
//!
//! [`TermIdStore`] is the crate's single live n-gram store. There is no live
//! `'|'`-delimited store: an old `'|'`-keyed portable model is read exactly once,
//! at load time, by the labelled legacy reader in `model.rs`, which re-keys it
//! onto term-ids before it ever becomes a live [`NgramModel`].
//!
//! Concurrency for `TermIdStore` is the backend's: mutation goes through
//! libdictenstein's lock-free `&self` byte read-modify-write
//! (`update_or_insert_bytes`, contract C1), which clones the value per CAS
//! attempt and publishes it with a root-pointer swap — which is exactly why
//! [`NgramEntry`] carries no interior atomics.

use std::sync::Arc;

use libdictenstein::dynamic_dawg::DynamicDawg;
use libdictenstein::persistent_artrie::PersistentARTrie;
use libdictenstein::Dictionary;

use super::entry::NgramEntry;
use super::vocabulary::{
    decode_ngram_key_bytes, encode_indices_to_key_bytes, encode_ngram_key_bytes,
    encode_ngram_key_existing_bytes, SharedVocabARTrie,
};

/// Read surface for scoring: n-gram entries addressed by word tokens.
///
/// The Kneser-Ney estimator needs exactly this — `count` for the discount
/// numerator/denominator and `entry` for the continuation statistics — so it is
/// generic over any `S: NgramLookup`. The only implementor is the byte-native
/// [`TermIdStore`].
pub trait NgramLookup {
    /// The model order (maximum n-gram length the store holds).
    fn max_order(&self) -> usize;

    /// The stored entry for `tokens`, or `None` if the n-gram is absent.
    fn entry(&self, tokens: &[&str]) -> Option<NgramEntry>;

    /// The raw count for `tokens` (0 if absent).
    #[inline]
    fn count(&self, tokens: &[&str]) -> u64 {
        self.entry(tokens).map(|e| e.count()).unwrap_or(0)
    }

    /// Number of stored n-grams.
    fn len(&self) -> usize;

    /// Whether the store holds no n-grams.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Training surface: insert-or-increment, continuation writes, and term-id iteration.
pub trait MutableNgramStore: NgramLookup {
    /// Increment the count of `tokens` by `by` (inserting the n-gram at `by` if new).
    /// Returns `true` iff the n-gram was newly inserted.
    fn bump(&self, tokens: &[&str], by: u64) -> bool;

    /// Set the continuation count (unique preceding contexts) of `tokens`.
    fn set_continuation_count(&self, tokens: &[&str], value: u32);

    /// Set the unique-continuations count (unique following words) of `tokens`.
    fn set_unique_continuations(&self, tokens: &[&str], value: u32);

    /// Iterate every stored n-gram as its decoded term-id sequence and entry.
    /// No `'|'` string is ever formed — the delimiter-collision class cannot occur.
    fn iter_term_ids(&self) -> Box<dyn Iterator<Item = (Vec<u64>, NgramEntry)> + '_>;
}

/// libgrammstein-side adapter over a byte-unit libdictenstein backend storing
/// [`NgramEntry`], addressed by raw LEB128 term-id byte keys. Method names are
/// distinct from the backend's inherent `*_bytes` methods so the impls can call
/// those without shadowing — and libdictenstein stays untouched.
pub trait ByteMappedDictionary {
    /// Read the entry stored under raw byte key `key`.
    fn entry_bytes(&self, key: &[u8]) -> Option<NgramEntry>;

    /// Iterate `(raw byte key, entry)` pairs.
    fn iter_entries_bytes(&self) -> Box<dyn Iterator<Item = (Vec<u8>, NgramEntry)> + '_>;

    /// Number of stored keys.
    fn entry_count(&self) -> usize;
}

/// The mutable half: a lock-free `&self` byte read-modify-write plus a bulk overwrite.
pub trait MutableByteMappedDictionary: ByteMappedDictionary {
    /// Lock-free `&self` read-modify-write: apply `f` to the current value, or
    /// insert `default`. Returns `true` iff a new entry was inserted.
    /// (Backed by libdictenstein's `update_or_insert_bytes`, contract C1.)
    fn modify_or_insert(
        &self,
        key: &[u8],
        default: NgramEntry,
        f: &dyn Fn(&mut NgramEntry),
    ) -> bool;

    /// Overwrite the value at `key` (bulk load / portable-model rebuild / transcode).
    fn overwrite_bytes(&self, key: &[u8], value: NgramEntry) -> bool;
}

// --- adapter impls (libdictenstein untouched; distinct trait method names) ---

impl ByteMappedDictionary for DynamicDawg<NgramEntry> {
    #[inline]
    fn entry_bytes(&self, key: &[u8]) -> Option<NgramEntry> {
        self.get_bytes_value(key)
    }

    fn iter_entries_bytes(&self) -> Box<dyn Iterator<Item = (Vec<u8>, NgramEntry)> + '_> {
        Box::new(self.iter_bytes_with_values())
    }

    #[inline]
    fn entry_count(&self) -> usize {
        Dictionary::len(self).unwrap_or(0)
    }
}

impl MutableByteMappedDictionary for DynamicDawg<NgramEntry> {
    #[inline]
    fn modify_or_insert(
        &self,
        key: &[u8],
        default: NgramEntry,
        f: &dyn Fn(&mut NgramEntry),
    ) -> bool {
        self.update_or_insert_bytes(key, default, f)
    }

    #[inline]
    fn overwrite_bytes(&self, key: &[u8], value: NgramEntry) -> bool {
        self.insert_bytes_with_value(key, value)
    }
}

impl ByteMappedDictionary for Arc<PersistentARTrie<NgramEntry>> {
    #[inline]
    fn entry_bytes(&self, key: &[u8]) -> Option<NgramEntry> {
        (**self).get_value_bytes(key)
    }

    fn iter_entries_bytes(&self) -> Box<dyn Iterator<Item = (Vec<u8>, NgramEntry)> + '_> {
        Box::new((**self).iter_bytes_with_values())
    }

    #[inline]
    fn entry_count(&self) -> usize {
        Dictionary::len(&**self).unwrap_or(0)
    }
}

impl MutableByteMappedDictionary for Arc<PersistentARTrie<NgramEntry>> {
    #[inline]
    fn modify_or_insert(
        &self,
        key: &[u8],
        default: NgramEntry,
        f: &dyn Fn(&mut NgramEntry),
    ) -> bool {
        (**self)
            .update_or_insert_bytes(key, default, f)
            .unwrap_or(false)
    }

    #[inline]
    fn overwrite_bytes(&self, key: &[u8], value: NgramEntry) -> bool {
        (**self).upsert_bytes(key, value).unwrap_or(false)
    }
}

/// The byte-native MKN store: LEB128 term-id keys → [`NgramEntry`], over a byte
/// backend `B`, with the vocabulary that encodes/decodes tokens ↔ term-ids.
///
/// `B` is `DynamicDawg<NgramEntry>` (in-memory) or `Arc<PersistentARTrie<NgramEntry>>`
/// (disk). Its public API is word-oriented; encode/decode happen only here.
///
/// `Clone` is available when the backend is cloneable; both concrete backends
/// clone cheaply (the in-memory DAWG and the disk trie are internally
/// `Arc`-shared), so a cloned store aliases the same n-gram data and vocabulary.
#[derive(Clone)]
pub struct TermIdStore<B> {
    backend: B,
    vocab: SharedVocabARTrie,
    max_order: usize,
}

impl<B> TermIdStore<B> {
    /// Wrap `backend` (a fresh or existing byte trie) with `vocab` for a model of `max_order`.
    pub fn new(backend: B, vocab: SharedVocabARTrie, max_order: usize) -> Self {
        Self {
            backend,
            vocab,
            max_order,
        }
    }

    /// The vocabulary (word ↔ term-id bijection).
    #[inline]
    pub fn vocabulary(&self) -> &SharedVocabARTrie {
        &self.vocab
    }

    /// The underlying byte backend.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B: ByteMappedDictionary> NgramLookup for TermIdStore<B> {
    #[inline]
    fn max_order(&self) -> usize {
        self.max_order
    }

    fn entry(&self, tokens: &[&str]) -> Option<NgramEntry> {
        // Read-only encode: never grows the vocabulary; an OOV token ⇒ no such n-gram.
        let key = encode_ngram_key_existing_bytes(tokens, &self.vocab)?;
        self.backend.entry_bytes(&key)
    }

    #[inline]
    fn len(&self) -> usize {
        self.backend.entry_count()
    }
}

impl<B: MutableByteMappedDictionary> MutableNgramStore for TermIdStore<B> {
    fn bump(&self, tokens: &[&str], by: u64) -> bool {
        let key = encode_ngram_key_bytes(tokens, &self.vocab);
        self.backend
            .modify_or_insert(&key, NgramEntry::new(by), &move |e| e.increment_by(by))
    }

    fn set_continuation_count(&self, tokens: &[&str], value: u32) {
        let key = encode_ngram_key_bytes(tokens, &self.vocab);
        self.backend
            .modify_or_insert(&key, NgramEntry::with_stats(0, value, 0), &move |e| {
                e.set_continuation_count(value)
            });
    }

    fn set_unique_continuations(&self, tokens: &[&str], value: u32) {
        let key = encode_ngram_key_bytes(tokens, &self.vocab);
        self.backend
            .modify_or_insert(&key, NgramEntry::with_stats(0, 0, value), &move |e| {
                e.set_unique_continuations(value)
            });
    }

    fn iter_term_ids(&self) -> Box<dyn Iterator<Item = (Vec<u64>, NgramEntry)> + '_> {
        Box::new(
            self.backend
                .iter_entries_bytes()
                .map(|(key, entry)| (decode_ngram_key_bytes(&key), entry)),
        )
    }
}

impl<B: MutableByteMappedDictionary> TermIdStore<B> {
    /// Load a raw `(term-id byte key, entry)` pair directly — used by the portable
    /// model rebuild and the legacy→term-id transcoder, where keys are already encoded.
    #[inline]
    pub fn insert_raw(&self, key: &[u8], entry: NgramEntry) -> bool {
        self.backend.overwrite_bytes(key, entry)
    }

    /// Set the continuation count of the n-gram addressed by the term-id sequence
    /// `ids` directly, without re-encoding tokens through the vocabulary.
    ///
    /// The trainer's byte-native continuation pass already holds decoded term-ids
    /// (from [`iter_term_ids`](MutableNgramStore::iter_term_ids)); this avoids the
    /// id → word → id round-trip the token-oriented
    /// [`set_continuation_count`](MutableNgramStore::set_continuation_count) would
    /// incur.
    pub fn set_continuation_count_ids(&self, ids: &[u64], value: u32) {
        let key = encode_indices_to_key_bytes(ids);
        self.backend
            .modify_or_insert(&key, NgramEntry::with_stats(0, value, 0), &move |e| {
                e.set_continuation_count(value)
            });
    }

    /// Set the unique-continuations count of the n-gram addressed by the term-id
    /// sequence `ids` directly (see [`set_continuation_count_ids`](Self::set_continuation_count_ids)).
    pub fn set_unique_continuations_ids(&self, ids: &[u64], value: u32) {
        let key = encode_indices_to_key_bytes(ids);
        self.backend
            .modify_or_insert(&key, NgramEntry::with_stats(0, 0, value), &move |e| {
                e.set_unique_continuations(value)
            });
    }
}

/// Storage-agnostic word-sequence iteration.
///
/// Consumers that need the decoded *words* of every stored n-gram — the WFST
/// export, the generation sampler, REPL completion — iterate through this.
/// Only [`TermIdStore`] implements it: it decodes each n-gram's term-id varints
/// through the vocabulary ([`get_term`](libdictenstein::persistent_artrie::vocab::PersistentVocabARTrie)).
/// No delimiter is ever split — the delimiter-collision class cannot occur.
pub trait IterableNgramStore: NgramLookup {
    /// Iterate `(words, entry)` for every stored n-gram.
    fn iter_ngrams(&self) -> Box<dyn Iterator<Item = (Vec<String>, NgramEntry)> + '_>;
}

impl<B: ByteMappedDictionary> IterableNgramStore for TermIdStore<B> {
    fn iter_ngrams(&self) -> Box<dyn Iterator<Item = (Vec<String>, NgramEntry)> + '_> {
        let vocab = self.vocab.clone();
        Box::new(
            self.backend
                .iter_entries_bytes()
                .filter_map(move |(key, entry)| {
                    let ids = decode_ngram_key_bytes(&key);
                    let guard = vocab.as_ref();
                    let mut words = Vec::with_capacity(ids.len());
                    for id in ids {
                        // A term-id absent from the vocabulary (never assigned)
                        // is skipped defensively.
                        words.push(guard.get_term(id)?);
                    }
                    Some((words, entry))
                }),
        )
    }
}
