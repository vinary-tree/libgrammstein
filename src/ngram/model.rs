//! N-gram language model with probability queries.
//!
//! This module provides the main [`NgramModel`] struct that combines an n-gram
//! store (the [`NgramLookup`] seam) with a smoothing algorithm for probability
//! estimation. The store is always the byte-native
//! [`TermIdStore`](super::store::TermIdStore) (LEB128 term-id keys). Old
//! `'|'`-joined portable files are re-keyed onto term-ids by a single labelled
//! one-shot reader ([`read_legacy_pipe_entries`]) at load time; no
//! `'|'`-delimited store is ever live.

use super::entry::NgramEntry;
use super::smoothing::KneserNeySmoothing;
use super::store::{IterableNgramStore, NgramLookup};

#[cfg(feature = "serde-extras")]
use super::entry::NgramEntrySnapshot;
#[cfg(feature = "serde-extras")]
use super::store::{MutableByteMappedDictionary, TermIdStore};
#[cfg(feature = "serde-extras")]
use super::tempdir::{create_guarded_temp_vocab_dir, VocabTempDir};
#[cfg(feature = "serde-extras")]
use super::vocabulary::{
    create_vocabulary, decode_ngram_key_bytes, encode_indices_to_key_bytes, encode_ngram_key_bytes,
    SharedVocabARTrie, FIRST_VALID_INDEX,
};

#[cfg(feature = "serde-extras")]
use std::path::Path;

/// N-gram language model with Modified Kneser-Ney smoothing.
///
/// This is the main interface for training and querying n-gram language models.
///
/// # Type Parameters
///
/// * `S` - The n-gram store, any [`NgramLookup`]. The sole implementor is the
///   byte-native [`TermIdStore`](super::store::TermIdStore) (LEB128 term-id
///   keys), instantiated over an in-memory `DynamicDawg<NgramEntry>` or a
///   disk-backed `Arc<PersistentARTrie<NgramEntry>>`.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::ngram::NgramModel;
/// use libgrammstein::corpus::PlaintextReader;
///
/// // Train a trigram model
/// let reader = PlaintextReader::from_file("corpus.txt")?;
/// let model = NgramModel::train(reader, 3)?;
///
/// // Query probabilities
/// let log_prob = model.log_prob("fox", &["quick", "brown"]);
/// println!("log P(fox | quick brown) = {}", log_prob);
///
/// // Score a sentence
/// let sentence_log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
/// ```
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "S: serde::Serialize + serde::de::DeserializeOwned")]
pub struct NgramModel<S>
where
    S: NgramLookup,
{
    /// N-gram store (the [`NgramLookup`] seam).
    store: S,

    /// Smoothing algorithm.
    smoothing: KneserNeySmoothing,

    /// Vocabulary size (number of unique unigrams).
    vocab_size: usize,

    /// Total unigram count (corpus size in tokens).
    total_count: u64,
}

impl<S> NgramModel<S>
where
    S: NgramLookup,
{
    /// Create a new n-gram model from a trained store.
    ///
    /// This is typically called after the training process completes.
    pub fn new(
        store: S,
        smoothing: KneserNeySmoothing,
        vocab_size: usize,
        total_count: u64,
    ) -> Self {
        Self {
            store,
            smoothing,
            vocab_size,
            total_count,
        }
    }

    /// Get the n-gram order (maximum context length + 1).
    #[inline]
    pub fn order(&self) -> usize {
        self.store.max_order()
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the total unigram count.
    #[inline]
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Get a reference to the underlying store.
    #[inline]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get the raw count for an n-gram.
    #[inline]
    pub fn count(&self, tokens: &[&str]) -> u64 {
        self.store.count(tokens)
    }

    /// Compute log probability of a word given context.
    ///
    /// Uses Modified Kneser-Ney smoothing with backoff to lower-order models.
    ///
    /// # Arguments
    ///
    /// * `word` - The word to compute probability for
    /// * `context` - The preceding context words (may be empty for unigram)
    ///
    /// # Returns
    ///
    /// Log probability (base e) of the word given the context.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // P(fox | quick brown)
    /// let log_prob = model.log_prob("fox", &["quick", "brown"]);
    ///
    /// // P(the) - unigram
    /// let log_prob_unigram = model.log_prob("the", &[]);
    /// ```
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        self.smoothing.log_prob(
            word,
            context,
            &self.store,
            self.vocab_size,
            self.total_count,
        )
    }

    /// The Modified Kneser-Ney smoothing parameters (discounts and the
    /// `N₁₊(•,•)` continuation denominator) computed for this model.
    pub fn smoothing(&self) -> &KneserNeySmoothing {
        &self.smoothing
    }

    /// Compute log probability of a complete sentence.
    ///
    /// Sums log probabilities of each word given its context, using the
    /// appropriate context length based on position in the sentence.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The sentence tokens
    ///
    /// # Returns
    ///
    /// Log probability (base e) of the entire sentence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
    /// ```
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let order = self.order();
        let mut total_log_prob = 0.0;

        for i in 0..tokens.len() {
            let word = tokens[i];
            let context_start = i.saturating_sub(order - 1);
            let context = &tokens[context_start..i];
            total_log_prob += self.log_prob(word, context);
        }

        total_log_prob
    }

    /// Check if a word is in the vocabulary (has been seen during training).
    #[inline]
    pub fn in_vocabulary(&self, word: &str) -> bool {
        self.store.entry(&[word]).is_some()
    }

    /// Get the number of n-grams stored in the model.
    #[inline]
    pub fn ngram_count(&self) -> usize {
        self.store.len()
    }

    /// Get the log probability assigned to out-of-vocabulary words.
    ///
    /// This is the uniform distribution over the vocabulary: log(1/V).
    #[inline]
    pub fn oov_log_prob(&self) -> f64 {
        if self.vocab_size == 0 {
            f64::NEG_INFINITY
        } else {
            -(self.vocab_size as f64).ln()
        }
    }
}

impl<S> NgramModel<S>
where
    S: IterableNgramStore,
{
    /// Iterate every stored n-gram as its decoded word sequence and entry.
    ///
    /// Storage-agnostic: the byte-native store decodes term-ids through its
    /// vocabulary, the legacy trie splits its `'|'`-joined key. Consumers that
    /// previously walked `model.trie().iter_entries()` (yielding a raw key)
    /// call this instead and operate on the `words` vec.
    #[inline]
    pub fn iter_ngrams(&self) -> Box<dyn Iterator<Item = (Vec<String>, NgramEntry)> + '_> {
        self.store.iter_ngrams()
    }
}

impl<S> Clone for NgramModel<S>
where
    S: NgramLookup + Clone,
{
    fn clone(&self) -> Self {
        Self {
            store: self.store.clone(),
            smoothing: self.smoothing.clone(),
            vocab_size: self.vocab_size,
            total_count: self.total_count,
        }
    }
}

// ============================================================================
// Portable serialization (doesn't require the store to implement serde).
// ============================================================================

/// The key encoding of a serialized [`PortableNgramModel`].
///
/// `#[serde(default)]` resolves to [`KeyEncoding::LegacyPipe`] for models
/// serialized before this tag existed, so old files keep deserializing.
#[cfg(feature = "serde-extras")]
#[derive(
    serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash,
)]
pub enum KeyEncoding {
    /// Legacy `'|'`-joined token keys (`"the|quick|brown"`). Delimiter-collision
    /// prone; stored in [`PortableNgramModel::entries`].
    #[default]
    LegacyPipe,
    /// LEB128 varint term-id keys stored as a Latin-1 `String` (each byte
    /// `0xNN` → scalar `U+00NN`), with a companion [`PortableVocabulary`];
    /// stored in [`PortableNgramModel::entries`].
    Latin1Varint,
    /// Raw LEB128 varint term-id byte keys, with a companion
    /// [`PortableVocabulary`]; stored in [`PortableNgramModel::entries_bytes`].
    /// The delimiter-collision class cannot occur under this encoding.
    TermIdBytes,
}

/// Portable vocabulary format for serialization.
///
/// Lists the vocabulary words in term-id order, so a self-contained portable
/// model can decode its LEB128 term-id keys back to human-readable words.
///
/// Index `0` corresponds to term-id [`FIRST_VALID_INDEX`](super::vocabulary::FIRST_VALID_INDEX)
/// (`1`), index `1` to term-id `2`, etc. (term-id `0` is reserved).
#[cfg(feature = "serde-extras")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct PortableVocabulary {
    /// Words in term-id order (`words[i]` is the word for term-id `i + 1`).
    pub words: Vec<String>,
}

/// Portable N-gram model format for serialization.
///
/// This format doesn't require the store's backend to implement serde traits,
/// making it compatible with all backends. It carries both the legacy string
/// key list ([`entries`](Self::entries)) and the byte-native term-id key list
/// ([`entries_bytes`](Self::entries_bytes)); [`key_encoding`](Self::key_encoding)
/// selects which one is authoritative. The extra fields are all
/// `#[serde(default)]` so files written before the byte-native migration still
/// deserialize (as [`KeyEncoding::LegacyPipe`]).
#[cfg(feature = "serde-extras")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PortableNgramModel {
    /// N-gram entries as `('|'`-joined-or-Latin-1 `String` key, snapshot) pairs.
    /// Authoritative for [`KeyEncoding::LegacyPipe`] / [`KeyEncoding::Latin1Varint`].
    pub entries: Vec<(String, NgramEntrySnapshot)>,
    /// N-gram entries as (raw LEB128 term-id byte key, snapshot) pairs.
    /// Authoritative for [`KeyEncoding::TermIdBytes`].
    #[serde(default)]
    pub entries_bytes: Vec<(Vec<u8>, NgramEntrySnapshot)>,
    /// Maximum n-gram order.
    pub max_order: usize,
    /// Vocabulary size (unique unigrams).
    pub vocab_size: usize,
    /// Total token count.
    pub total_count: u64,
    /// Smoothing parameters.
    pub smoothing: KneserNeySmoothing,
    /// Optional vocabulary for term-id-keyed models.
    #[serde(default)]
    pub vocabulary: Option<PortableVocabulary>,
    /// Which key list is authoritative.
    #[serde(default)]
    pub key_encoding: KeyEncoding,
    /// Fingerprint of the vocabulary the term-id keys were encoded against.
    /// `None` for legacy files. Verified on load so a mismatched (model, vocab)
    /// pair is rejected rather than silently mis-decoded.
    #[serde(default)]
    pub vocab_fingerprint: Option<u64>,
}

/// Deterministic, version-stable fingerprint of a vocabulary, computed over its
/// **dense** normalized form — present terms in ascending id order, renumbered
/// `FIRST_VALID_INDEX, +1, +2, …` — so it is invariant to the id-burning gaps
/// that parallel interning leaves behind.
///
/// Used to stamp a term-id model with the identity of the vocabulary its keys
/// were encoded against, and to reject a mismatched vocabulary on load. It
/// delegates to [`compute_portable_vocab_fingerprint`] over the very same
/// densified `words` that [`to_portable`](NgramModel::to_portable) emits, so a
/// save-time stamp and a load-time check compute the identical value by
/// construction and can never diverge.
#[cfg(feature = "serde-extras")]
pub(crate) fn compute_vocab_fingerprint(vocab: &SharedVocabARTrie) -> u64 {
    compute_portable_vocab_fingerprint(&PortableNgramModel::portable_vocabulary_from(vocab))
}

/// Fingerprint of the words in a [`PortableVocabulary`] (equivalent to
/// [`compute_vocab_fingerprint`] over the vocabulary those words rebuild).
#[cfg(feature = "serde-extras")]
fn compute_portable_vocab_fingerprint(vocab: &PortableVocabulary) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let len = vocab.words.len() as u64;
    let mut hash = FNV_OFFSET ^ len;
    for (offset, term) in vocab.words.iter().enumerate() {
        for byte in term.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= offset as u64 + 1;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Enumerate a live vocabulary's **present** terms in ascending id order,
/// returning the dense `words` (`words[i]` is the term for dense id
/// `FIRST_VALID_INDEX + i`) together with an `old_id → new_id` remap indexed by
/// old id.
///
/// Parallel interning is only "nearly-dense": a lost write-once race in
/// `PersistentVocabARTrie::insert` burns the id it had already reserved, so the
/// live id space develops gaps and a *winning* term can hold an id greater than
/// the entry count. Densifying here — enumerating the half-open
/// `start_index()..next_index()` and renumbering the survivors
/// `FIRST_VALID_INDEX, +1, +2, …` — restores the dense `{FIRST_VALID_INDEX ..}`
/// space that the portable format and the load path assume. A burned/absent id
/// maps to `0` (never a valid new id); by construction no n-gram key references
/// one, since only the CAS winner's id is ever encoded into a key.
#[cfg(feature = "serde-extras")]
fn densify_vocabulary(vocab: &SharedVocabARTrie) -> (Vec<String>, Vec<u64>) {
    let guard = vocab.as_ref();
    let next = guard.next_index();
    let mut old_to_new = vec![0u64; next as usize];
    let mut words = Vec::with_capacity(guard.len());
    let mut new_id = FIRST_VALID_INDEX;
    for old_id in guard.start_index()..next {
        if let Some(term) = guard.get_term(old_id) {
            old_to_new[old_id as usize] = new_id;
            words.push(term);
            new_id += 1;
        }
    }
    (words, old_to_new)
}

#[cfg(feature = "serde-extras")]
impl PortableNgramModel {
    /// Materialize a dense [`PortableVocabulary`] from a live vocabulary — present
    /// terms in ascending id order (see [`densify_vocabulary`]).
    fn portable_vocabulary_from(vocab: &SharedVocabARTrie) -> PortableVocabulary {
        PortableVocabulary {
            words: densify_vocabulary(vocab).0,
        }
    }
}

/// One-shot reader for the **legacy `'|'`-joined wire format**.
///
/// This is the *only* place in the crate's model path where a `'|'` delimiter is
/// interpreted. It parses each historical `"a|b|c"` key exactly once, resolves
/// (assigning as needed) each token to a term-id in `store`'s vocabulary, and
/// inserts the varint-encoded term-id key. After it returns, no `'|'`-delimited
/// representation exists in memory — the model is fully term-id-keyed.
///
/// It is lossy only for the already-corrupted legacy case where a token itself
/// contained `'|'` (which the term-id format this loader targets represents
/// losslessly) — exactly the bug the migration eliminates going forward.
#[cfg(feature = "serde-extras")]
fn read_legacy_pipe_entries<B: MutableByteMappedDictionary>(
    entries: &[(String, NgramEntrySnapshot)],
    store: &TermIdStore<B>,
) {
    for (key, snapshot) in entries {
        let tokens: Vec<&str> = key.split('|').collect();
        let bytes = encode_ngram_key_bytes(&tokens, store.vocabulary());
        store.insert_raw(&bytes, NgramEntry::from(*snapshot));
    }
}

// --- Byte-native `TermIdStore`-backed export (TermIdBytes) ---

#[cfg(feature = "serde-extras")]
impl<B> NgramModel<TermIdStore<B>>
where
    B: MutableByteMappedDictionary,
{
    /// Export a byte-native term-id model to portable [`KeyEncoding::TermIdBytes`]
    /// form, embedding the store's vocabulary and its fingerprint.
    pub fn to_portable(&self) -> PortableNgramModel {
        let vocab = self.store.vocabulary();
        // Densify the (nearly-dense) live term-ids: emit dense `words` and rewrite
        // every raw n-gram key through the same `old_id → new_id` map, so a
        // parallel-trained model — whose vocabulary has burned ids / gaps — round
        // trips exactly instead of failing the fingerprint check on reload.
        let (words, old_to_new) = densify_vocabulary(vocab);

        let entries_bytes: Vec<(Vec<u8>, NgramEntrySnapshot)> = self
            .store
            .backend()
            .iter_entries_bytes()
            .map(|(key, entry)| {
                let remapped: Vec<u64> = decode_ngram_key_bytes(&key)
                    .into_iter()
                    .map(|id| {
                        let new = old_to_new[id as usize];
                        debug_assert_ne!(new, 0, "n-gram key referenced absent term-id {id}");
                        new
                    })
                    .collect();
                (
                    encode_indices_to_key_bytes(&remapped),
                    NgramEntrySnapshot::from(&entry),
                )
            })
            .collect();

        let vocabulary = PortableVocabulary { words };
        let vocab_fingerprint = Some(compute_portable_vocab_fingerprint(&vocabulary));

        PortableNgramModel {
            entries: Vec::new(),
            entries_bytes,
            max_order: self.store.max_order(),
            vocab_size: self.vocab_size,
            total_count: self.total_count,
            smoothing: self.smoothing.clone(),
            vocabulary: Some(vocabulary),
            key_encoding: KeyEncoding::TermIdBytes,
            vocab_fingerprint,
        }
    }

    /// Save a byte-native term-id model to a portable binary file.
    pub fn save_portable<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let portable = self.to_portable();
        write_portable(&portable, path)
    }
}

// --- Byte-native `TermIdStore`-backed load (unified dispatch + transcode) ---

#[cfg(feature = "serde-extras")]
impl<B> NgramModel<TermIdStore<B>>
where
    B: MutableByteMappedDictionary,
{
    /// Load any portable model into a byte-native term-id model, transcoding
    /// legacy encodings as needed.
    ///
    /// `backend_factory` supplies the (empty) byte backend. The vocabulary is
    /// reconstructed in a fresh temporary [`SharedVocabARTrie`]:
    /// - [`KeyEncoding::TermIdBytes`] / [`KeyEncoding::Latin1Varint`]: the
    ///   embedded [`PortableVocabulary`] is replayed in term-id order (so the
    ///   term-ids of the keys stay valid) and its fingerprint is verified; the
    ///   byte keys are inserted directly.
    /// - [`KeyEncoding::LegacyPipe`]: each `'|'`-joined key is split and
    ///   re-encoded against the fresh vocabulary.
    pub fn load_portable<P, F>(path: P, backend_factory: F) -> crate::Result<Self>
    where
        P: AsRef<Path>,
        F: FnOnce() -> B,
    {
        let portable = read_portable(path)?;
        Self::from_portable(portable, backend_factory)
    }

    /// Reconstruct a byte-native term-id model from a portable struct, rebuilding
    /// a fresh temporary vocabulary from the embedded [`PortableVocabulary`] (see
    /// [`load_portable`](Self::load_portable)).
    pub fn from_portable<F>(portable: PortableNgramModel, backend_factory: F) -> crate::Result<Self>
    where
        F: FnOnce() -> B,
    {
        let (vocab, guard) = rebuild_vocabulary(&portable)?;
        // If `from_portable_with_vocabulary` errors (e.g. fingerprint mismatch on
        // the supplied vocab), `guard` drops here and unlinks the temp directory.
        let mut model = Self::from_portable_with_vocabulary(portable, backend_factory(), vocab)?;
        // Success: hand the temp-directory owner to the store so the directory
        // lives exactly as long as the vocabulary it backs.
        model.store.attach_vocab_tempdir(guard);
        Ok(model)
    }

    /// Reconstruct a byte-native term-id model from a portable struct,
    /// materializing the rebuilt vocabulary at a **caller-supplied**,
    /// caller-owned `vocab_path` instead of a fresh temporary directory.
    ///
    /// This is the placement-controlling counterpart to
    /// [`from_portable`](Self::from_portable): it replays the embedded
    /// [`PortableVocabulary`] into a vocabulary created at `vocab_path` (whose
    /// parent directory is created if missing) and verifies the fingerprint,
    /// exactly as `from_portable` does — but attaches **no** cleanup guard,
    /// because the vocabulary is persistent and owned by the caller. The caller is
    /// responsible for the directory's lifecycle (e.g. a managed, wiped working
    /// directory), so nothing under `std::env::temp_dir()` is created and nothing
    /// is auto-removed.
    pub fn from_portable_at<F, Q>(
        portable: PortableNgramModel,
        backend_factory: F,
        vocab_path: Q,
    ) -> crate::Result<Self>
    where
        F: FnOnce() -> B,
        Q: AsRef<Path>,
    {
        let vocab = rebuild_vocabulary_at(&portable, vocab_path.as_ref())?;
        Self::from_portable_with_vocabulary(portable, backend_factory(), vocab)
    }

    /// Reconstruct a byte-native term-id model from a portable struct against a
    /// *caller-supplied* vocabulary — e.g. the persistent vocabulary a checkpoint
    /// stores alongside the model, so the language model and the spelling
    /// dictionary share one term-id space.
    ///
    /// For term-id encodings the vocabulary's [`compute_vocab_fingerprint`] is
    /// verified against the model's stamped
    /// [`PortableNgramModel::vocab_fingerprint`]; a mismatch is a typed error
    /// (closing the silent-wrong-vocabulary hole). For a legacy `'|'`-joined
    /// source the tokens are re-encoded against (and inserted into) `vocab`.
    pub fn from_portable_with_vocabulary(
        portable: PortableNgramModel,
        backend: B,
        vocab: SharedVocabARTrie,
    ) -> crate::Result<Self> {
        if matches!(
            portable.key_encoding,
            KeyEncoding::TermIdBytes | KeyEncoding::Latin1Varint
        ) {
            if let Some(expected) = portable.vocab_fingerprint {
                let actual = compute_vocab_fingerprint(&vocab);
                if actual != expected {
                    return Err(crate::Error::Model(format!(
                        "vocabulary fingerprint mismatch: model stamped {expected:#018x}, \
                         supplied vocabulary hashes to {actual:#018x}"
                    )));
                }
            }
        }

        let store = TermIdStore::new(backend, vocab, portable.max_order);

        match portable.key_encoding {
            KeyEncoding::TermIdBytes => {
                for (key, snapshot) in &portable.entries_bytes {
                    store.insert_raw(key, NgramEntry::from(*snapshot));
                }
            }
            KeyEncoding::Latin1Varint => {
                // The Latin-1 `String` key is the raw varint byte key with each
                // byte lifted into `U+00NN`; recover the bytes losslessly.
                for (key, snapshot) in &portable.entries {
                    let bytes: Vec<u8> = key.chars().map(|c| c as u8).collect();
                    store.insert_raw(&bytes, NgramEntry::from(*snapshot));
                }
            }
            KeyEncoding::LegacyPipe => {
                // Historical `'|'`-joined wire format — read once, here, into
                // term-id keys (see `read_legacy_pipe_entries`).
                read_legacy_pipe_entries(&portable.entries, &store);
            }
        }

        Ok(Self {
            store,
            smoothing: portable.smoothing,
            vocab_size: portable.vocab_size,
            total_count: portable.total_count,
        })
    }
}

/// Replay a portable model's embedded [`PortableVocabulary`] into an
/// already-created vocabulary, verifying the fingerprint for term-id encodings.
///
/// For [`KeyEncoding::TermIdBytes`] / [`KeyEncoding::Latin1Varint`] the embedded
/// words are inserted in term-id order (assigning term-ids `1, 2, …`, matching
/// the original 1-based ordering) so the keys' term-ids stay valid, and the
/// stamped [`PortableNgramModel::vocab_fingerprint`] is checked against the
/// reconstructed vocabulary. For [`KeyEncoding::LegacyPipe`] this is a no-op (the
/// vocabulary is populated lazily during transcode).
///
/// Shared by the temp-directory materializer ([`rebuild_vocabulary`]) and the
/// caller-path materializer ([`rebuild_vocabulary_at`]); it is agnostic to where
/// `vocab` was created.
#[cfg(feature = "serde-extras")]
fn replay_portable_vocabulary(
    vocab: &SharedVocabARTrie,
    portable: &PortableNgramModel,
) -> crate::Result<()> {
    if matches!(
        portable.key_encoding,
        KeyEncoding::TermIdBytes | KeyEncoding::Latin1Varint
    ) {
        let portable_vocab = portable.vocabulary.as_ref().ok_or_else(|| {
            crate::Error::Model(
                "term-id portable model is missing its embedded vocabulary".to_string(),
            )
        })?;

        {
            let vocab_ref = vocab.as_ref();
            for word in &portable_vocab.words {
                vocab_ref
                    .insert(word)
                    .map_err(|e| crate::Error::Model(format!("vocabulary rebuild failed: {e}")))?;
            }
        }

        if let Some(expected) = portable.vocab_fingerprint {
            let actual = compute_portable_vocab_fingerprint(portable_vocab);
            if actual != expected {
                return Err(crate::Error::Model(format!(
                    "vocabulary fingerprint mismatch: model stamped {expected:#018x}, \
                     embedded vocabulary hashes to {actual:#018x}"
                )));
            }
        }
    }

    Ok(())
}

/// Rebuild a [`SharedVocabARTrie`] from a portable model in a fresh temporary
/// directory, returning it together with the [`VocabTempDir`] guard that removes
/// the directory when the owning store (and all its clones) drop.
///
/// The guard is armed *before* the vocabulary is created, so every early-return
/// error path — a failed vocabulary creation, a missing embedded vocabulary, or a
/// fingerprint mismatch inside [`replay_portable_vocabulary`] — unlinks the temp
/// directory as the local `guard` drops. On success the guard is handed back to
/// be co-owned by the model's store (see [`from_portable`](NgramModel::from_portable)).
#[cfg(feature = "serde-extras")]
fn rebuild_vocabulary(
    portable: &PortableNgramModel,
) -> crate::Result<(SharedVocabARTrie, VocabTempDir)> {
    let (dir, guard) = create_guarded_temp_vocab_dir("vocab")?;
    let vocab = create_vocabulary(&dir.join("vocabulary"))
        .map_err(|e| crate::Error::Model(format!("vocabulary rebuild failed: {e}")))?;
    replay_portable_vocabulary(&vocab, portable)?;
    Ok((vocab, guard))
}

/// Rebuild a [`SharedVocabARTrie`] from a portable model at a caller-supplied,
/// caller-owned `vocab_path` — no temporary directory and no cleanup guard.
///
/// The parent directory of `vocab_path` is created if missing. Backs
/// [`NgramModel::from_portable_at`]; the returned vocabulary is persistent and its
/// directory is the caller's to manage.
#[cfg(feature = "serde-extras")]
fn rebuild_vocabulary_at(
    portable: &PortableNgramModel,
    vocab_path: &Path,
) -> crate::Result<SharedVocabARTrie> {
    if let Some(parent) = vocab_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let vocab = create_vocabulary(vocab_path)
        .map_err(|e| crate::Error::Model(format!("vocabulary rebuild failed: {e}")))?;
    replay_portable_vocabulary(&vocab, portable)?;
    Ok(vocab)
}

/// Serialize a portable model to `path` with bincode.
#[cfg(feature = "serde-extras")]
fn write_portable<P: AsRef<Path>>(portable: &PortableNgramModel, path: P) -> crate::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    bincode::serialize_into(writer, portable)?;
    Ok(())
}

/// Deserialize a portable model from `path` with bincode.
#[cfg(feature = "serde-extras")]
fn read_portable<P: AsRef<Path>>(path: P) -> crate::Result<PortableNgramModel> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let portable = bincode::deserialize_from(reader)?;
    Ok(portable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::PlaintextReader;
    use crate::ngram::store::TermIdStore;
    use crate::ngram::TrainerBuilder;
    use libdictenstein::dynamic_dawg::DynamicDawg;
    use std::io::Write;
    use tempfile::TempDir;

    /// In-memory byte-native model produced by the trainer.
    type TestModel = NgramModel<TermIdStore<DynamicDawg<NgramEntry>>>;

    fn create_test_corpus(dir: &std::path::Path, content: &str) -> std::path::PathBuf {
        let path = dir.join("test.txt");
        let mut file = std::fs::File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    fn create_test_ngram_model() -> TestModel {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox \
                       the quick brown fox the quick brown dog the lazy fox";
        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let dictionary = DynamicDawg::<NgramEntry>::new();
        TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("N-gram training failed")
    }

    #[test]
    fn test_model_properties() {
        let model = create_test_ngram_model();
        assert_eq!(model.order(), 3);
        assert!(model.vocab_size() > 0);
        assert!(model.total_count() > 0);
    }

    #[test]
    fn test_log_prob() {
        let model = create_test_ngram_model();

        // Test known n-gram
        let log_prob = model.log_prob("fox", &["brown"]);
        assert!(log_prob.is_finite());
        assert!(log_prob <= 0.0);

        // Test unigram
        let unigram_prob = model.log_prob("the", &[]);
        assert!(unigram_prob.is_finite());
    }

    #[test]
    fn test_log_prob_finite_for_unseen_ngrams() {
        // Regression for the Modified Kneser-Ney backoff bug: an unseen n-gram whose
        // context IS seen must still yield a finite (backed-off) log probability.
        let model = create_test_ngram_model();

        // "lazy" is a seen context, but "lazy dog" never occurs in the corpus.
        let unseen = model.log_prob("dog", &["lazy"]);
        assert!(
            unseen.is_finite(),
            "unseen bigram must be finite, got {unseen}"
        );
        assert!(unseen <= 0.0);

        // A higher-order unseen n-gram backs off through multiple levels.
        let unseen3 = model.log_prob("dog", &["the", "lazy"]);
        assert!(
            unseen3.is_finite(),
            "unseen trigram must be finite, got {unseen3}"
        );

        // An out-of-vocabulary word and an empty-token continuation must be finite too.
        assert!(model.log_prob("zzz_oov", &["the"]).is_finite());
        assert!(model.log_prob("", &["the"]).is_finite());
    }

    #[test]
    fn test_sentence_log_prob() {
        let model = create_test_ngram_model();

        let log_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[cfg(feature = "serde-extras")]
    #[test]
    fn test_ngram_save_load_roundtrip() {
        let model = create_test_ngram_model();
        let temp_file = tempfile::NamedTempFile::new().expect("Failed to create temp file");

        // Save the model in portable (term-id) format.
        model
            .save_portable(temp_file.path())
            .expect("Failed to save model");

        let metadata = std::fs::metadata(temp_file.path()).expect("Failed to get file metadata");
        assert!(metadata.len() > 0, "Saved model file should not be empty");

        // Load it back into a fresh byte-native store.
        let loaded: TestModel =
            TestModel::load_portable(temp_file.path(), DynamicDawg::<NgramEntry>::new)
                .expect("Failed to load model");

        assert_eq!(model.order(), loaded.order());
        assert_eq!(model.vocab_size(), loaded.vocab_size());
        assert_eq!(model.total_count(), loaded.total_count());

        let orig_prob = model.log_prob("fox", &["the", "quick"]);
        let loaded_prob = loaded.log_prob("fox", &["the", "quick"]);
        assert!(
            probs_equal(orig_prob, loaded_prob),
            "Log probabilities should match after roundtrip: {orig_prob} vs {loaded_prob}"
        );

        let orig_sentence_prob = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        let loaded_sentence_prob = loaded.sentence_log_prob(&["the", "quick", "brown", "fox"]);
        assert!(
            probs_equal(orig_sentence_prob, loaded_sentence_prob),
            "Sentence log probabilities should match: {orig_sentence_prob} vs {loaded_sentence_prob}"
        );
    }

    /// Helper to compare probabilities that may be -inf.
    #[cfg(feature = "serde-extras")]
    fn probs_equal(a: f64, b: f64) -> bool {
        if a.is_infinite() && b.is_infinite() {
            a.signum() == b.signum()
        } else if a.is_nan() || b.is_nan() {
            false
        } else {
            (a - b).abs() < 1e-10
        }
    }

    #[test]
    fn test_pathmap_model() {
        // A trigram model over the natural in-memory byte backend answers a finite
        // log probability (smoke test that training + scoring compose).
        let dir = TempDir::new().expect("Failed to create temp dir");
        let content = "the quick brown fox";
        let path = create_test_corpus(dir.path(), content);
        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");

        let dictionary = DynamicDawg::<NgramEntry>::new();
        let model = TrainerBuilder::new(dictionary)
            .order(3)
            .train(reader)
            .expect("N-gram training failed");

        let log_prob = model.log_prob("fox", &["brown"]);
        assert!(log_prob.is_finite());
    }
}
