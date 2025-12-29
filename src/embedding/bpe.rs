//! Byte Pair Encoding (BPE) tokenizer.
//!
//! BPE is a subword tokenization algorithm that learns a vocabulary of
//! subword units by iteratively merging the most frequent character pairs.
//!
//! # Algorithm
//!
//! 1. Initialize vocabulary with all characters in the corpus
//! 2. Count all adjacent symbol pairs
//! 3. Merge the most frequent pair into a new symbol
//! 4. Repeat until vocabulary reaches target size
//!
//! # References
//!
//! - Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation
//!   of Rare Words with Subword Units. ACL.

use crate::corpus::CorpusReader;
use crate::Result;

use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// End-of-word marker for BPE tokens.
pub const BPE_END_OF_WORD: &str = "</w>";

/// Unknown token for out-of-vocabulary words.
pub const BPE_UNKNOWN: &str = "<unk>";

/// A merge operation: (left, right) -> merged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MergeOp {
    /// Left symbol.
    pub left: String,
    /// Right symbol.
    pub right: String,
    /// Merged result.
    pub merged: String,
}

impl MergeOp {
    /// Create a new merge operation.
    pub fn new(left: String, right: String) -> Self {
        let merged = format!("{}{}", left, right);
        Self { left, right, merged }
    }
}

/// BPE tokenizer with learned vocabulary and merge operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BpeTokenizer {
    /// Vocabulary: token -> index.
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: index -> token.
    reverse_vocab: Vec<String>,
    /// Ordered merge operations.
    merges: Vec<MergeOp>,
    /// Merge lookup for fast encoding: (left, right) -> priority (lower = earlier).
    merge_ranks: HashMap<(String, String), usize>,
    /// Cache for encoded words (not serialized - reconstructed on load).
    #[serde(skip)]
    cache: DashMap<String, Vec<String>>,
    /// Maximum cache size.
    max_cache_size: usize,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer from vocabulary and merges.
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<MergeOp>) -> Self {
        let mut reverse_vocab = vec![String::new(); vocab.len()];
        for (token, &idx) in &vocab {
            if (idx as usize) < reverse_vocab.len() {
                reverse_vocab[idx as usize] = token.clone();
            }
        }

        let merge_ranks: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, m)| ((m.left.clone(), m.right.clone()), i))
            .collect();

        Self {
            vocab,
            reverse_vocab,
            merges,
            merge_ranks,
            cache: DashMap::new(),
            max_cache_size: 100_000,
        }
    }

    /// Get vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token index, or None if not in vocabulary.
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token from index.
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Check if token is in vocabulary.
    #[inline]
    pub fn contains(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// Encode a word into BPE tokens.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tokens = tokenizer.encode_word("hello");
    /// // Might return: ["hel", "lo</w>"]
    /// ```
    pub fn encode_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        // Check cache first
        if let Some(cached) = self.cache.get(word) {
            return cached.clone();
        }

        // Split into characters with end-of-word marker
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if let Some(last) = symbols.last_mut() {
            last.push_str(BPE_END_OF_WORD);
        }

        // Apply merges greedily
        loop {
            // Find the best merge (lowest rank)
            let mut best_merge: Option<(usize, &MergeOp)> = None;
            let mut best_idx: Option<usize> = None;

            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if best_merge.is_none() || rank < best_merge.unwrap().0 {
                        best_merge = Some((rank, &self.merges[rank]));
                        best_idx = Some(i);
                    }
                }
            }

            // Apply the best merge
            match (best_merge, best_idx) {
                (Some((_, merge)), Some(idx)) => {
                    symbols[idx] = merge.merged.clone();
                    symbols.remove(idx + 1);
                }
                _ => break, // No more merges possible
            }
        }

        // Cache the result (with size limit)
        if self.cache.len() < self.max_cache_size {
            self.cache.insert(word.to_string(), symbols.clone());
        }

        symbols
    }

    /// Encode text into BPE token IDs.
    ///
    /// Words are first split on whitespace, then each word is encoded.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .flat_map(|word| {
                self.encode_word(word)
                    .into_iter()
                    .filter_map(|token| self.token_to_id(&token))
            })
            .collect()
    }

    /// Encode text into BPE token strings.
    pub fn encode_to_tokens(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .flat_map(|word| self.encode_word(word))
            .collect()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<&str> = ids
            .iter()
            .filter_map(|&id| self.id_to_token(id))
            .collect();

        let mut result = String::new();
        for token in tokens {
            if token.ends_with(BPE_END_OF_WORD) {
                result.push_str(&token[..token.len() - BPE_END_OF_WORD.len()]);
                result.push(' ');
            } else {
                result.push_str(token);
            }
        }

        result.trim_end().to_string()
    }

    /// Get the merge operations.
    pub fn merges(&self) -> &[MergeOp] {
        &self.merges
    }

    /// Clear the encoding cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

/// BPE trainer for learning vocabulary from corpus.
pub struct BpeTrainer {
    /// Target vocabulary size.
    vocab_size: usize,
    /// Minimum frequency for a pair to be merged.
    min_frequency: u64,
    /// Special tokens to add to vocabulary.
    special_tokens: Vec<String>,
}

impl BpeTrainer {
    /// Create a new BPE trainer.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_frequency: 2,
            special_tokens: vec![BPE_UNKNOWN.to_string()],
        }
    }

    /// Set minimum frequency for merges.
    pub fn with_min_frequency(mut self, min_freq: u64) -> Self {
        self.min_frequency = min_freq;
        self
    }

    /// Add special tokens.
    pub fn with_special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.special_tokens = tokens;
        self
    }

    /// Train BPE tokenizer from corpus.
    ///
    /// # Algorithm
    ///
    /// 1. Count word frequencies in corpus
    /// 2. Initialize vocabulary with characters
    /// 3. Iteratively merge most frequent pairs until vocab_size reached
    pub fn train<R: CorpusReader>(&self, reader: &R) -> Result<BpeTokenizer> {
        log::info!("Starting BPE training with target vocab size: {}", self.vocab_size);

        // Step 1: Count word frequencies
        let word_freqs = self.count_word_frequencies(reader)?;
        log::info!("Counted {} unique words", word_freqs.len());

        // Step 2: Initialize with characters
        let (mut vocab, mut word_splits) = self.initialize_vocab(&word_freqs);
        log::info!("Initial vocabulary size: {}", vocab.len());

        // Step 3: Learn merges
        let merges = self.learn_merges(&mut vocab, &mut word_splits, &word_freqs);
        log::info!("Learned {} merge operations", merges.len());

        // Build final vocabulary with indices
        let mut final_vocab: HashMap<String, u32> = HashMap::new();
        let mut idx = 0u32;

        // Add special tokens first
        for token in &self.special_tokens {
            final_vocab.insert(token.clone(), idx);
            idx += 1;
        }

        // Add all tokens from vocab
        for token in vocab {
            if !final_vocab.contains_key(&token) {
                final_vocab.insert(token, idx);
                idx += 1;
            }
        }

        Ok(BpeTokenizer::new(final_vocab, merges))
    }

    /// Count word frequencies in corpus.
    fn count_word_frequencies<R: CorpusReader>(
        &self,
        reader: &R,
    ) -> Result<HashMap<String, u64>> {
        let word_counts: DashMap<String, AtomicU64> = DashMap::new();

        let sentences: Vec<String> = reader.sentences().collect();
        if sentences.is_empty() {
            return Err(crate::Error::EmptyCorpus);
        }

        sentences.par_iter().for_each(|sentence| {
            for word in sentence.split_whitespace() {
                let word = word.to_lowercase();
                word_counts
                    .entry(word)
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(1, Ordering::Relaxed);
            }
        });

        Ok(word_counts
            .into_iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect())
    }

    /// Initialize vocabulary with characters from words.
    fn initialize_vocab(
        &self,
        word_freqs: &HashMap<String, u64>,
    ) -> (HashSet<String>, HashMap<String, Vec<String>>) {
        let mut vocab: HashSet<String> = HashSet::new();
        let mut word_splits: HashMap<String, Vec<String>> = HashMap::new();

        for word in word_freqs.keys() {
            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if let Some(last) = chars.last_mut() {
                last.push_str(BPE_END_OF_WORD);
            }

            for ch in &chars {
                vocab.insert(ch.clone());
            }

            word_splits.insert(word.clone(), chars);
        }

        (vocab, word_splits)
    }

    /// Learn merge operations.
    fn learn_merges(
        &self,
        vocab: &mut HashSet<String>,
        word_splits: &mut HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, u64>,
    ) -> Vec<MergeOp> {
        let mut merges = Vec::new();
        let target_merges = self.vocab_size.saturating_sub(vocab.len());

        for iteration in 0..target_merges {
            // Count pair frequencies
            let pair_freqs = self.count_pair_frequencies(word_splits, word_freqs);

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, &count)| (pair.clone(), count));

            match best_pair {
                Some(((left, right), count)) if count >= self.min_frequency => {
                    let merge = MergeOp::new(left.clone(), right.clone());

                    // Add merged token to vocabulary
                    vocab.insert(merge.merged.clone());

                    // Apply merge to all words
                    self.apply_merge(word_splits, &left, &right, &merge.merged);

                    merges.push(merge);

                    if (iteration + 1) % 1000 == 0 {
                        log::debug!(
                            "Merge {}: ({}, {}) -> {} (freq: {})",
                            iteration + 1,
                            left,
                            right,
                            merges.last().unwrap().merged,
                            count
                        );
                    }
                }
                _ => {
                    log::info!("No more pairs to merge at iteration {}", iteration);
                    break;
                }
            }
        }

        merges
    }

    /// Count pair frequencies across all words.
    fn count_pair_frequencies(
        &self,
        word_splits: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, u64>,
    ) -> HashMap<(String, String), u64> {
        let mut pair_freqs: HashMap<(String, String), u64> = HashMap::new();

        for (word, symbols) in word_splits {
            let freq = word_freqs.get(word).copied().unwrap_or(0);
            for i in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// Apply a merge to all word splits.
    fn apply_merge(
        &self,
        word_splits: &mut HashMap<String, Vec<String>>,
        left: &str,
        right: &str,
        merged: &str,
    ) {
        for symbols in word_splits.values_mut() {
            let mut i = 0;
            while i < symbols.len().saturating_sub(1) {
                if symbols[i] == left && symbols[i + 1] == right {
                    symbols[i] = merged.to_string();
                    symbols.remove(i + 1);
                }
                i += 1;
            }
        }
    }
}

/// Extract character n-grams (subwords) from a word.
///
/// This is used for FastText-style subword embeddings.
///
/// # Arguments
///
/// * `word` - The word to extract subwords from
/// * `min_n` - Minimum n-gram length (typically 3)
/// * `max_n` - Maximum n-gram length (typically 6)
///
/// # Returns
///
/// A vector of subword strings.
pub fn extract_subwords(word: &str, min_n: usize, max_n: usize) -> Vec<String> {
    // Add boundary markers
    let marked = format!("<{}>", word);
    let chars: Vec<char> = marked.chars().collect();
    let mut subwords = Vec::new();

    for n in min_n..=max_n {
        if n > chars.len() {
            break;
        }
        for i in 0..=(chars.len() - n) {
            let subword: String = chars[i..i + n].iter().collect();
            subwords.push(subword);
        }
    }

    subwords
}

/// Hash a subword to a bucket index for FastText-style embeddings.
///
/// Uses FNV-1a hash.
#[inline]
pub fn hash_subword(subword: &str, num_buckets: usize) -> usize {
    const FNV_PRIME: u64 = 0x100000001b3;
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;

    let mut hash = FNV_OFFSET;
    for byte in subword.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    (hash % num_buckets as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_op() {
        let merge = MergeOp::new("hel".to_string(), "lo".to_string());
        assert_eq!(merge.merged, "hello");
    }

    #[test]
    fn test_extract_subwords() {
        let subwords = extract_subwords("hello", 3, 4);
        assert!(subwords.contains(&"<he".to_string()));
        assert!(subwords.contains(&"hel".to_string()));
        assert!(subwords.contains(&"ell".to_string()));
        assert!(subwords.contains(&"llo".to_string()));
        assert!(subwords.contains(&"lo>".to_string()));
        assert!(subwords.contains(&"<hel".to_string()));
        assert!(subwords.contains(&"hell".to_string()));
        assert!(subwords.contains(&"ello".to_string()));
        assert!(subwords.contains(&"llo>".to_string()));
    }

    #[test]
    fn test_hash_subword() {
        let hash1 = hash_subword("hello", 100000);
        let hash2 = hash_subword("hello", 100000);
        assert_eq!(hash1, hash2);

        let hash3 = hash_subword("world", 100000);
        assert_ne!(hash1, hash3);

        assert!(hash1 < 100000);
        assert!(hash3 < 100000);
    }

    #[test]
    fn test_bpe_tokenizer_encode() {
        // Create a simple vocabulary
        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("h".to_string(), 1);
        vocab.insert("e".to_string(), 2);
        vocab.insert("l".to_string(), 3);
        vocab.insert(format!("o{}", BPE_END_OF_WORD), 4);
        vocab.insert("he".to_string(), 5);
        vocab.insert("hel".to_string(), 6);
        vocab.insert(format!("lo{}", BPE_END_OF_WORD), 7);

        let merges = vec![
            MergeOp::new("h".to_string(), "e".to_string()),
            MergeOp::new("he".to_string(), "l".to_string()),
            MergeOp::new("l".to_string(), format!("o{}", BPE_END_OF_WORD)),
        ];

        let tokenizer = BpeTokenizer::new(vocab, merges);

        let tokens = tokenizer.encode_word("hello");
        assert_eq!(tokens, vec!["hel".to_string(), format!("lo{}", BPE_END_OF_WORD)]);
    }

    #[test]
    fn test_bpe_decode() {
        let mut vocab = HashMap::new();
        vocab.insert("hel".to_string(), 0);
        vocab.insert(format!("lo{}", BPE_END_OF_WORD), 1);
        vocab.insert("wor".to_string(), 2);
        vocab.insert(format!("ld{}", BPE_END_OF_WORD), 3);

        let tokenizer = BpeTokenizer::new(vocab, vec![]);

        let decoded = tokenizer.decode(&[0, 1, 2, 3]);
        assert_eq!(decoded, "hello world");
    }
}
