//! N-gram language model with probability queries.
//!
//! This module provides the main `NgramModel` struct that combines the n-gram
//! trie with smoothing algorithms for probability estimation.

use crate::ngram::smoothing::KneserNeySmoothing;
use crate::ngram::{NgramEntry, NgramTrie};
use liblevenshtein::dictionary::MutableMappedDictionary;

/// N-gram language model with Modified Kneser-Ney smoothing.
///
/// This is the main interface for training and querying n-gram language models.
///
/// # Type Parameters
///
/// * `D` - Dictionary backend type (e.g., `DynamicDawgChar<NgramEntry>`)
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
pub struct NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// N-gram trie storage.
    trie: NgramTrie<D>,

    /// Smoothing algorithm.
    smoothing: KneserNeySmoothing,

    /// Vocabulary size (number of unique unigrams).
    vocab_size: usize,

    /// Total unigram count (corpus size in tokens).
    total_count: u64,
}

impl<D> NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    /// Create a new n-gram model from a trained trie.
    ///
    /// This is typically called after the training process completes.
    pub fn new(
        trie: NgramTrie<D>,
        smoothing: KneserNeySmoothing,
        vocab_size: usize,
        total_count: u64,
    ) -> Self {
        Self {
            trie,
            smoothing,
            vocab_size,
            total_count,
        }
    }

    /// Get the n-gram order (maximum context length + 1).
    #[inline]
    pub fn order(&self) -> usize {
        self.trie.max_order()
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

    /// Get a reference to the underlying trie.
    #[inline]
    pub fn trie(&self) -> &NgramTrie<D> {
        &self.trie
    }

    /// Get the raw count for an n-gram.
    #[inline]
    pub fn count(&self, tokens: &[&str]) -> u64 {
        self.trie.count(tokens)
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
        self.smoothing.log_prob(word, context, &self.trie, self.vocab_size, self.total_count)
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
        self.trie.contains(&[word])
    }

    /// Get the number of n-grams stored in the model.
    #[inline]
    pub fn ngram_count(&self) -> usize {
        self.trie.len()
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

impl<D> Clone for NgramModel<D>
where
    D: MutableMappedDictionary<Value = NgramEntry>,
{
    fn clone(&self) -> Self {
        Self {
            trie: self.trie.clone(),
            smoothing: self.smoothing.clone(),
            vocab_size: self.vocab_size,
            total_count: self.total_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be added once the full training pipeline is implemented
}
