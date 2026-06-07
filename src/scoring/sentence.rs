//! Sentence-level scoring utilities.

use crate::ngram::NgramModel;
use liblevenshtein::dictionary::MutableMappedDictionary;

/// Sentence scorer providing various scoring methods.
pub struct SentenceScorer<'a, D>
where
    D: MutableMappedDictionary<Value = crate::ngram::NgramEntry>,
{
    model: &'a NgramModel<D>,
}

impl<'a, D> SentenceScorer<'a, D>
where
    D: MutableMappedDictionary<Value = crate::ngram::NgramEntry>,
{
    /// Create a new sentence scorer.
    pub fn new(model: &'a NgramModel<D>) -> Self {
        Self { model }
    }

    /// Compute log probability of a sentence.
    ///
    /// Returns the sum of log P(w_i | context) for each word.
    pub fn log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    /// Compute normalized log probability (per-word average).
    ///
    /// Useful for comparing sentences of different lengths.
    pub fn normalized_log_prob(&self, tokens: &[&str]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        self.log_prob(tokens) / tokens.len() as f64
    }

    /// Compute sentence perplexity.
    ///
    /// PPL = exp(-1/N * log P(sentence))
    pub fn perplexity(&self, tokens: &[&str]) -> f64 {
        let normalized = self.normalized_log_prob(tokens);
        (-normalized).exp()
    }

    /// Score multiple sentence candidates and rank them.
    ///
    /// Returns sentences sorted by log probability (highest first).
    pub fn rank_sentences<'b>(&self, sentences: &[&'b [&'b str]]) -> Vec<(&'b [&'b str], f64)> {
        let mut scored: Vec<_> = sentences.iter().map(|s| (*s, self.log_prob(s))).collect();

        // Sort by score descending (highest log prob first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
    }

    /// Find the best sentence among candidates.
    pub fn best_sentence<'b>(&self, sentences: &[&'b [&'b str]]) -> Option<(&'b [&'b str], f64)> {
        sentences
            .iter()
            .map(|s| (*s, self.log_prob(s)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}
