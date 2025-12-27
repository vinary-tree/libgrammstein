//! Perplexity evaluation for language models.
//!
//! Perplexity measures how well a language model predicts a corpus.
//! Lower perplexity indicates a better model.

use crate::corpus::CorpusReader;
use crate::ngram::NgramModel;
use crate::Result;

use liblevenshtein::dictionary::MutableMappedDictionary;

/// Perplexity evaluator for language models.
///
/// Perplexity is defined as:
/// PPL = exp(-1/N * Σ log P(w_i | context))
///
/// where N is the total number of tokens.
pub struct Perplexity<'a, D>
where
    D: MutableMappedDictionary<Value = crate::ngram::NgramEntry>,
{
    model: &'a NgramModel<D>,
}

impl<'a, D> Perplexity<'a, D>
where
    D: MutableMappedDictionary<Value = crate::ngram::NgramEntry>,
{
    /// Create a new perplexity evaluator for the given model.
    pub fn new(model: &'a NgramModel<D>) -> Self {
        Self { model }
    }

    /// Compute perplexity over a corpus.
    ///
    /// Returns the perplexity value and statistics.
    pub fn corpus_perplexity<R: CorpusReader>(&self, reader: &R) -> Result<PerplexityResult> {
        let mut total_log_prob = 0.0;
        let mut total_tokens = 0usize;
        let mut oov_count = 0usize;
        let mut sentence_count = 0usize;

        for sentence in reader.sentences() {
            let tokens: Vec<&str> = sentence.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let (log_prob, oov) = self.sentence_log_prob_with_oov(&tokens);
            total_log_prob += log_prob;
            total_tokens += tokens.len();
            oov_count += oov;
            sentence_count += 1;
        }

        if total_tokens == 0 {
            return Err(crate::Error::EmptyCorpus);
        }

        let avg_log_prob = total_log_prob / total_tokens as f64;
        let perplexity = (-avg_log_prob).exp();

        Ok(PerplexityResult {
            perplexity,
            total_log_prob,
            total_tokens,
            oov_count,
            oov_rate: oov_count as f64 / total_tokens as f64,
            sentence_count,
        })
    }

    /// Compute log probability of a sentence.
    pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        self.model.sentence_log_prob(tokens)
    }

    /// Compute log probability with OOV count.
    fn sentence_log_prob_with_oov(&self, tokens: &[&str]) -> (f64, usize) {
        let mut log_prob = 0.0;
        let mut oov_count = 0;
        let order = self.model.order();

        for i in 0..tokens.len() {
            let context_start = i.saturating_sub(order - 1);
            let context: Vec<&str> = tokens[context_start..i].to_vec();
            let word = tokens[i];

            let word_log_prob = self.model.log_prob(word, &context);

            // Check if this was an OOV (backed off to unigram or uniform)
            if word_log_prob <= self.model.oov_log_prob() {
                oov_count += 1;
            }

            log_prob += word_log_prob;
        }

        (log_prob, oov_count)
    }
}

/// Result of perplexity evaluation.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// The perplexity value.
    pub perplexity: f64,

    /// Total log probability.
    pub total_log_prob: f64,

    /// Total number of tokens evaluated.
    pub total_tokens: usize,

    /// Number of out-of-vocabulary tokens.
    pub oov_count: usize,

    /// OOV rate (oov_count / total_tokens).
    pub oov_rate: f64,

    /// Number of sentences evaluated.
    pub sentence_count: usize,
}

impl std::fmt::Display for PerplexityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Perplexity: {:.2} | Tokens: {} | OOV: {:.2}% | Sentences: {}",
            self.perplexity,
            self.total_tokens,
            self.oov_rate * 100.0,
            self.sentence_count
        )
    }
}
