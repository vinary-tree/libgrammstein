//! Modified Kneser-Ney smoothing implementation.
//!
//! Modified Kneser-Ney (MKN) is considered the state-of-the-art smoothing
//! technique for n-gram language models. It uses:
//!
//! - Absolute discounting with different discounts for different count levels
//! - Interpolated backoff to lower-order models
//! - Continuation counts for lower-order probability estimation
//!
//! # References
//!
//! - Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing
//!   techniques for language modeling. Computer Speech & Language.

use super::super::store::NgramLookup;

/// Modified Kneser-Ney smoothing parameters and algorithm.
///
/// Uses three discount values (D1, D2, D3+) computed from n-gram counts:
///
/// ```text
/// Y = n1 / (n1 + 2*n2)
/// D1 = 1 - 2*Y * (n2/n1)
/// D2 = 2 - 3*Y * (n3/n2)
/// D3+ = 3 - 4*Y * (n4/n3)
/// ```
///
/// where n1, n2, n3, n4 are counts of n-grams occurring exactly 1, 2, 3, 4 times.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KneserNeySmoothing {
    /// Discount for n-grams with count = 1.
    d1: f64,
    /// Discount for n-grams with count = 2.
    d2: f64,
    /// Discount for n-grams with count >= 3.
    d3_plus: f64,
    /// `N₁₊(•,•)` — the total number of distinct bigram types, i.e. `Σ_w N₁₊(•,w)`.
    ///
    /// This is the correct denominator for the lower-order continuation
    /// probability `P_cont(w) = N₁₊(•,w) / N₁₊(•,•)`. Populated at training time
    /// (see `Trainer::compute_smoothing_params`). `#[serde(default)]` so models
    /// serialized before this field existed load with `0`, which `unigram_prob`
    /// treats as "unknown" and falls back to the previous `vocab_size` denominator.
    #[serde(default)]
    total_bigram_types: u64,
}

impl KneserNeySmoothing {
    /// Create with default discounts for a given order.
    ///
    /// Uses typical discount values when count statistics are not available.
    pub fn new(_order: usize) -> Self {
        Self::default_discounts()
    }

    /// Create smoothing parameters from n-gram count statistics.
    ///
    /// # Arguments
    ///
    /// * `n1` - Number of n-grams occurring exactly once
    /// * `n2` - Number of n-grams occurring exactly twice
    /// * `n3` - Number of n-grams occurring exactly 3 times
    /// * `n4` - Number of n-grams occurring exactly 4 times
    ///
    /// # Returns
    ///
    /// Computed smoothing parameters.
    pub fn from_counts(n1: u64, n2: u64, n3: u64, n4: u64) -> Self {
        // Avoid division by zero
        let n1 = n1.max(1) as f64;
        let n2 = n2.max(1) as f64;
        let n3 = n3.max(1) as f64;
        let n4 = n4.max(1) as f64;

        // Y = n1 / (n1 + 2*n2)
        let y = n1 / (n1 + 2.0 * n2);

        // Compute discounts using Chen & Goodman formula
        let d1 = (1.0 - 2.0 * y * (n2 / n1)).clamp(0.0, 1.0);
        let d2 = (2.0 - 3.0 * y * (n3 / n2)).clamp(0.0, 2.0);
        let d3_plus = (3.0 - 4.0 * y * (n4 / n3)).clamp(0.0, 3.0);

        Self {
            d1,
            d2,
            d3_plus,
            total_bigram_types: 0,
        }
    }

    /// Create with default discount values.
    ///
    /// Uses typical values when count statistics are not available.
    pub fn default_discounts() -> Self {
        Self {
            d1: 0.75,
            d2: 0.85,
            d3_plus: 0.95,
            total_bigram_types: 0,
        }
    }

    /// Attach the corpus-wide `N₁₊(•,•)` total (number of distinct bigram types),
    /// used as the denominator for lower-order continuation probabilities.
    pub fn with_total_bigram_types(mut self, total_bigram_types: u64) -> Self {
        self.total_bigram_types = total_bigram_types;
        self
    }

    /// The corpus-wide `N₁₊(•,•)` total, or `0` if it was never populated
    /// (e.g. a model serialized before this statistic was tracked).
    pub fn total_bigram_types(&self) -> u64 {
        self.total_bigram_types
    }

    /// Get the discount for a given count.
    #[inline]
    fn discount(&self, count: u64) -> f64 {
        match count {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3_plus,
        }
    }

    /// Compute log probability using Modified Kneser-Ney smoothing.
    ///
    /// For the highest order:
    /// ```text
    /// P_MKN(w|h) = max(c(h,w) - D, 0) / c(h) + λ(h) * P_MKN(w|h')
    /// ```
    ///
    /// For lower orders (continuation probability):
    /// ```text
    /// P_cont(w|h') = N_{1+}(•,h',w) / N_{1+}(•,h',•)
    /// ```
    ///
    /// where:
    /// - c(h,w) = count of n-gram (h,w)
    /// - c(h) = count of context h
    /// - D = discount based on count
    /// - λ(h) = interpolation weight
    /// - N_{1+}(•,h',w) = continuation count (unique preceding contexts)
    pub fn log_prob<S>(
        &self,
        word: &str,
        context: &[&str],
        store: &S,
        vocab_size: usize,
        total_count: u64,
    ) -> f64
    where
        S: NgramLookup,
    {
        let prob = self.prob_recursive(word, context, store, vocab_size, total_count, true);
        prob.ln()
    }

    /// Recursive probability computation with backoff.
    fn prob_recursive<S>(
        &self,
        word: &str,
        context: &[&str],
        store: &S,
        vocab_size: usize,
        total_count: u64,
        is_highest_order: bool,
    ) -> f64
    where
        S: NgramLookup,
    {
        if context.is_empty() {
            // Unigram case: use continuation probability or raw probability
            return self.unigram_prob(word, store, vocab_size, total_count, is_highest_order);
        }

        // Build the full n-gram: context + word
        let mut ngram: Vec<&str> = context.to_vec();
        ngram.push(word);

        // Get n-gram count
        let ngram_count = store.count(&ngram);
        let context_count = store.count(context);

        if context_count == 0 {
            // Context not found, backoff to shorter context
            return self.prob_recursive(word, &context[1..], store, vocab_size, total_count, false);
        }

        // Discounted probability
        let discount = self.discount(ngram_count);
        let discounted_count = (ngram_count as f64 - discount).max(0.0);
        let discounted_prob = discounted_count / context_count as f64;

        // Interpolation weight: λ(h) = D · N_{1+}(h,•) / c(h), where N_{1+}(h,•) is the
        // number of unique words following h. The discount that reserves backoff mass
        // must stay positive even when the queried n-gram is UNSEEN: `discount(0) == 0`
        // would otherwise zero both the discounted term and λ, giving prob = 0 and
        // log_prob = -inf. For unseen n-grams fall back to the primary discount d1 so
        // mass is still reserved for the (always-positive) backoff distribution. Seen
        // n-grams (count ≥ 1) keep their existing per-count discount unchanged.
        let lambda_discount = if ngram_count == 0 { self.d1 } else { discount };
        let unique_continuations = store
            .entry(context)
            .map(|e| e.unique_continuations() as f64)
            .unwrap_or(1.0)
            .max(1.0);
        let lambda = (lambda_discount * unique_continuations) / context_count as f64;

        // Backoff probability (always > 0; the recursion bottoms out at unigram_prob).
        let backoff_prob =
            self.prob_recursive(word, &context[1..], store, vocab_size, total_count, false);

        // Interpolated probability. Guard against a degenerate zero (e.g. an unseen
        // n-gram with all-zero discounts) so the log probability stays finite, falling
        // back to the strictly-positive backoff mass.
        let prob = discounted_prob + lambda * backoff_prob;
        if prob > 0.0 {
            prob
        } else {
            backoff_prob
        }
    }

    /// Compute unigram probability.
    fn unigram_prob<S>(
        &self,
        word: &str,
        store: &S,
        vocab_size: usize,
        total_count: u64,
        is_highest_order: bool,
    ) -> f64
    where
        S: NgramLookup,
    {
        let entry = store.entry(&[word]);

        if is_highest_order {
            // Highest order: use raw count
            let count = entry.map(|e| e.count()).unwrap_or(0);
            if count == 0 {
                // OOV: uniform distribution over vocabulary
                return 1.0 / vocab_size as f64;
            }
            count as f64 / total_count as f64
        } else {
            // Lower order: use continuation probability
            // P_cont(w) = N_{1+}(•,w) / N_{1+}(•,•)
            let continuation_count = entry.map(|e| e.continuation_count()).unwrap_or(0);
            if continuation_count == 0 {
                // OOV: uniform distribution
                return 1.0 / vocab_size as f64;
            }
            // Normalize by N₁₊(•,•) = Σ_w N₁₊(•,w) (total distinct bigram types).
            // Models serialized before this statistic existed store 0; fall back
            // to the previous vocab_size approximation in that case.
            let denominator = if self.total_bigram_types > 0 {
                self.total_bigram_types
            } else {
                vocab_size as u64
            };
            continuation_count as f64 / denominator as f64
        }
    }
}

impl Default for KneserNeySmoothing {
    fn default() -> Self {
        Self::default_discounts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discount_computation() {
        // Test with typical count distribution
        let smoothing = KneserNeySmoothing::from_counts(1000, 500, 300, 200);

        // Discounts should be in reasonable ranges
        assert!(smoothing.d1 > 0.0 && smoothing.d1 < 1.0);
        assert!(smoothing.d2 > 0.0 && smoothing.d2 < 2.0);
        assert!(smoothing.d3_plus > 0.0 && smoothing.d3_plus < 3.0);
    }

    #[test]
    fn test_discount_by_count() {
        let smoothing = KneserNeySmoothing::default_discounts();

        assert_eq!(smoothing.discount(0), 0.0);
        assert_eq!(smoothing.discount(1), smoothing.d1);
        assert_eq!(smoothing.discount(2), smoothing.d2);
        assert_eq!(smoothing.discount(3), smoothing.d3_plus);
        assert_eq!(smoothing.discount(100), smoothing.d3_plus);
    }

    #[test]
    fn test_default_discounts() {
        let smoothing = KneserNeySmoothing::default_discounts();

        assert_eq!(smoothing.d1, 0.75);
        assert_eq!(smoothing.d2, 0.85);
        assert_eq!(smoothing.d3_plus, 0.95);
    }

    #[test]
    fn test_total_bigram_types_builder_and_default() {
        // Defaults to 0 ("unknown") so unigram_prob falls back to the vocab_size
        // denominator for models that never populated the statistic.
        assert_eq!(
            KneserNeySmoothing::default_discounts().total_bigram_types(),
            0
        );
        assert_eq!(KneserNeySmoothing::new(3).total_bigram_types(), 0);
        assert_eq!(
            KneserNeySmoothing::from_counts(1, 1, 1, 1).total_bigram_types(),
            0
        );

        let smoothing = KneserNeySmoothing::default_discounts().with_total_bigram_types(42);
        assert_eq!(smoothing.total_bigram_types(), 42);
    }
}
