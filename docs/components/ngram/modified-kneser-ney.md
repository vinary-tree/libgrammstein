# Modified Kneser-Ney Smoothing

Modified Kneser-Ney (MKN) is the state-of-the-art smoothing algorithm for N-gram language models. This document explains how it works and how libgrammstein implements it.

## The Problem: Unseen N-grams

Maximum Likelihood Estimation (MLE) assigns zero probability to N-grams not seen in training:

```
P_MLE(w | context) = count(context w) / count(context)
```

If "the quick purple" never appeared in training, `P_MLE(purple | the quick) = 0`. This causes:
- Zero probability for any sentence containing this N-gram
- Log probability becomes negative infinity

## Smoothing Intuition

Smoothing techniques "steal" probability mass from seen events and redistribute it to unseen events. The key questions are:

1. **How much** probability mass to steal?
2. **How to distribute** it among unseen events?

## Kneser-Ney vs Modified Kneser-Ney

### Original Kneser-Ney

Uses a single discount value D for all N-grams:

```
P_KN(w | context) = max(count(context w) - D, 0) / count(context)
                  + γ(context) × P_KN(w | shorter_context)
```

### Modified Kneser-Ney

Uses **three different discounts** based on the count:

| Count | Discount |
|-------|----------|
| 1 | D₁ |
| 2 | D₂ |
| ≥3 | D₃₊ |

This is based on empirical observation that the optimal discount varies with count.

## The Complete Algorithm

### Step 1: Compute Discount Values

From the training corpus, calculate:

```
n₁ = number of N-grams appearing exactly once
n₂ = number of N-grams appearing exactly twice
n₃ = number of N-grams appearing exactly three times
n₄ = number of N-grams appearing exactly four times

Y = n₁ / (n₁ + 2 × n₂)

D₁ = 1 - 2Y × (n₂ / n₁)
D₂ = 2 - 3Y × (n₃ / n₂)
D₃₊ = 3 - 4Y × (n₄ / n₃)
```

Typical values: D₁ ≈ 0.6, D₂ ≈ 0.8, D₃₊ ≈ 0.9

### Step 2: Compute Highest-Order Probability

For the highest order N (e.g., 5-gram):

```
P_MKN(w | h) = [count(h w) - D(count(h w))]₊ / count(h)
             + γ(h) × P_MKN(w | h')

Where:
- h = history (context)
- h' = history with first word removed (backoff context)
- [x]₊ = max(x, 0)
- D(c) = D₁ if c=1, D₂ if c=2, D₃₊ if c≥3
```

### Step 3: Compute Backoff Weight γ

```
γ(h) = (D₁ × N₁(h) + D₂ × N₂(h) + D₃₊ × N₃₊(h)) / count(h)

Where:
- N₁(h) = number of unique words following h with count 1
- N₂(h) = number of unique words following h with count 2
- N₃₊(h) = number of unique words following h with count ≥ 3
```

### Step 4: Compute Lower-Order Probabilities

For orders below the highest, use **continuation counts** instead of raw counts:

```
P_MKN_lower(w | h) = [N₁₊(• h w) - D(N₁₊(• h w))]₊ / N₁₊(• h •)
                   + γ_lower(h) × P_MKN_lower(w | h')

Where:
- N₁₊(• h w) = number of unique contexts where "h w" appears
             = |{v : count(v h w) > 0}|
- N₁₊(• h •) = total continuation count for history h
             = |{(v, w) : count(v h w) > 0}|
```

### Step 5: Base Case (Unigram)

The unigram level backs off to a uniform distribution:

```
P_MKN(w) = [N₁₊(• w) - D(N₁₊(• w))]₊ / N₁₊(• •)
         + γ_uniform × (1 / |V|)

Where:
- |V| = vocabulary size
```

## Why Continuation Counts?

Consider the word "Francisco". In raw frequency, it's common because of "San Francisco". But it only ever follows "San", so it's not a good predictor for unknown contexts.

Continuation counts measure **versatility**:

| Word | Raw Count | Continuation Count |
|------|-----------|-------------------|
| Francisco | 1000 | 1 (only follows "San") |
| city | 500 | 50 (follows many words) |

Using continuation counts, "city" correctly gets higher lower-order probability than "Francisco".

## libgrammstein Implementation

### KneserNeySmoothing Struct

```rust
#[derive(Clone, Debug)]
pub struct KneserNeySmoothing {
    /// Discount for count = 1
    pub d1: f64,

    /// Discount for count = 2
    pub d2: f64,

    /// Discount for count >= 3
    pub d3_plus: f64,
}

impl KneserNeySmoothing {
    /// Compute discounts from N-gram count statistics
    pub fn from_counts(n1: u64, n2: u64, n3: u64, n4: u64) -> Self {
        let y = n1 as f64 / (n1 + 2 * n2) as f64;

        Self {
            d1: 1.0 - 2.0 * y * (n2 as f64 / n1 as f64),
            d2: 2.0 - 3.0 * y * (n3 as f64 / n2 as f64),
            d3_plus: 3.0 - 4.0 * y * (n4 as f64 / n3 as f64),
        }
    }

    /// Get discount for a given count
    pub fn discount(&self, count: u64) -> f64 {
        match count {
            0 => 0.0,
            1 => self.d1,
            2 => self.d2,
            _ => self.d3_plus,
        }
    }
}
```

### NgramEntry Fields

```rust
pub struct NgramEntry {
    /// Raw count: count(history w)
    pub count: u64,

    /// Continuation count: N₁₊(• history w)
    /// How many unique contexts this N-gram appears in
    pub continuation_count: u32,

    /// Unique continuations: N₁₊(history •)
    /// How many unique words follow this context
    pub unique_continuations: u32,
}
```

### Probability Computation

```rust
impl<D: MutableMappedDictionary<Value = NgramEntry>> NgramModel<D> {
    /// Compute log P(word | context) using Modified Kneser-Ney
    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        self.mkn_prob(word, context, self.order).ln()
    }

    fn mkn_prob(&self, word: &str, context: &[&str], order: usize) -> f64 {
        if order == 0 {
            // Base case: uniform distribution
            return 1.0 / self.vocab_size as f64;
        }

        // Truncate context to current order
        let effective_context = if context.len() >= order - 1 {
            &context[context.len() - (order - 1)..]
        } else {
            context
        };

        // Build key for this N-gram
        let key = self.build_key(effective_context, word);

        // Look up entry
        let (count, context_count, unique_cont) = self.lookup_counts(&key, effective_context);

        if context_count == 0 {
            // Context never seen, back off entirely
            return self.mkn_prob(word, context, order - 1);
        }

        // Use continuation counts for lower orders
        let effective_count = if order == self.order {
            count as f64
        } else {
            self.lookup_continuation_count(&key) as f64
        };

        // Apply discount
        let discount = self.smoothing.discount(effective_count as u64);
        let adjusted = (effective_count - discount).max(0.0);

        // Highest-order probability
        let p_high = adjusted / context_count as f64;

        // Backoff weight
        let gamma = self.compute_gamma(effective_context, context_count, unique_cont);

        // Lower-order probability (recursive)
        let p_low = self.mkn_prob(word, context, order - 1);

        // Combine
        p_high + gamma * p_low
    }

    fn compute_gamma(&self, context: &[&str], context_count: u64, unique_cont: u32) -> f64 {
        // γ = (D₁×N₁ + D₂×N₂ + D₃₊×N₃₊) / count(context)
        // Simplified: use unique_continuations as approximation
        let d_avg = (self.smoothing.d1 + self.smoothing.d2 + self.smoothing.d3_plus) / 3.0;
        (d_avg * unique_cont as f64) / context_count as f64
    }
}
```

## Training: Collecting Continuation Counts

### Two-Pass Algorithm

**Pass 1**: Count all N-grams

```rust
fn pass1_count_ngrams(sentences: impl Iterator<Item = String>) {
    for sentence in sentences {
        let tokens = tokenize(&sentence);
        for n in 1..=order {
            for window in tokens.windows(n) {
                let key = window.join("|");
                dictionary.update_or_insert(&key, NgramEntry::default(), |entry| {
                    entry.count += 1;
                });
            }
        }
    }
}
```

**Pass 2**: Collect continuation counts

```rust
fn pass2_continuation_counts() {
    // For each N-gram, track unique preceding contexts
    for (key, entry) in dictionary.iter() {
        let parts: Vec<&str> = key.split('|').collect();
        if parts.len() > 1 {
            // The shorter suffix is the "continuation"
            let suffix_key = parts[1..].join("|");
            dictionary.update(&suffix_key, |suffix_entry| {
                suffix_entry.continuation_count += 1;
            });
        }
    }

    // For each context, count unique continuations
    for (key, entry) in dictionary.iter() {
        let parts: Vec<&str> = key.split('|').collect();
        if parts.len() > 1 {
            let context_key = parts[..parts.len()-1].join("|");
            dictionary.update(&context_key, |ctx_entry| {
                ctx_entry.unique_continuations += 1;
            });
        }
    }
}
```

## Numerical Stability

### Log-Space Computation

For long sequences, probabilities become very small. libgrammstein works in log space:

```rust
pub fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
    // Sum of log probabilities (avoids underflow)
    tokens.windows(self.order)
        .map(|window| self.log_prob(window.last().unwrap(), &window[..window.len()-1]))
        .sum()
}
```

### Handling Zero Probabilities

MKN guarantees non-zero probabilities through the uniform backoff at unigram level:

```
P(w) ≥ γ_unigram × (1 / |V|) > 0
```

## Perplexity Computation

Perplexity measures how "surprised" the model is by test data:

```rust
pub fn perplexity(&self, tokens: &[&str]) -> f64 {
    let log_prob = self.sentence_log_prob(tokens);
    let avg_log_prob = log_prob / tokens.len() as f64;
    (-avg_log_prob).exp()
}
```

Lower perplexity = better model. Typical values:
- 50-100: Good model on similar domain
- 100-300: Reasonable cross-domain
- 1000+: Poor fit or small training data

## Comparison with Other Smoothing Methods

| Method | Pros | Cons |
|--------|------|------|
| Add-k | Simple | Suboptimal, uniform treatment |
| Good-Turing | Theoretically motivated | Complex implementation |
| Witten-Bell | Intuitive | Less effective than KN |
| Kneser-Ney | State-of-the-art | Single discount |
| **Modified KN** | **Best empirical results** | Most complex |

## Next Steps

- [Trie Storage](trie-storage.md): How N-grams are stored
- [Query API](query-api.md): Complete query interface
- [N-gram Overview](overview.md): Higher-level concepts
