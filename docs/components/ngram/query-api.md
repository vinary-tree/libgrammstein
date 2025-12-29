# N-gram Query API

This document details the query methods available on `NgramModel`.

## Core Query Methods

### log_prob

Compute log probability of a word given context.

```rust
fn log_prob(&self, word: &str, context: &[&str]) -> f64
```

**Parameters**:
- `word`: Target word to score
- `context`: Preceding words (up to `order - 1`)

**Returns**: Log probability (base e, always negative or zero)

**Example**:
```rust
let prob = model.log_prob("fox", &["quick", "brown"]);
// prob ≈ -2.345 (log probability)

// Convert to probability
let p = prob.exp();  // p ≈ 0.096
```

**Behavior**:
- Uses Modified Kneser-Ney smoothing
- Backs off to shorter contexts if needed
- Returns `unk_log_prob()` for unknown words without context

### prob

Compute probability (not log) of a word given context.

```rust
fn prob(&self, word: &str, context: &[&str]) -> f64
```

Convenience wrapper: `self.log_prob(word, context).exp()`

### sentence_log_prob

Compute log probability of an entire sentence.

```rust
fn sentence_log_prob(&self, tokens: &[&str]) -> f64
```

**Example**:
```rust
let tokens = ["the", "quick", "brown", "fox"];
let log_prob = model.sentence_log_prob(&tokens);
// log_prob ≈ -12.456
```

**Computation**:
```rust
// Equivalent to:
let mut total = 0.0;
for i in 0..tokens.len() {
    let context_start = i.saturating_sub(order - 1);
    total += model.log_prob(&tokens[i], &tokens[context_start..i]);
}
```

### perplexity

Compute perplexity of a sentence.

```rust
fn perplexity(&self, tokens: &[&str]) -> f64
```

**Formula**: `exp(-log_prob / N)` where N is token count

**Interpretation**:
- Lower perplexity = better model fit
- Perplexity of N means model is as uncertain as uniform choice from N words

## Vocabulary Methods

### in_vocabulary

Check if a word is in the model's vocabulary.

```rust
fn in_vocabulary(&self, word: &str) -> bool
```

**Example**:
```rust
if model.in_vocabulary("fox") {
    // Word seen during training
} else {
    // OOV word
}
```

### vocab_size

Get vocabulary size.

```rust
fn vocab_size(&self) -> usize
```

Returns count of unique unigrams.

### ngram_count

Get total n-gram count.

```rust
fn ngram_count(&self) -> usize
```

Returns count of all n-grams (all orders).

## Model Properties

### order

Get the model order.

```rust
fn order(&self) -> usize
```

**Example**:
```rust
let order = model.order();  // 3 for trigram model
let context_len = order - 1;  // Maximum context length
```

### unk_log_prob

Get log probability for unknown words.

```rust
fn unk_log_prob(&self) -> f64
```

Returns the probability assigned to unseen words (typically very low).

## Iteration

### iter_vocabulary

Iterate over vocabulary words.

```rust
fn iter_vocabulary(&self) -> impl Iterator<Item = &str>
```

**Example**:
```rust
for word in model.iter_vocabulary() {
    println!("{}: {}", word, model.prob(word, &[]));
}
```

### iter_ngrams

Iterate over all n-grams.

```rust
fn iter_ngrams(&self) -> impl Iterator<Item = (&str, &NgramEntry)>
```

**Example**:
```rust
for (ngram, entry) in model.iter_ngrams() {
    println!("{}: count={}", ngram, entry.count());
}
```

## Prediction

### predict_next

Get most likely next words.

```rust
fn predict_next(&self, context: &[&str], k: usize) -> Vec<(String, f64)>
```

**Parameters**:
- `context`: Preceding words
- `k`: Number of predictions to return

**Returns**: Vector of (word, log_prob) sorted by probability descending

**Example**:
```rust
let predictions = model.predict_next(&["the", "quick"], 5);
for (word, log_prob) in predictions {
    println!("{}: {:.4}", word, log_prob);
}
// Output:
// brown: -1.234
// fox: -2.345
// dog: -2.567
// ...
```

### sample

Sample a word from the distribution.

```rust
fn sample(&self, context: &[&str], rng: &mut impl Rng) -> String
```

**Example**:
```rust
use rand::thread_rng;

let mut rng = thread_rng();
let word = model.sample(&["the", "quick"], &mut rng);
println!("Sampled: {}", word);
```

### generate

Generate a sequence of words.

```rust
fn generate(&self, seed: &[&str], length: usize, rng: &mut impl Rng) -> Vec<String>
```

**Example**:
```rust
let mut rng = thread_rng();
let text = model.generate(&["the"], 10, &mut rng);
println!("{}", text.join(" "));
// Output: "the quick brown fox jumps over the lazy dog and"
```

## Batch Operations

### batch_log_prob

Score multiple queries efficiently.

```rust
fn batch_log_prob(&self, queries: &[(String, Vec<String>)]) -> Vec<f64>
```

**Example**:
```rust
let queries = vec![
    ("fox".to_string(), vec!["quick".to_string(), "brown".to_string()]),
    ("dog".to_string(), vec!["lazy".to_string()]),
];

let scores = model.batch_log_prob(&queries);
```

### parallel_score_sentences

Score sentences in parallel.

```rust
fn parallel_score_sentences(&self, sentences: &[Vec<&str>]) -> Vec<f64>
```

Uses Rayon for parallel processing.

## Count Access

### get_count

Get raw count for an n-gram.

```rust
fn get_count(&self, ngram: &[&str]) -> Option<u64>
```

**Example**:
```rust
let count = model.get_count(&["the", "quick", "brown"]);
// count = Some(42) or None if not found
```

### get_context_count

Get count of context occurrences.

```rust
fn get_context_count(&self, context: &[&str]) -> u64
```

**Example**:
```rust
let count = model.get_context_count(&["the", "quick"]);
// How many times "the quick" was followed by any word
```

## Query Patterns

### Efficient Batch Scoring

```rust
use rayon::prelude::*;

// Parallel scoring
let scores: Vec<f64> = sentences.par_iter()
    .map(|s| model.sentence_log_prob(s))
    .collect();
```

### Finding Unusual N-grams

```rust
// Find n-grams with low probability
let unusual: Vec<_> = test_ngrams.iter()
    .filter(|ng| model.log_prob(&ng.last().unwrap(), &ng[..ng.len()-1]) < -10.0)
    .collect();
```

### Perplexity Calculation

```rust
fn corpus_perplexity(model: &NgramModel<D>, sentences: &[Vec<&str>]) -> f64 {
    let mut total_log_prob = 0.0;
    let mut total_tokens = 0;

    for sentence in sentences {
        total_log_prob += model.sentence_log_prob(sentence);
        total_tokens += sentence.len();
    }

    (-total_log_prob / total_tokens as f64).exp()
}
```

## See Also

- [Trie Storage](trie-storage.md) - Backend details
- [NgramModel API](../../api/ngram.md) - Complete API reference
- [Training Guide](../../training/ngram.md) - Training workflow
