# OOV Handling

This document describes strategies for handling out-of-vocabulary (OOV) words in libgrammstein language models.

## Understanding OOV

OOV words are tokens not seen during training:

```rust
let model = train_on_corpus(&training_data)?;

// In vocabulary
model.in_vocabulary("the");     // true
model.in_vocabulary("computer"); // true

// Out of vocabulary
model.in_vocabulary("supercalifragilistic"); // false
model.in_vocabulary("COVID-19");              // false (if trained before 2020)
```

## OOV Rates

Typical OOV rates by domain:

| Domain | OOV Rate |
|--------|----------|
| News (large corpus) | 1-3% |
| Social media | 5-15% |
| Medical/Legal | 10-20% |
| Code/Technical | 15-30% |
| User-generated | 10-25% |

## N-gram OOV Handling

### Unknown Word Probability

N-gram models assign a fixed probability to unknown words:

```rust
impl<D> NgramModel<D> {
    pub fn unk_log_prob(&self) -> f64 {
        // Based on held-out estimation or smoothing
        self.unk_log_probability
    }

    pub fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        if !self.in_vocabulary(word) {
            return self.unk_log_prob();
        }
        // ... normal computation
    }
}
```

### UNK Token Training

Train with explicit `<UNK>` token:

```rust
let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .min_word_freq(5)     // Words appearing < 5 times become <UNK>
    .unk_token("<UNK>")
    .train(&corpus)?;
```

### Open Vocabulary

Alternative: Don't use UNK, let smoothing handle it:

```rust
let model = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .min_word_freq(1)     // Keep all words
    .smoothing(Smoothing::ModifiedKneserNey)
    .train(&corpus)?;

// OOV words get backoff probability from shorter contexts
```

## Embedding OOV Handling

### Subword Fallback

Subword embeddings provide vectors for any word:

```rust
impl SubwordEmbedding {
    pub fn word_vector(&self, word: &str) -> Array1<f32> {
        // Check if word is known
        if let Some(idx) = self.word_to_idx.get(word) {
            return self.combine_word_and_subwords(idx, word);
        }

        // OOV: Use only subword embeddings
        let subwords = self.extract_subwords(word);
        self.average_subword_vectors(&subwords)
    }
}
```

**Example**:
```
Word: "unforgettable"
Subwords: ["<un", "unf", "nfo", "for", "org", "rge", "get", "ett", ...]
Vector: Average of subword embeddings
```

### Morphological Decomposition

Subwords capture morphology:

```
"unhappiness" → [<un, unh, nha, hap, app, ppi, pin, ine, nes, ess, ss>]
                 prefix        root                         suffix
```

This allows reasonable vectors for:
- Inflected forms: "running", "ran", "runs"
- Derived words: "unhappy", "happiness", "unhappiness"
- Compounds: "software", "hardware"

## Hybrid OOV Strategies

### Strategy 1: Embedding Fallback

Use n-gram when available, embeddings for OOV:

```rust
let config = HybridConfig {
    strategy: InterpolationStrategy::NgramWithEmbeddingFallback,
    ..Default::default()
};
```

**Behavior**:
```
Known word: P = P_ngram(w|c)
OOV word:   P = P_embed(w|c)
```

### Strategy 2: Dynamic Weighting

Shift weight toward embeddings for OOV:

```rust
let config = HybridConfig {
    strategy: InterpolationStrategy::Dynamic {
        base_alpha: 0.8,  // 80% n-gram for known words
        oov_alpha: 0.2,   // 20% n-gram for OOV (80% embedding)
    },
    ..Default::default()
};
```

### Strategy 3: Context-Aware OOV

Consider OOV rate in context:

```rust
fn adaptive_score(&self, word: &str, context: &[&str]) -> f64 {
    // Count OOV in context
    let oov_ratio = context.iter()
        .filter(|w| !self.ngram.in_vocabulary(w))
        .count() as f64 / context.len().max(1) as f64;

    // Reduce n-gram weight when context is unreliable
    let alpha = self.base_alpha * (1.0 - oov_ratio * 0.5);

    self.interpolate(word, context, alpha)
}
```

## OOV Mitigation Techniques

### 1. Vocabulary Expansion

Add domain-specific terms:

```rust
// Train base model
let mut model = train_on_general_corpus()?;

// Add domain vocabulary
let domain_words = read_domain_vocabulary("medical_terms.txt")?;
for word in domain_words {
    model.add_to_vocabulary(&word)?;
}
```

### 2. Morphological Normalization

Reduce OOV through normalization:

```rust
fn normalize_word(word: &str) -> String {
    let mut normalized = word.to_lowercase();

    // Remove common suffixes
    for suffix in &["'s", "'d", "'ll", "'ve", "ing", "ed", "ly"] {
        if normalized.ends_with(suffix) {
            normalized = normalized[..normalized.len()-suffix.len()].to_string();
        }
    }

    normalized
}
```

### 3. Spell Correction Integration

Correct OOV to known words:

```rust
fn handle_oov(&self, word: &str, context: &[&str]) -> f64 {
    if self.in_vocabulary(word) {
        return self.score(word, context);
    }

    // Try spell correction
    if let Some(corrected) = self.spell_correct(word) {
        // Weight by edit distance
        let confidence = 1.0 / (1.0 + edit_distance(word, &corrected) as f64);
        return self.score(&corrected, context) + confidence.ln();
    }

    self.unk_log_prob()
}
```

### 4. Transliteration

Handle script variations:

```rust
fn normalize_script(word: &str) -> String {
    // Transliterate non-ASCII to ASCII approximations
    unidecode(word)
}
```

## Measuring OOV Impact

### OOV Rate Calculation

```rust
fn oov_rate(model: &NgramModel<D>, test_words: &[&str]) -> f64 {
    let oov_count = test_words.iter()
        .filter(|w| !model.in_vocabulary(w))
        .count();

    oov_count as f64 / test_words.len() as f64
}
```

### Perplexity Breakdown

```rust
fn analyze_perplexity(
    model: &HybridLanguageModel<D>,
    test: &[Vec<String>],
) {
    let mut iv_log_prob = 0.0;
    let mut oov_log_prob = 0.0;
    let mut iv_count = 0;
    let mut oov_count = 0;

    for sentence in test {
        let tokens: Vec<&str> = sentence.iter().map(|s| s.as_str()).collect();
        for (i, token) in tokens.iter().enumerate() {
            let context = &tokens[..i];
            let log_prob = model.score(token, context);

            if model.ngram_model().in_vocabulary(token) {
                iv_log_prob += log_prob;
                iv_count += 1;
            } else {
                oov_log_prob += log_prob;
                oov_count += 1;
            }
        }
    }

    println!("In-vocabulary PPL: {:.2}", (-iv_log_prob / iv_count as f64).exp());
    println!("OOV PPL: {:.2}", (-oov_log_prob / oov_count as f64).exp());
    println!("OOV rate: {:.1}%", oov_count as f64 / (iv_count + oov_count) as f64 * 100.0);
}
```

## Best Practices

1. **Train on representative data**: Include domain-specific text in training

2. **Use subword embeddings**: Essential for OOV handling

3. **Monitor OOV rates**: Track OOV in production for model updates

4. **Dynamic strategies**: Adapt interpolation weights based on OOV

5. **Vocabulary maintenance**: Periodically update vocabulary with new terms

## See Also

- [Interpolation Strategies](interpolation.md) - Strategy details
- [BPE Tokenization](../embedding/bpe.md) - Subword approach
- [Domain Adaptation](../../examples/domain-adaptation.md) - Handling domain shift
