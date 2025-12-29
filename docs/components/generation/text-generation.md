# Text Generation with N-gram Language Models

Text generation is the task of producing coherent text sequences given a starting prompt. This document explains how libgrammstein's `TextGenerator` uses autoregressive sampling to generate text from trained N-gram language models.

## What is Autoregressive Generation?

Language models assign probabilities to word sequences. Given a context (preceding words), they predict the probability distribution over possible next words:

```
P(next_word | context)
```

**Autoregressive generation** uses this capability to produce text one token at a time:

1. Start with a prompt (initial context)
2. Query the model for P(w | context) for all vocabulary words w
3. Select a next word using a sampling strategy
4. Append the selected word to the context
5. Repeat until a stopping condition is met

```
┌─────────────────────────────────────────────────────────────────┐
│                  Autoregressive Generation Loop                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Prompt: ["the", "quick"]                                      │
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   Context    │────▶│   Model      │────▶│   Sample     │   │
│   │ [the, quick] │     │ P(w|context) │     │   "brown"    │   │
│   └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                     │           │
│                        ┌────────────────────────────┘           │
│                        ▼                                        │
│   ┌──────────────────────────────────────────┐                  │
│   │   Context: [the, quick, brown]           │                  │
│   │   Generated: [brown]                     │                  │
│   └──────────────────────────────────────────┘                  │
│                        │                                        │
│                        ▼  (repeat)                              │
│   ┌──────────────────────────────────────────┐                  │
│   │   Context: [quick, brown, fox]           │                  │
│   │   Generated: [brown, fox]                │                  │
│   └──────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Mathematically, the probability of generating a sequence factorizes as:

```
P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ | w₁, ..., wᵢ₋₁)
```

## The Vocabulary Problem

To sample from the next-word distribution, we must compute probabilities for all possible next words. This requires:

1. **Knowing the vocabulary**: The set of all words the model has seen
2. **Computing probabilities**: Querying P(w | context) for each vocabulary word

### Extracting Vocabulary from the Model

libgrammstein extracts the vocabulary by iterating over unigrams in the N-gram trie:

```rust
fn extract_vocabulary(model: &NgramModel<D>) -> Vec<String> {
    let mut vocab: HashSet<String> = HashSet::new();

    for (key, _) in model.trie().iter_entries() {
        // Unigrams have no separator character
        if !key.contains(NGRAM_SEPARATOR) {
            vocab.insert(key);
        }
    }

    vocab.into_iter().collect()
}
```

The vocabulary is cached at generator construction time to avoid repeated iteration.

### Computational Complexity

For each generated token, we compute |V| probability queries, where |V| is the vocabulary size. For a vocabulary of 50,000 words and 100 tokens to generate:

```
Total queries = 50,000 × 100 = 5,000,000
```

This is why N-gram models are efficient for generation—each query is O(n) where n is the N-gram order, not O(|V|) like neural models.

## Greedy Decoding

The simplest generation strategy: always pick the highest-probability token.

```
w* = argmax_w P(w | context)
```

### Algorithm

```rust
fn best_token(&self, context: &[&str]) -> Option<String> {
    let mut best_token = None;
    let mut best_score = f64::NEG_INFINITY;

    for word in &self.vocabulary {
        let score = self.model.log_prob(word, context);
        if score > best_score {
            best_score = score;
            best_token = Some(word.clone());
        }
    }

    best_token
}
```

### Properties

| Property | Greedy Decoding |
|----------|-----------------|
| **Deterministic** | Yes - same prompt always produces same output |
| **Diversity** | Low - tends to repeat common patterns |
| **Quality** | High - never selects low-probability tokens |
| **Speed** | Fast - no sampling overhead |

### Example

```rust
use libgrammstein::generation::{TextGenerator, GenerationConfig};

let config = GenerationConfig::greedy().with_max_tokens(10);
let generator = TextGenerator::new(model, config);

// Always produces the same output for the same prompt
let result1 = generator.generate(&["the", "quick"]);
let result2 = generator.generate(&["the", "quick"]);
assert_eq!(result1, result2);
```

### Greedy Configuration

```rust
impl GenerationConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,  // Temperature ≤ 0 triggers greedy mode
            top_p: 1.0,
            top_k: Some(1),    // Only consider top-1 token
            ..Default::default()
        }
    }
}
```

## Temperature Scaling

Temperature controls the "sharpness" of the probability distribution before sampling.

### The Problem

Consider a distribution over three words:

| Word | P(word) |
|------|---------|
| "fox" | 0.50 |
| "dog" | 0.30 |
| "cat" | 0.20 |

Sampling directly might feel too random. Temperature lets us control this.

### Mathematical Formulation

Given log probabilities, temperature τ adjusts the distribution:

```
P'(w | context) = exp(log P(w | context) / τ) / Z

where Z = Σ_v exp(log P(v | context) / τ)
```

### Effect of Temperature

| Temperature | Effect | Distribution |
|-------------|--------|--------------|
| τ < 1.0 | **Sharper** | High-probability words get even higher probability |
| τ = 1.0 | **Neutral** | Original distribution unchanged |
| τ > 1.0 | **Flatter** | Probabilities become more uniform |
| τ → 0 | **Greedy** | Converges to argmax |
| τ → ∞ | **Uniform** | All words equally likely |

### Visual Example

For P(fox)=0.50, P(dog)=0.30, P(cat)=0.20:

```
τ = 0.5 (sharper):     τ = 1.0 (neutral):    τ = 2.0 (flatter):
fox: 0.71              fox: 0.50             fox: 0.39
dog: 0.20              dog: 0.30             dog: 0.33
cat: 0.09              cat: 0.20             cat: 0.28
```

### Implementation

```rust
// Apply temperature scaling to log probabilities
if self.config.temperature != 1.0 {
    let inv_temp = 1.0 / self.config.temperature;
    for (_, log_prob) in &mut candidates {
        *log_prob *= inv_temp;  // log(p^(1/τ)) = log(p)/τ
    }
}
```

### Numerical Stability

To convert temperature-scaled log probabilities to probabilities without overflow:

```rust
// Find maximum for numerical stability
let max_log_prob = candidates.iter()
    .map(|(_, lp)| *lp)
    .fold(f64::NEG_INFINITY, f64::max);

// Subtract max before exp (log-sum-exp trick)
let probs: Vec<(String, f64)> = candidates.into_iter()
    .map(|(word, lp)| {
        let prob = (lp - max_log_prob).exp();
        (word, prob)
    })
    .collect();
```

This ensures the largest exponent is 0, preventing overflow.

## Top-k Sampling

Restrict sampling to the k highest-probability tokens, then sample from this reduced set.

### Motivation

Even with temperature, very low-probability tokens might occasionally be selected. Top-k filtering provides a hard cutoff.

### Algorithm

```rust
// Sort by probability (descending)
probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

// Keep only top-k
if let Some(k) = self.config.top_k {
    probs.truncate(k);
}
```

### Trade-offs

| k Value | Diversity | Quality |
|---------|-----------|---------|
| k = 1 | None (greedy) | Highest |
| k = 10 | Low | High |
| k = 50 | Medium | Medium |
| k = 100 | High | Variable |

### Example Configuration

```rust
let config = GenerationConfig {
    temperature: 0.8,
    top_k: Some(40),  // Only sample from top 40 tokens
    top_p: 1.0,       // Disable nucleus sampling
    ..Default::default()
};
```

## Nucleus (Top-p) Sampling

Nucleus sampling (also called "top-p") dynamically adjusts the candidate set based on cumulative probability.

### The Key Insight

Unlike top-k which uses a fixed number of candidates, nucleus sampling adapts:
- When the model is **confident**: few candidates (tight nucleus)
- When the model is **uncertain**: many candidates (wide nucleus)

### Mathematical Definition

Find the smallest set V_p of tokens such that their cumulative probability exceeds threshold p:

```
V_p = argmin_{V' ⊆ V} |V'|  such that  Σ_{w ∈ V'} P(w) ≥ p
```

### Algorithm

```rust
fn nucleus_filter(&self, probs: Vec<(String, f64)>) -> Vec<(String, f64)> {
    let mut cumulative = 0.0;
    let mut filtered = Vec::new();

    // probs is already sorted descending by probability
    for (word, prob) in probs {
        cumulative += prob;
        filtered.push((word, prob));

        if cumulative >= self.config.top_p {
            break;
        }
    }

    filtered
}
```

### Example: Adaptive Behavior

**Confident distribution** (p=0.9):
```
P(fox)=0.85, P(dog)=0.10, P(cat)=0.05
Nucleus: {fox}  (0.85 < 0.9, but adding dog gives 0.95 ≥ 0.9)
Result: 2 candidates
```

**Uncertain distribution** (p=0.9):
```
P(a)=0.15, P(b)=0.14, P(c)=0.13, P(d)=0.12, P(e)=0.11, ...
Nucleus: {a, b, c, d, e, f, g}
Result: 7+ candidates
```

### Comparison: Top-k vs Nucleus

| Aspect | Top-k | Nucleus (Top-p) |
|--------|-------|-----------------|
| Candidates | Fixed count (k) | Variable count |
| Adapts to confidence | No | Yes |
| Typical values | k=40 | p=0.9 |
| Risk of cutting good options | Yes (if k too small) | Lower |
| Risk of including bad options | Yes (if k too large) | Lower |

### Recommended Value

The original paper recommends p=0.9 or p=0.95 for natural language generation.

```rust
let config = GenerationConfig::nucleus(0.9);
```

## Combining Strategies

The complete sampling pipeline applies strategies in order:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sampling Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Compute log P(w|context) for all w ∈ Vocabulary            │
│                         │                                       │
│                         ▼                                       │
│   2. Apply Temperature Scaling                                  │
│      log P'(w) = log P(w) / τ                                   │
│                         │                                       │
│                         ▼                                       │
│   3. Convert to Probabilities (with numerical stability)        │
│      P(w) = exp(log P'(w) - max_log) / Σ                        │
│                         │                                       │
│                         ▼                                       │
│   4. Sort by Probability (descending)                           │
│                         │                                       │
│                         ▼                                       │
│   5. Apply Top-k Filter (if enabled)                            │
│      Keep only top k tokens                                     │
│                         │                                       │
│                         ▼                                       │
│   6. Apply Nucleus Filter (if top_p < 1.0)                      │
│      Keep smallest set with cumulative prob ≥ p                 │
│                         │                                       │
│                         ▼                                       │
│   7. Re-normalize                                               │
│      P'(w) = P(w) / Σ_{w' in filtered} P(w')                    │
│                         │                                       │
│                         ▼                                       │
│   8. Sample from Categorical Distribution                       │
│      Select w with probability P'(w)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Implementation

```rust
fn sample_token(&self, context: &[&str], rng: &mut dyn RngCore) -> Option<String> {
    // Step 1: Compute log probabilities
    let mut candidates: Vec<(String, f64)> = self.vocabulary.iter()
        .map(|word| {
            let log_prob = self.model.log_prob(word, context);
            (word.clone(), log_prob)
        })
        .filter(|(_, lp)| lp.is_finite())
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Step 2: Temperature scaling
    if self.config.temperature != 1.0 {
        let inv_temp = 1.0 / self.config.temperature;
        for (_, log_prob) in &mut candidates {
            *log_prob *= inv_temp;
        }
    }

    // Step 3: Convert to probabilities with numerical stability
    let max_log_prob = candidates.iter()
        .map(|(_, lp)| *lp)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut probs: Vec<(String, f64)> = candidates.into_iter()
        .map(|(word, lp)| {
            let prob = (lp - max_log_prob).exp();
            (word, prob)
        })
        .filter(|(_, p)| *p > self.config.min_prob)
        .collect();

    // Normalize
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    for (_, p) in &mut probs {
        *p /= total;
    }

    // Step 4: Sort descending
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Step 5: Top-k filter
    if let Some(k) = self.config.top_k {
        probs.truncate(k);
    }

    // Step 6: Nucleus filter
    if self.config.top_p < 1.0 {
        probs = self.nucleus_filter(probs);
    }

    // Step 7: Re-normalize
    let total: f64 = probs.iter().map(|(_, p)| *p).sum();
    let weights: Vec<f64> = probs.iter().map(|(_, p)| *p / total).collect();

    // Step 8: Sample
    match WeightedIndex::new(&weights) {
        Ok(dist) => {
            let idx = dist.sample(rng);
            Some(probs[idx].0.clone())
        }
        Err(_) => probs.first().map(|(w, _)| w.clone()),
    }
}
```

## Stop Conditions

Generation terminates when any of these conditions is met:

### 1. Maximum Token Limit

```rust
for _ in 0..self.config.max_tokens {
    // ... generate one token
}
```

### 2. Stop Tokens

Punctuation or special tokens that indicate the end of a coherent unit:

```rust
if self.config.stop_tokens.contains(&token) {
    generated.push(token);  // Include the stop token
    break;
}
```

Default stop tokens: `.`, `!`, `?`

### 3. No Valid Candidates

When the model has no probability mass on any vocabulary word:

```rust
if let Some(token) = next {
    // continue
} else {
    break;  // No valid next token
}
```

## Reproducibility

For testing and debugging, you can set a random seed for reproducible sampling:

```rust
let config = GenerationConfig::nucleus(0.9)
    .with_seed(42);  // Fixed seed for reproducibility
```

### Implementation

```rust
let mut rng: Box<dyn RngCore> = match self.config.seed {
    Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
    None => Box::new(rand::thread_rng()),
};
```

With a fixed seed, the same prompt produces identical output across runs.

## Complete Example

### Training a Model

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::TrainerBuilder;
use liblevenshtein::dictionary::pathmap::PathMapDictionary;

// Prepare corpus
let reader = PlaintextReader::from_file("corpus.txt")?;

// Train 5-gram model
let dictionary = PathMapDictionary::new();
let model = TrainerBuilder::new(dictionary)
    .order(5)
    .train(&reader)?;
```

### Configuring the Generator

```rust
use libgrammstein::generation::{TextGenerator, GenerationConfig};

// Default configuration (nucleus sampling, p=0.9)
let default_gen = TextGenerator::new(model.clone(), GenerationConfig::default());

// Greedy (deterministic)
let greedy_gen = TextGenerator::new(model.clone(), GenerationConfig::greedy());

// Creative (high temperature, nucleus)
let creative_gen = TextGenerator::new(
    model.clone(),
    GenerationConfig::nucleus(0.95)
        .with_temperature(1.2)
        .with_max_tokens(100)
);

// Focused (low temperature)
let focused_gen = TextGenerator::new(
    model.clone(),
    GenerationConfig::nucleus(0.9)
        .with_temperature(0.7)
        .with_max_tokens(50)
);
```

### Generating Text

```rust
let prompt = ["the", "quick", "brown"];

println!("Greedy: {}", greedy_gen.generate(&prompt).join(" "));
println!("Default: {}", default_gen.generate(&prompt).join(" "));
println!("Creative: {}", creative_gen.generate(&prompt).join(" "));
println!("Focused: {}", focused_gen.generate(&prompt).join(" "));
```

### Sample Outputs

Given a model trained on "The quick brown fox jumps over the lazy dog" repeated with variations:

| Strategy | Output |
|----------|--------|
| Greedy | "fox jumps over the lazy dog." |
| Default | "fox runs in the park." |
| Creative | "fox sleeps under the old tree near the river." |
| Focused | "fox jumps over the dog." |

## Configuration Reference

### GenerationConfig Fields

```rust
pub struct GenerationConfig {
    /// Maximum tokens to generate (default: 50)
    pub max_tokens: usize,

    /// Temperature for sampling (default: 1.0)
    /// - 0.0 or less: greedy decoding
    /// - 0.0-1.0: sharper distribution
    /// - 1.0: neutral
    /// - >1.0: flatter distribution
    pub temperature: f64,

    /// Nucleus sampling threshold (default: 0.9)
    /// - 1.0: disabled
    /// - 0.9: typical value
    pub top_p: f64,

    /// Top-k sampling (default: None)
    /// - None: disabled
    /// - Some(k): only consider top k tokens
    pub top_k: Option<usize>,

    /// Minimum probability threshold (default: 1e-10)
    pub min_prob: f64,

    /// Stop tokens (default: [".", "!", "?"])
    pub stop_tokens: Vec<String>,

    /// Random seed for reproducibility (default: None)
    pub seed: Option<u64>,
}
```

### Builder Methods

```rust
GenerationConfig::default()           // Nucleus p=0.9, temp=1.0
GenerationConfig::greedy()            // Deterministic, temp=0
GenerationConfig::nucleus(0.95)       // Custom nucleus threshold

// Chaining
config
    .with_max_tokens(100)
    .with_temperature(0.8)
    .with_seed(42)
    .with_stop_tokens(vec!["</s>".to_string()])
```

## Strategy Selection Guide

| Use Case | Recommended Strategy |
|----------|---------------------|
| Deterministic outputs | `GenerationConfig::greedy()` |
| General text generation | `GenerationConfig::nucleus(0.9)` |
| Creative writing | nucleus(0.95) + temperature(1.2) |
| Focused/factual content | nucleus(0.8) + temperature(0.7) |
| Code generation | greedy or nucleus(0.8) |
| Dialogue | nucleus(0.9) + temperature(1.0) |

## Next Steps

- [N-gram Overview](../ngram/overview.md): Understanding N-gram language models
- [Modified Kneser-Ney](../ngram/modified-kneser-ney.md): The smoothing algorithm behind probability estimation
- [Query API](../ngram/query-api.md): How log_prob() computes probabilities
