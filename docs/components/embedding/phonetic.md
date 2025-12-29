# Phonetic Embeddings

Combining orthographic and phonetic similarity for error-tolerant word matching.

## What is a Phonetic Embedding?

A phonetic embedding extends standard word embeddings to incorporate how words *sound*, not just how they're spelled. This is critical for:

- **Spell correction**: "fone" → "phone"
- **Homophone detection**: "knight" ≈ "night"
- **OOV handling**: Unknown words matched by pronunciation
- **ASR error recovery**: Acoustic confusions share phonetic similarity

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Phonetic Embedding Space                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     Orthographic Space                    Phonetic Space                    │
│                                                                             │
│        knight ─────────────┐       ┌─────────── knight                     │
│                            │       │                                        │
│        night ──────────────┼───────┼──────────── night   (same sound!)     │
│                            │       │                                        │
│        phone ─────────────┐│       │┌────────── phone                      │
│                           ││       ││                                       │
│        fone ──────────────┼┼───────┼┼────────── fone     (same sound!)     │
│                           ││       ││                                       │
│        cat ───────────────┘│       │└────────── cat                        │
│                            │       │                                        │
│        dog ────────────────┘       └─────────── dog                        │
│                                                                             │
│     Combined: sim = (1-λ)·ortho + λ·phonetic                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How It Works

The `PhoneticEmbedding` combines two similarity measures:

1. **Orthographic similarity**: Character n-gram embeddings (FastText-style)
2. **Phonetic similarity**: Embeddings of phonetically normalized words

### Phonetic Normalization

Words are normalized using Zompist phonetic rewrite rules before computing similarity:

```
"phone"  → normalize → "fon"
"fone"   → normalize → "fon"    ← Same normalized form!

"knight" → normalize → "nait"
"night"  → normalize → "nait"   ← Same normalized form!
```

### Similarity Formula

```
similarity(w₁, w₂) = (1 - λ) × ortho_sim(w₁, w₂) + λ × phonetic_sim(w₁, w₂)

where:
  ortho_sim(w₁, w₂)    = cosine(embed(w₁), embed(w₂))
  phonetic_sim(w₁, w₂) = cosine(embed(normalize(w₁)), embed(normalize(w₂)))
  λ                    = phonetic_weight (default: 0.3)
```

## Terminology

| Term | Definition |
|------|------------|
| **Phonetic weight** | Balance between orthographic and phonetic (0-1) |
| **Normalization** | Converting spelling to phonetic representation |
| **Zompist rules** | 62 verified English spelling-to-pronunciation rules |
| **Homophone** | Words that sound the same but differ in spelling |
| **OOV** | Out-of-vocabulary (unknown word) |

## Creating Phonetic Embeddings

### Basic Usage

```rust
use libgrammstein::embedding::{PhoneticEmbedding, SubwordEmbedding};

// Start with trained orthographic embeddings
let ortho_model = SubwordEmbedding::load("embeddings.bin")?;

// Wrap with phonetic capabilities
let phonetic = PhoneticEmbedding::new(ortho_model);

// Compute combined similarity
let sim = phonetic.similarity("knight", "night");
println!("knight ~ night: {:.3}", sim);  // High similarity!
```

### Configuration

```rust
let phonetic = PhoneticEmbedding::new(ortho_model)
    .with_phonetic_weight(0.3)     // 30% phonetic, 70% orthographic
    .with_cache_size(100_000);     // Cache normalized forms
```

### Using Arc for Shared Ownership

```rust
use std::sync::Arc;

// Share embeddings across threads
let ortho = Arc::new(SubwordEmbedding::load("embeddings.bin")?);
let phonetic = PhoneticEmbedding::from_arc(ortho.clone());

// ortho can still be used elsewhere
```

### Custom Phonetic Rules

```rust
use liblevenshtein::phonetic::zompist_rules;

// Use English Zompist rules (default)
let phonetic = PhoneticEmbedding::new(ortho_model);

// Or provide custom rules for other languages
let my_rules = load_custom_rules("german_phonetic.rules")?;
let phonetic = PhoneticEmbedding::new(ortho_model)
    .with_rules(my_rules);
```

## Similarity Methods

### Combined Similarity

```rust
// Default: combines orthographic and phonetic
let sim = phonetic.similarity("phone", "fone");
// Uses: (1-λ)·ortho + λ·phonetic
```

### Pure Phonetic Similarity

```rust
// Phonetic only (ignores spelling)
let sim = phonetic.phonetic_similarity("knight", "night");
// Compares: embed(normalize("knight")) vs embed(normalize("night"))
```

### Finding Similar Words

```rust
// Find most similar words (combined)
let similar = phonetic.most_similar("phone", 5);
for (word, score) in similar {
    println!("{}: {:.3}", word, score);
}

// Find phonetically similar words
let homophones = phonetic.most_similar_phonetically("night", 5);
// Might include: knight, nite, ...
```

## Phonetic Normalization

### Direct Access

```rust
// Get normalized form of a word
let normalized = phonetic.normalize("knight");
println!("knight → {}", normalized);  // "nait" or similar

// Normalization is cached for efficiency
phonetic.normalize("knight");  // Fast: cache hit
```

### Cache Management

```rust
// Check cache size
println!("Cached: {} normalizations", phonetic.cache_size());

// Clear cache (useful if memory constrained)
phonetic.clear_cache();
```

## Zompist Phonetic Rules

The default rules are based on the Zompist English spelling-to-pronunciation system, formally verified in Coq/Rocq.

### Rule Categories

| Category | Examples | Description |
|----------|----------|-------------|
| **Affrication** | ch→tʃ, j→dʒ | Combine stops with fricatives |
| **Digraphs** | ph→f, th→θ | Two letters → one sound |
| **Initial clusters** | kn→n, wr→r | Silent initial consonants |
| **Soft c/g** | cent→sent | Before e, i, y |
| **Silent letters** | knight→nait | gh, kn, wr, etc. |
| **Vowel digraphs** | ea→i, oo→u | Combined vowel sounds |

### Formal Verification

The rules are verified to have these properties:

1. **Termination**: Always reaches a fixed point
2. **Bounded expansion**: Output ≤ input + 20 characters
3. **Idempotence**: Normalizing twice = normalizing once
4. **Determinism**: Same input always produces same output

## Thread Safety

`PhoneticEmbedding` is thread-safe (`Send + Sync`):

```rust
use std::sync::Arc;
use std::thread;

let phonetic = Arc::new(PhoneticEmbedding::new(ortho_model));

// Use from multiple threads
let handles: Vec<_> = (0..4).map(|i| {
    let phonetic = phonetic.clone();
    thread::spawn(move || {
        let sim = phonetic.similarity("phone", "fone");
        println!("Thread {}: {:.3}", i, sim);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

Implementation details:
- Embeddings stored in `Arc<SubwordEmbedding>`
- Normalization cache uses `DashMap` (lock-free concurrent hash map)

## Configuration Trade-offs

### Phonetic Weight

| Weight | Behavior | Use Case |
|--------|----------|----------|
| 0.0 | Pure orthographic | Standard word similarity |
| 0.3 | Mild phonetic boost | Spell correction (default) |
| 0.5 | Balanced | ASR error recovery |
| 0.7 | Strong phonetic | Homophone detection |
| 1.0 | Pure phonetic | Pronunciation matching |

### Cache Size

| Size | Memory | Benefit |
|------|--------|---------|
| 10,000 | ~1 MB | Common words |
| 100,000 | ~10 MB | Large vocabulary (default) |
| 1,000,000 | ~100 MB | Full coverage |

## Complete Example: Spell Correction

```rust
use libgrammstein::embedding::{PhoneticEmbedding, SubwordEmbedding};
use std::collections::HashSet;

fn spell_correct(
    query: &str,
    vocabulary: &HashSet<String>,
    phonetic: &PhoneticEmbedding,
    threshold: f64,
) -> Vec<(String, f64)> {
    // Find candidates by phonetic similarity
    let mut candidates: Vec<(String, f64)> = vocabulary
        .iter()
        .map(|word| (word.clone(), phonetic.similarity(query, word)))
        .filter(|(_, sim)| *sim >= threshold)
        .collect();

    // Sort by similarity (descending)
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return top candidates
    candidates.into_iter().take(5).collect()
}

fn main() {
    let ortho_model = SubwordEmbedding::load("embeddings.bin").unwrap();
    let phonetic = PhoneticEmbedding::new(ortho_model)
        .with_phonetic_weight(0.4);

    let vocabulary: HashSet<String> = load_vocabulary();

    // Correct a misspelling
    let corrections = spell_correct("fone", &vocabulary, &phonetic, 0.7);

    println!("Corrections for 'fone':");
    for (word, score) in corrections {
        println!("  {} ({:.3})", word, score);
    }
    // Output:
    //   phone (0.95)
    //   phones (0.82)
    //   phoneme (0.75)
}
```

## Integration with ASR

Phonetic embeddings integrate with lling-llang's `PhoneticRescoreLayer`:

```rust
use libgrammstein::embedding::PhoneticEmbedding;
use lling_llang::layers::PhoneticRescoreLayer;

// Create phonetic embedding
let phonetic = PhoneticEmbedding::new(ortho_model)
    .with_phonetic_weight(0.3);

// Use in lling-llang lattice rescoring
// (See lling-llang docs for full integration)
```

## Performance

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `similarity()` | O(n) | n = word length |
| `normalize()` | O(n × r) | r = number of rules |
| `most_similar()` | O(V) | V = vocabulary size |

### Benchmarks

```
similarity("phone", "fone"):     ~50 μs
normalize("knight"):             ~10 μs (cached: ~100 ns)
most_similar("phone", 10):       ~5 ms (V=50,000)
```

## Related Documentation

- [Subword Embeddings](overview.md) - Base FastText-style embeddings
- [Acoustic Word Embeddings](acoustic-word.md) - Audio-based embeddings
- [lling-llang Phonetic Rescoring](../../../lling-llang/docs/integration/libgrammstein/phonetic-rescore.md)
