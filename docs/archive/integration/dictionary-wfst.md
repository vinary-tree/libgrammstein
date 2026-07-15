> ⚠ **Archived — superseded, not current design.**
> This describes a `grammstein dictionary export-fst` / OpenFST-text workflow that does **not** exist
> in the codebase (there is no `dictionary` subcommand and no `export_fst_text` / `trie_to_fst`
> symbol in `src/`). The supported WFST path is the `LanguageModel` trait composed with a
> `HierarchicalCorrector`; see
> [`../../integration/lling-llang/overview.md`](../../integration/lling-llang/overview.md) and
> [`../../integration/liblevenshtein/overview.md`](../../integration/liblevenshtein/overview.md).

# Dictionary WFST Integration

This document describes how to integrate libgrammstein dictionaries with Weighted Finite-State Transducers (WFSTs) for spell correction.

## Overview

WFSTs provide efficient error correction by combining:

1. **Dictionary**: Valid words from libgrammstein extraction
2. **Error model**: Character-level edit operations
3. **Language model**: Word-level probabilities

```
Input: "recieve"
        ↓
WFST Composition: Error × Dictionary × LM
        ↓
Output: "receive" (score: 0.95)
```

## Architecture

```
                    ┌─────────────────┐
                    │  libgrammstein  │
                    │   Dictionary    │
                    └────────┬────────┘
                             │
                             ▼
┌────────────┐     ┌─────────────────┐     ┌────────────┐
│   Error    │────>│     WFST        │────>│  Language  │
│   Model    │     │   Composer      │     │   Model    │
└────────────┘     └─────────────────┘     └────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Correction    │
                    │    Candidates   │
                    └─────────────────┘
```

## Dictionary Export

### For WFST Tools

Export dictionary in OpenFST-compatible format:

```rust
use libgrammstein::dictionary::SpellingDictionary;

let dict = SpellingDictionary::load("spelling.dict")?;

// Export as lexicon FST text
dict.export_fst_text("lexicon.txt")?;

// Format:
// word1 word1 0.0
// word2 word2 0.0
// ...
```

### With Frequencies

Include log probabilities from frequency:

```rust
dict.export_fst_text_with_weights("lexicon.txt", |word, entry| {
    // Convert frequency to log probability
    let total = dict.total_tokens() as f64;
    let log_prob = (entry.frequency as f64 / total).ln();
    -log_prob  // Negative log prob (cost)
})?;

// Format:
// the the 2.345
// of of 3.456
// and and 3.567
```

### Trie-Based FST

Direct conversion from DoubleArrayTrieChar:

```rust
use libgrammstein::dictionary::fst_export::trie_to_fst;

let fst = trie_to_fst(dict.trie())?;
fst.write("lexicon.fst")?;
```

## liblevenshtein Integration

### Fuzzy Dictionary Lookup

Use liblevenshtein for candidate generation:

```rust
use liblevenshtein::levenshtein::Levenshtein;

let dict = SpellingDictionary::load("spelling.dict")?;
let lev = Levenshtein::new(dict.trie());

// Find words within edit distance 2
let candidates = lev.search("recieve", 2);
// ["receive", "relieve", "deceive", ...]
```

### Weighted Candidates

Combine edit distance with frequency:

```rust
fn score_candidate(
    dict: &SpellingDictionary,
    original: &str,
    candidate: &str,
) -> f64 {
    let edit_dist = levenshtein_distance(original, candidate);
    let freq = dict.frequency(candidate).unwrap_or(1) as f64;
    let total = dict.total_tokens() as f64;

    // Score = log P(word) - λ × edit_distance
    let lambda = 2.0;  // Edit penalty
    (freq / total).ln() - lambda * edit_dist as f64
}

fn rank_candidates(
    dict: &SpellingDictionary,
    original: &str,
    candidates: Vec<String>,
) -> Vec<(String, f64)> {
    let mut scored: Vec<_> = candidates.into_iter()
        .map(|c| {
            let score = score_candidate(dict, original, &c);
            (c, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored
}
```

## Language Model Integration

### N-gram Rescoring

Combine dictionary with language model:

```rust
use libgrammstein::hybrid::HybridLanguageModel;

fn correct_in_context(
    dict: &SpellingDictionary,
    lm: &HybridLanguageModel<D>,
    text: &[&str],
    position: usize,
) -> String {
    let word = text[position];

    if dict.contains(word) {
        return word.to_string();
    }

    // Get candidates from fuzzy matching
    let lev = Levenshtein::new(dict.trie());
    let candidates = lev.search(word, 2);

    // Score with language model context
    let context_start = position.saturating_sub(2);
    let context = &text[context_start..position];

    let best = candidates.into_iter()
        .map(|c| {
            let lm_score = lm.score(&c, context);
            let edit_score = -edit_distance(word, &c) as f64 * 2.0;
            (c, lm_score + edit_score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    best.map(|(c, _)| c).unwrap_or_else(|| word.to_string())
}
```

### Beam Search Correction

For full sentence correction:

```rust
fn correct_sentence(
    dict: &SpellingDictionary,
    lm: &HybridLanguageModel<D>,
    sentence: &[&str],
    beam_width: usize,
) -> Vec<String> {
    let mut beam: Vec<(Vec<String>, f64)> = vec![(vec![], 0.0)];

    for (i, word) in sentence.iter().enumerate() {
        let candidates = if dict.contains(word) {
            vec![word.to_string()]
        } else {
            let lev = Levenshtein::new(dict.trie());
            lev.search(word, 2)
        };

        let mut new_beam = Vec::new();

        for (prefix, prefix_score) in &beam {
            let context: Vec<&str> = prefix.iter().map(|s| s.as_str()).collect();
            let ctx = &context[context.len().saturating_sub(2)..];

            for candidate in &candidates {
                let lm_score = lm.score(candidate, ctx);
                let edit_score = if candidate == *word { 0.0 } else { -2.0 };

                let mut new_prefix = prefix.clone();
                new_prefix.push(candidate.clone());

                new_beam.push((new_prefix, prefix_score + lm_score + edit_score));
            }
        }

        // Keep top beam_width
        new_beam.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        beam = new_beam.into_iter().take(beam_width).collect();
    }

    beam.into_iter().next().map(|(s, _)| s).unwrap_or_default()
}
```

## OpenFST Integration

### Creating Lexicon FST

```bash
# Generate lexicon from libgrammstein dictionary
grammstein dictionary export-fst ./spelling.dict ./lexicon.txt

# Compile with OpenFST
fstcompile --isymbols=chars.txt --osymbols=words.txt lexicon.txt lexicon.fst

# Optimize
fstarcsort --sort_type=ilabel lexicon.fst lexicon_sorted.fst
```

### Error Model FST

```rust
fn create_error_model_fst(max_edits: usize) -> String {
    let mut fst = String::new();

    // Start state
    fst.push_str("0\n");  // Start state

    for edit in 0..=max_edits {
        // Correct character (no edit)
        for c in 'a'..='z' {
            fst.push_str(&format!("{} {} {} {} 0.0\n", edit, edit, c, c));
        }

        if edit < max_edits {
            // Substitution
            for c1 in 'a'..='z' {
                for c2 in 'a'..='z' {
                    if c1 != c2 {
                        fst.push_str(&format!("{} {} {} {} 1.0\n", edit, edit + 1, c1, c2));
                    }
                }
            }

            // Deletion (input char, epsilon output)
            for c in 'a'..='z' {
                fst.push_str(&format!("{} {} {} <eps> 1.0\n", edit, edit + 1, c));
            }

            // Insertion (epsilon input, output char)
            for c in 'a'..='z' {
                fst.push_str(&format!("{} {} <eps> {} 1.0\n", edit, edit + 1, c));
            }
        }

        // Final state
        fst.push_str(&format!("{} 0.0\n", edit));
    }

    fst
}
```

### Composition

```bash
# Compose error model with lexicon
fstcompose error.fst lexicon.fst > correction.fst

# Add language model
fstcompose correction.fst lm.fst > full_correction.fst

# Find best correction
echo "recieve" | fstshortestpath full_correction.fst | fstprint
```

## Performance Optimization

### Caching Frequent Lookups

```rust
use lru::LruCache;
use std::sync::Mutex;

pub struct CachedCorrector {
    dict: SpellingDictionary,
    lm: HybridLanguageModel<D>,
    cache: Mutex<LruCache<String, Vec<(String, f64)>>>,
}

impl CachedCorrector {
    pub fn correct(&self, word: &str) -> Vec<(String, f64)> {
        // Check cache
        if let Some(cached) = self.cache.lock().unwrap().get(word) {
            return cached.clone();
        }

        // Compute correction
        let result = self.compute_correction(word);

        // Cache result
        self.cache.lock().unwrap().put(word.to_string(), result.clone());

        result
    }
}
```

### Lazy FST Loading

```rust
use once_cell::sync::Lazy;

static CORRECTION_FST: Lazy<CorrectionFst> = Lazy::new(|| {
    CorrectionFst::load("correction.fst").expect("Failed to load FST")
});
```

## Best Practices

1. **Pre-filter candidates**: Use edit distance 1-2 for most cases

2. **Include frequency weights**: Higher frequency → more likely correction

3. **Use language model context**: Context dramatically improves accuracy

4. **Cache frequent queries**: Common misspellings benefit from caching

5. **Tune edit penalties**: Balance edit distance vs. frequency

6. **Consider keyboard layout**: Adjacent keys = cheaper substitution

## See Also

- [Dictionary Overview](../../components/dictionary/overview.md) - Dictionary module
- [Spell Correction Example](../../examples/spell-correction.md) - Complete example
- [Hybrid Model](../../api/hybrid.md) - Language model API
- [liblevenshtein](https://github.com/f1r3fly-io/liblevenshtein-rust) - Fuzzy matching
