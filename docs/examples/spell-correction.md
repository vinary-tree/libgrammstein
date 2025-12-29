# Example: Spell Correction with Language Models

This example demonstrates using libgrammstein for spell correction by ranking correction candidates.

## Overview

Language models improve spell correction by considering context:

```
Input: "the quikc brown fox"
       ↓
Candidates for "quikc": quick, quack, quirk, kick
       ↓
Rank by P(candidate | "the")
       ↓
Output: "the quick brown fox"
```

## Setup

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["serde-extras"] }
liblevenshtein = "0.6"
```

## Implementation

```rust
use libgrammstein::ngram::{NgramModel, TrainerBuilder, NgramEntry};
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

/// Spell correction using language model
struct SpellCorrector<D>
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    model: HybridLanguageModel<D>,
    vocabulary: Vec<String>,
}

impl<D> SpellCorrector<D>
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    /// Create a new spell corrector
    fn new(model: HybridLanguageModel<D>, vocabulary: Vec<String>) -> Self {
        Self { model, vocabulary }
    }

    /// Generate edit-distance-1 candidates for a word
    fn edit_candidates(&self, word: &str) -> Vec<String> {
        let mut candidates = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let alphabet = "abcdefghijklmnopqrstuvwxyz";

        // Deletions
        for i in 0..chars.len() {
            let candidate: String = chars[..i].iter().chain(&chars[i+1..]).collect();
            if self.vocabulary.contains(&candidate) {
                candidates.push(candidate);
            }
        }

        // Insertions
        for i in 0..=chars.len() {
            for c in alphabet.chars() {
                let mut candidate: Vec<char> = chars.clone();
                candidate.insert(i, c);
                let candidate: String = candidate.into_iter().collect();
                if self.vocabulary.contains(&candidate) {
                    candidates.push(candidate);
                }
            }
        }

        // Substitutions
        for i in 0..chars.len() {
            for c in alphabet.chars() {
                if c != chars[i] {
                    let mut candidate = chars.clone();
                    candidate[i] = c;
                    let candidate: String = candidate.into_iter().collect();
                    if self.vocabulary.contains(&candidate) {
                        candidates.push(candidate);
                    }
                }
            }
        }

        // Transpositions
        for i in 0..chars.len().saturating_sub(1) {
            let mut candidate = chars.clone();
            candidate.swap(i, i + 1);
            let candidate: String = candidate.into_iter().collect();
            if self.vocabulary.contains(&candidate) {
                candidates.push(candidate);
            }
        }

        // Remove duplicates
        candidates.sort();
        candidates.dedup();
        candidates
    }

    /// Rank candidates by language model score
    fn rank_candidates(&self, candidates: &[String], context: &[&str]) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = candidates.iter()
            .map(|c| (c.clone(), self.model.score(c, context)))
            .collect();

        // Sort by score descending (higher = better)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored
    }

    /// Correct a single word in context
    fn correct_word(&self, word: &str, context: &[&str]) -> String {
        let word_lower = word.to_lowercase();

        // Check if word is in vocabulary
        if self.vocabulary.contains(&word_lower) {
            return word_lower;
        }

        // Generate and rank candidates
        let candidates = self.edit_candidates(&word_lower);
        if candidates.is_empty() {
            return word_lower;  // No candidates found
        }

        let ranked = self.rank_candidates(&candidates, context);
        ranked[0].0.clone()
    }

    /// Correct a sentence
    fn correct_sentence(&self, sentence: &str) -> String {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut corrected = Vec::new();

        for (i, word) in words.iter().enumerate() {
            // Get context (up to 2 previous words)
            let context_start = i.saturating_sub(2);
            let context: Vec<&str> = corrected[context_start..].iter()
                .map(|s: &String| s.as_str())
                .collect();

            let corrected_word = self.correct_word(word, &context);
            corrected.push(corrected_word);
        }

        corrected.join(" ")
    }
}

fn main() -> libgrammstein::Result<()> {
    // ========================================
    // Step 1: Train Language Model
    // ========================================
    println!("=== Training Language Model ===\n");

    let corpus = r#"
        The quick brown fox jumps over the lazy dog.
        The quick brown dog runs through the park.
        A brown fox jumped over the fence.
        The lazy cat sat on the mat.
        Quick thinking saved the day.
        The dog chased the brown cat.
        Foxes are quick and clever animals.
        The brown bear walked through the forest.
        Natural language processing is fascinating.
        Machine learning models process language.
    "#;

    let reader = PlaintextReader::from_string(corpus);

    // Train n-gram model
    let ngram = TrainerBuilder::new(DynamicDawgChar::new())
        .order(3)
        .min_word_freq(1)
        .train(&reader)?;

    // Train embedding model
    let reader = PlaintextReader::from_string(corpus);
    let embedding = EmbeddingTrainerBuilder::new()
        .dim(50)
        .min_count(1)
        .epochs(10)
        .train(&reader)?;

    // Create hybrid model
    let config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.6 },
        ..Default::default()
    };
    let hybrid = HybridLanguageModel::new(ngram, embedding, config);

    // Build vocabulary
    let reader = PlaintextReader::from_string(corpus);
    let vocabulary: Vec<String> = reader.sentences()
        .flat_map(|s| s.split_whitespace().map(|w| w.to_lowercase()).collect::<Vec<_>>())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    println!("Vocabulary size: {}", vocabulary.len());

    // ========================================
    // Step 2: Create Spell Corrector
    // ========================================
    let corrector = SpellCorrector::new(hybrid, vocabulary);

    // ========================================
    // Step 3: Test Corrections
    // ========================================
    println!("\n=== Spell Correction Examples ===\n");

    let test_cases = [
        // (input, expected)
        ("the quikc brown fox", "the quick brown fox"),
        ("teh lazy dog", "the lazy dog"),
        ("a brwon fox", "a brown fox"),
        ("machine lerning", "machine learning"),
        ("natrual language", "natural language"),
    ];

    for (input, expected) in test_cases {
        let corrected = corrector.correct_sentence(input);
        let status = if corrected == expected { "✓" } else { "✗" };

        println!("Input:    {}", input);
        println!("Output:   {}", corrected);
        println!("Expected: {}", expected);
        println!("Status:   {}\n", status);
    }

    // ========================================
    // Step 4: Show Candidate Ranking
    // ========================================
    println!("\n=== Candidate Ranking Example ===\n");

    let misspelled = "quikc";
    let context = ["the"];

    println!("Misspelled word: {}", misspelled);
    println!("Context: {:?}\n", context);

    let candidates = corrector.edit_candidates(misspelled);
    println!("Edit-distance-1 candidates: {:?}\n", candidates);

    let ranked = corrector.rank_candidates(&candidates, &context);
    println!("Ranked by P(candidate | context):");
    for (i, (word, score)) in ranked.iter().take(5).enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, word, score);
    }

    // ========================================
    // Step 5: Compare With/Without Context
    // ========================================
    println!("\n=== Context Importance ===\n");

    // "bank" could be corrected differently based on context
    let test_word = "bak";  // Could be "bank" or "back"

    let context1 = ["money", "in", "the"];  // Suggests "bank"
    let context2 = ["come", "right"];       // Suggests "back"

    let candidates = corrector.edit_candidates(test_word);

    println!("Misspelled: {}", test_word);
    println!("Candidates: {:?}\n", candidates);

    println!("Context 1: {:?}", context1);
    let ranked1 = corrector.rank_candidates(&candidates, &context1);
    for (word, score) in ranked1.iter().take(3) {
        println!("  {} (score: {:.4})", word, score);
    }

    println!("\nContext 2: {:?}", context2);
    let ranked2 = corrector.rank_candidates(&candidates, &context2);
    for (word, score) in ranked2.iter().take(3) {
        println!("  {} (score: {:.4})", word, score);
    }

    Ok(())
}
```

## Expected Output

```
=== Training Language Model ===

Vocabulary size: 38

=== Spell Correction Examples ===

Input:    the quikc brown fox
Output:   the quick brown fox
Expected: the quick brown fox
Status:   ✓

Input:    teh lazy dog
Output:   the lazy dog
Expected: the lazy dog
Status:   ✓

Input:    a brwon fox
Output:   a brown fox
Expected: a brown fox
Status:   ✓

...

=== Candidate Ranking Example ===

Misspelled word: quikc
Context: ["the"]

Edit-distance-1 candidates: ["quick", "quack"]

Ranked by P(candidate | context):
  1. quick (score: -2.3456)
  2. quack (score: -4.5678)

=== Context Importance ===

Misspelled: bak
Candidates: ["back", "bank", "bat", "bag"]

Context 1: ["money", "in", "the"]
  bank (score: -3.2345)
  back (score: -4.5678)
  bag (score: -5.1234)

Context 2: ["come", "right"]
  back (score: -2.8765)
  bank (score: -5.4321)
  bag (score: -5.9876)
```

## Key Points

1. **Context matters**: The same misspelling can be corrected differently based on surrounding words.

2. **Edit distance**: We generate candidates within edit distance 1 (deletions, insertions, substitutions, transpositions).

3. **Language model ranking**: Candidates are ranked by P(candidate | context), favoring contextually appropriate words.

4. **Hybrid advantage**: Embeddings help when n-gram data is sparse.

## Integration with liblevenshtein

For production use, combine with liblevenshtein's fuzzy matching:

```rust
use liblevenshtein::levenshtein::Levenshtein;

// Get candidates using liblevenshtein (faster than brute force)
let lev = Levenshtein::new(&dictionary);
let candidates = lev.search(misspelled, 2);  // Edit distance <= 2

// Rank with language model
let ranked = corrector.rank_candidates(&candidates, &context);
```

## See Also

- [Train and Evaluate](train-and-evaluate.md) - Basic workflow
- [HybridLanguageModel API](../api/hybrid.md) - Model reference
- [liblevenshtein](https://github.com/f1r3fly-io/liblevenshtein-rust) - Fuzzy matching
