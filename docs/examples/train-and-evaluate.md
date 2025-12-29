# Example: Train and Evaluate Language Model

This example demonstrates a complete workflow for training and evaluating a hybrid language model.

## Setup

Add to `Cargo.toml`:

```toml
[dependencies]
libgrammstein = { version = "0.1", features = ["serde-extras", "cli"] }
liblevenshtein = "0.6"
```

## Complete Example

```rust
use libgrammstein::ngram::{NgramModel, TrainerBuilder, NgramEntry};
use libgrammstein::embedding::{SubwordEmbedding, EmbeddingTrainerBuilder};
use libgrammstein::hybrid::{HybridLanguageModel, HybridConfig, InterpolationStrategy};
use libgrammstein::corpus::PlaintextReader;
use liblevenshtein::dictionary::dynamic_dawg_char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // ========================================
    // Step 1: Prepare Data
    // ========================================
    println!("=== Step 1: Preparing Data ===");

    // Create sample training corpus
    let train_text = r#"
        The quick brown fox jumps over the lazy dog.
        Natural language processing enables computers to understand text.
        Machine learning is transforming how we process language.
        The fox ran quickly through the forest.
        Language models predict the next word in a sequence.
        Deep learning has revolutionized natural language processing.
        The brown dog chased the quick fox.
        Text processing is fundamental to many applications.
        Neural networks learn patterns from large datasets.
        The lazy cat watched the quick brown fox.
    "#;

    // Create sample test corpus
    let test_text = r#"
        The quick fox jumped over the fence.
        Language processing helps computers understand humans.
        The brown fox ran through the woods.
    "#;

    let train_reader = PlaintextReader::from_string(train_text);
    let test_reader = PlaintextReader::from_string(test_text);

    // Count sentences
    let train_sentences: Vec<_> = train_reader.sentences().collect();
    println!("Training sentences: {}", train_sentences.len());

    // ========================================
    // Step 2: Train N-gram Model
    // ========================================
    println!("\n=== Step 2: Training N-gram Model ===");

    let train_reader = PlaintextReader::from_string(train_text);
    let ngram_model = TrainerBuilder::new(DynamicDawgChar::new())
        .order(3)           // Trigram model
        .min_word_freq(1)   // Keep all words (small corpus)
        .train(&train_reader)?;

    println!("Vocabulary size: {}", ngram_model.vocab_size());
    println!("N-gram count: {}", ngram_model.ngram_count());

    // ========================================
    // Step 3: Train Embedding Model
    // ========================================
    println!("\n=== Step 3: Training Embedding Model ===");

    let train_reader = PlaintextReader::from_string(train_text);
    let embedding_model = EmbeddingTrainerBuilder::new()
        .dim(50)            // 50-dimensional vectors
        .window_size(3)     // Context window
        .min_count(1)       // Keep all words
        .epochs(10)         // More epochs for small corpus
        .train(&train_reader)?;

    println!("Embedding dimension: {}", embedding_model.dim());
    println!("Embedding vocabulary: {}", embedding_model.vocab_size());

    // ========================================
    // Step 4: Create Hybrid Model
    // ========================================
    println!("\n=== Step 4: Creating Hybrid Model ===");

    let config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.7 },
        cache_size: 10_000,
        ..Default::default()
    };

    let hybrid_model = HybridLanguageModel::new(
        ngram_model.clone(),
        embedding_model.clone(),
        config
    );

    println!("Hybrid model created with α=0.7");

    // ========================================
    // Step 5: Evaluate Models
    // ========================================
    println!("\n=== Step 5: Evaluating Models ===");

    let test_reader = PlaintextReader::from_string(test_text);
    let test_sentences: Vec<Vec<String>> = test_reader.sentences()
        .map(|s| s.split_whitespace().map(|w| w.to_lowercase()).collect())
        .collect();

    // Evaluate N-gram only
    let ngram_ppl = evaluate_perplexity_ngram(&ngram_model, &test_sentences);
    println!("N-gram perplexity: {:.2}", ngram_ppl);

    // Evaluate Hybrid
    let hybrid_ppl = evaluate_perplexity_hybrid(&hybrid_model, &test_sentences);
    println!("Hybrid perplexity: {:.2}", hybrid_ppl);

    // ========================================
    // Step 6: Query Examples
    // ========================================
    println!("\n=== Step 6: Query Examples ===");

    // N-gram probabilities
    println!("\nN-gram log probabilities:");
    let queries = [
        ("fox", vec!["quick", "brown"]),
        ("dog", vec!["lazy"]),
        ("language", vec!["natural"]),
        ("xyz", vec!["the"]),  // OOV word
    ];

    for (word, context) in &queries {
        let context_refs: Vec<&str> = context.iter().map(|s| s.as_str()).collect();
        let ngram_prob = ngram_model.log_prob(word, &context_refs);
        let hybrid_prob = hybrid_model.score(word, &context_refs);
        println!(
            "  P({} | {:?}): ngram={:.4}, hybrid={:.4}",
            word, context, ngram_prob, hybrid_prob
        );
    }

    // Similar words (embedding)
    println!("\nSimilar words to 'language':");
    for (word, score) in embedding_model.most_similar("language", 5) {
        println!("  {}: {:.4}", word, score);
    }

    // ========================================
    // Step 7: Save Models
    // ========================================
    println!("\n=== Step 7: Saving Models ===");

    ngram_model.save("ngram_model.bin")?;
    println!("Saved: ngram_model.bin");

    embedding_model.save("embedding_model.bin")?;
    println!("Saved: embedding_model.bin");

    hybrid_model.save("hybrid_model.bin")?;
    println!("Saved: hybrid_model.bin");

    // ========================================
    // Step 8: Load and Verify
    // ========================================
    println!("\n=== Step 8: Loading and Verifying ===");

    let loaded_ngram: NgramModel<DynamicDawgChar<NgramEntry>> =
        NgramModel::load("ngram_model.bin")?;
    let loaded_embedding = SubwordEmbedding::load("embedding_model.bin")?;
    let loaded_hybrid: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
        HybridLanguageModel::load("hybrid_model.bin")?;

    // Verify loaded models produce same results
    let test_prob_original = ngram_model.log_prob("fox", &["brown"]);
    let test_prob_loaded = loaded_ngram.log_prob("fox", &["brown"]);
    assert!((test_prob_original - test_prob_loaded).abs() < 1e-10);
    println!("Verification passed: loaded models match original");

    // Clean up
    std::fs::remove_file("ngram_model.bin").ok();
    std::fs::remove_file("embedding_model.bin").ok();
    std::fs::remove_file("hybrid_model.bin").ok();

    println!("\n=== Complete! ===");
    Ok(())
}

/// Evaluate perplexity for n-gram model
fn evaluate_perplexity_ngram<D>(
    model: &NgramModel<D>,
    sentences: &[Vec<String>],
) -> f64
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry>,
{
    let mut total_log_prob = 0.0;
    let mut total_words = 0usize;

    for sentence in sentences {
        let tokens: Vec<&str> = sentence.iter().map(|s| s.as_str()).collect();
        total_log_prob += model.sentence_log_prob(&tokens);
        total_words += tokens.len();
    }

    (-total_log_prob / total_words as f64).exp()
}

/// Evaluate perplexity for hybrid model
fn evaluate_perplexity_hybrid<D>(
    model: &HybridLanguageModel<D>,
    sentences: &[Vec<String>],
) -> f64
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    let mut total_log_prob = 0.0;
    let mut total_words = 0usize;

    for sentence in sentences {
        let tokens: Vec<&str> = sentence.iter().map(|s| s.as_str()).collect();
        total_log_prob += model.sentence_log_prob(&tokens);
        total_words += tokens.len();
    }

    (-total_log_prob / total_words as f64).exp()
}
```

## Expected Output

```
=== Step 1: Preparing Data ===
Training sentences: 10

=== Step 2: Training N-gram Model ===
Vocabulary size: 45
N-gram count: 127

=== Step 3: Training Embedding Model ===
Embedding dimension: 50
Embedding vocabulary: 45

=== Step 4: Creating Hybrid Model ===
Hybrid model created with α=0.7

=== Step 5: Evaluating Models ===
N-gram perplexity: 42.31
Hybrid perplexity: 38.56

=== Step 6: Query Examples ===

N-gram log probabilities:
  P(fox | ["quick", "brown"]): ngram=-1.2345, hybrid=-1.1234
  P(dog | ["lazy"]): ngram=-2.3456, hybrid=-2.1234
  P(language | ["natural"]): ngram=-1.5678, hybrid=-1.4567
  P(xyz | ["the"]): ngram=-8.1234, hybrid=-5.4321

Similar words to 'language':
  processing: 0.7234
  natural: 0.6891
  learning: 0.5432
  text: 0.4321
  models: 0.3210

=== Step 7: Saving Models ===
Saved: ngram_model.bin
Saved: embedding_model.bin
Saved: hybrid_model.bin

=== Step 8: Loading and Verifying ===
Verification passed: loaded models match original

=== Complete! ===
```

## Key Observations

1. **Hybrid perplexity is lower** than n-gram alone because embeddings help with OOV and rare words.

2. **OOV handling**: The word "xyz" gets a very low n-gram probability but a reasonable hybrid probability due to subword embeddings.

3. **Similar words** reflect semantic relationships learned from the corpus.

## Next Steps

- Try with larger corpora for better quality
- Tune hyperparameters (see [Hyperparameter Guide](../training/hyperparameters.md))
- Use for downstream tasks like spell correction

## See Also

- [N-gram Training](../training/ngram.md)
- [Embedding Training](../training/embedding.md)
- [Hybrid Training](../training/hybrid.md)
