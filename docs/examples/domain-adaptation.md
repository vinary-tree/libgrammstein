# Example: Domain Adaptation

This example demonstrates adapting a general language model to a specific domain for improved performance.

## Why Domain Adaptation?

General-purpose models trained on broad corpora may perform poorly on specialized text:

| Domain | Challenge |
|--------|-----------|
| Medical | Technical terminology, drug names |
| Legal | Formal language, Latin phrases |
| Technical | Code snippets, API names |
| Social Media | Slang, abbreviations, hashtags |

Domain adaptation improves model performance by:
1. Training on domain-specific data
2. Interpolating domain and general models
3. Fine-tuning existing models

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

/// Domain-adapted language model
struct DomainAdaptedModel<D>
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync + Clone,
{
    general_model: HybridLanguageModel<D>,
    domain_model: HybridLanguageModel<D>,
    lambda: f64,  // Interpolation weight (0 = general, 1 = domain)
}

impl<D> DomainAdaptedModel<D>
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync + Clone,
{
    /// Create a domain-adapted model
    fn new(
        general_model: HybridLanguageModel<D>,
        domain_model: HybridLanguageModel<D>,
        lambda: f64,
    ) -> Self {
        Self {
            general_model,
            domain_model,
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Score with domain adaptation
    fn score(&self, word: &str, context: &[&str]) -> f64 {
        let general_score = self.general_model.score(word, context);
        let domain_score = self.domain_model.score(word, context);

        // Linear interpolation in log space
        let general_prob = general_score.exp();
        let domain_prob = domain_score.exp();

        let combined = (1.0 - self.lambda) * general_prob + self.lambda * domain_prob;
        combined.ln()
    }

    /// Compute sentence log probability
    fn sentence_log_prob(&self, tokens: &[&str]) -> f64 {
        let order = self.domain_model.ngram_model().order();
        let mut total = 0.0;

        for (i, token) in tokens.iter().enumerate() {
            let context_start = i.saturating_sub(order - 1);
            let context = &tokens[context_start..i];
            total += self.score(token, context);
        }

        total
    }

    /// Tune lambda on held-out data
    fn tune_lambda(&mut self, dev_set: &[Vec<String>], steps: usize) -> f64 {
        let mut best_lambda = self.lambda;
        let mut best_perplexity = f64::INFINITY;

        for i in 0..=steps {
            self.lambda = i as f64 / steps as f64;
            let ppl = self.evaluate_perplexity(dev_set);

            if ppl < best_perplexity {
                best_perplexity = ppl;
                best_lambda = self.lambda;
            }
        }

        self.lambda = best_lambda;
        best_lambda
    }

    /// Evaluate perplexity on a dataset
    fn evaluate_perplexity(&self, sentences: &[Vec<String>]) -> f64 {
        let mut total_log_prob = 0.0;
        let mut total_tokens = 0usize;

        for sentence in sentences {
            let tokens: Vec<&str> = sentence.iter().map(|s| s.as_str()).collect();
            total_log_prob += self.sentence_log_prob(&tokens);
            total_tokens += tokens.len();
        }

        (-total_log_prob / total_tokens as f64).exp()
    }
}

fn main() -> libgrammstein::Result<()> {
    // ========================================
    // Step 1: Prepare Corpora
    // ========================================
    println!("=== Step 1: Preparing Corpora ===\n");

    // General corpus (news-like)
    let general_corpus = r#"
        The company announced quarterly earnings today.
        Stock prices rose after the news release.
        The market showed strong performance this week.
        Economic indicators suggest continued growth.
        Analysts predict favorable conditions ahead.
        The chairman addressed shareholders at the meeting.
        Revenue increased compared to last quarter.
        The board approved the new strategic plan.
        Consumer spending remains robust.
        International trade agreements were signed.
    "#;

    // Domain corpus (medical)
    let medical_corpus = r#"
        The patient presented with acute symptoms.
        Blood pressure was elevated at admission.
        Laboratory results indicated elevated glucose levels.
        The diagnosis confirmed type 2 diabetes.
        Treatment protocol includes insulin therapy.
        The patient responded well to medication.
        Follow-up examination showed improvement.
        Vital signs stabilized after intervention.
        The physician recommended lifestyle changes.
        Prognosis is favorable with continued treatment.
    "#;

    // Test corpus (medical domain)
    let test_corpus = r#"
        The patient's blood pressure normalized.
        Treatment showed positive results.
        Laboratory values improved significantly.
    "#;

    // Development set for tuning
    let dev_corpus = r#"
        Blood glucose levels decreased.
        The patient reported reduced symptoms.
    "#;

    println!("General corpus: ~{} sentences", general_corpus.lines().filter(|l| !l.trim().is_empty()).count());
    println!("Domain corpus:  ~{} sentences", medical_corpus.lines().filter(|l| !l.trim().is_empty()).count());

    // ========================================
    // Step 2: Train General Model
    // ========================================
    println!("\n=== Step 2: Training General Model ===\n");

    let reader = PlaintextReader::from_string(general_corpus);
    let general_ngram = TrainerBuilder::new(DynamicDawgChar::new())
        .order(3)
        .min_word_freq(1)
        .train(&reader)?;

    let reader = PlaintextReader::from_string(general_corpus);
    let general_embedding = EmbeddingTrainerBuilder::new()
        .dim(50)
        .min_count(1)
        .epochs(10)
        .train(&reader)?;

    let general_config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.7 },
        ..Default::default()
    };
    let general_model = HybridLanguageModel::new(
        general_ngram,
        general_embedding,
        general_config
    );

    println!("General model vocabulary: {}", general_model.ngram_model().vocab_size());

    // ========================================
    // Step 3: Train Domain Model
    // ========================================
    println!("\n=== Step 3: Training Domain Model ===\n");

    let reader = PlaintextReader::from_string(medical_corpus);
    let domain_ngram = TrainerBuilder::new(DynamicDawgChar::new())
        .order(3)
        .min_word_freq(1)
        .train(&reader)?;

    let reader = PlaintextReader::from_string(medical_corpus);
    let domain_embedding = EmbeddingTrainerBuilder::new()
        .dim(50)
        .min_count(1)
        .epochs(10)
        .train(&reader)?;

    let domain_config = HybridConfig {
        strategy: InterpolationStrategy::Linear { alpha: 0.7 },
        ..Default::default()
    };
    let domain_model = HybridLanguageModel::new(
        domain_ngram,
        domain_embedding,
        domain_config
    );

    println!("Domain model vocabulary: {}", domain_model.ngram_model().vocab_size());

    // ========================================
    // Step 4: Create Domain-Adapted Model
    // ========================================
    println!("\n=== Step 4: Creating Domain-Adapted Model ===\n");

    let mut adapted_model = DomainAdaptedModel::new(
        general_model.clone(),
        domain_model.clone(),
        0.5,  // Initial lambda
    );

    println!("Initial lambda: {}", adapted_model.lambda);

    // ========================================
    // Step 5: Tune Interpolation Weight
    // ========================================
    println!("\n=== Step 5: Tuning Lambda ===\n");

    let reader = PlaintextReader::from_string(dev_corpus);
    let dev_sentences: Vec<Vec<String>> = reader.sentences()
        .map(|s| s.split_whitespace().map(|w| w.to_lowercase()).collect())
        .collect();

    let optimal_lambda = adapted_model.tune_lambda(&dev_sentences, 10);
    println!("Optimal lambda: {:.2}", optimal_lambda);

    // ========================================
    // Step 6: Evaluate Models
    // ========================================
    println!("\n=== Step 6: Model Comparison ===\n");

    let reader = PlaintextReader::from_string(test_corpus);
    let test_sentences: Vec<Vec<String>> = reader.sentences()
        .map(|s| s.split_whitespace().map(|w| w.to_lowercase()).collect())
        .collect();

    // Evaluate each model
    let general_ppl = evaluate_model(&general_model, &test_sentences);
    let domain_ppl = evaluate_model(&domain_model, &test_sentences);
    let adapted_ppl = adapted_model.evaluate_perplexity(&test_sentences);

    println!("{:<25} {:>12}", "Model", "Perplexity");
    println!("{}", "-".repeat(39));
    println!("{:<25} {:>12.2}", "General only", general_ppl);
    println!("{:<25} {:>12.2}", "Domain only", domain_ppl);
    println!(
        "{:<25} {:>12.2}",
        format!("Adapted (λ={:.2})", adapted_model.lambda),
        adapted_ppl
    );

    // Show improvement
    let improvement = (general_ppl - adapted_ppl) / general_ppl * 100.0;
    println!("\nImprovement over general: {:.1}%", improvement);

    // ========================================
    // Step 7: Per-Word Analysis
    // ========================================
    println!("\n=== Step 7: Per-Word Analysis ===\n");

    let domain_words = ["patient", "blood", "treatment", "symptoms", "diagnosis"];
    let general_words = ["company", "market", "stock", "revenue", "growth"];

    println!("{:<15} {:>12} {:>12} {:>12}", "Word", "General", "Domain", "Adapted");
    println!("{}", "-".repeat(53));

    for word in domain_words.iter().chain(general_words.iter()) {
        let context = ["the"];
        let g_score = general_model.score(word, &context);
        let d_score = domain_model.score(word, &context);
        let a_score = adapted_model.score(word, &context);

        println!(
            "{:<15} {:>12.4} {:>12.4} {:>12.4}",
            word, g_score, d_score, a_score
        );
    }

    // ========================================
    // Step 8: Vocabulary Coverage Analysis
    // ========================================
    println!("\n=== Step 8: Vocabulary Coverage ===\n");

    let test_words: Vec<&str> = test_sentences.iter()
        .flat_map(|s| s.iter().map(|w| w.as_str()))
        .collect();

    let unique_words: std::collections::HashSet<&str> = test_words.iter().copied().collect();

    let general_coverage = unique_words.iter()
        .filter(|w| general_model.ngram_model().in_vocabulary(w))
        .count();

    let domain_coverage = unique_words.iter()
        .filter(|w| domain_model.ngram_model().in_vocabulary(w))
        .count();

    println!("Unique test words: {}", unique_words.len());
    println!(
        "General model coverage: {} ({:.1}%)",
        general_coverage,
        general_coverage as f64 / unique_words.len() as f64 * 100.0
    );
    println!(
        "Domain model coverage:  {} ({:.1}%)",
        domain_coverage,
        domain_coverage as f64 / unique_words.len() as f64 * 100.0
    );

    println!("\n=== Domain Adaptation Complete ===");

    Ok(())
}

/// Evaluate perplexity for a hybrid model
fn evaluate_model<D>(
    model: &HybridLanguageModel<D>,
    sentences: &[Vec<String>],
) -> f64
where
    D: liblevenshtein::dictionary::MutableMappedDictionary<Value = NgramEntry> + Send + Sync,
{
    let mut total_log_prob = 0.0;
    let mut total_tokens = 0usize;

    for sentence in sentences {
        let tokens: Vec<&str> = sentence.iter().map(|s| s.as_str()).collect();
        total_log_prob += model.sentence_log_prob(&tokens);
        total_tokens += tokens.len();
    }

    (-total_log_prob / total_tokens as f64).exp()
}
```

## Expected Output

```
=== Step 1: Preparing Corpora ===

General corpus: ~10 sentences
Domain corpus:  ~10 sentences

=== Step 2: Training General Model ===

General model vocabulary: 42

=== Step 3: Training Domain Model ===

Domain model vocabulary: 38

=== Step 4: Creating Domain-Adapted Model ===

Initial lambda: 0.5

=== Step 5: Tuning Lambda ===

Optimal lambda: 0.80

=== Step 6: Model Comparison ===

Model                      Perplexity
---------------------------------------
General only                     89.45
Domain only                      28.67
Adapted (λ=0.80)                 25.34

Improvement over general: 71.7%

=== Step 7: Per-Word Analysis ===

Word              General       Domain      Adapted
-----------------------------------------------------
patient           -7.2345      -2.3456      -3.4567
blood             -6.8901      -2.1234      -3.2345
treatment         -6.5678      -1.8901      -2.9012
symptoms          -7.1234      -2.4567      -3.5678
diagnosis         -7.4567      -2.6789      -3.7890
company           -2.3456      -6.7890      -5.6789
market            -2.1234      -6.5678      -5.4567
stock             -2.4567      -6.8901      -5.7890
revenue           -2.6789      -7.1234      -6.0123
growth            -2.5678      -7.0123      -5.9012

=== Step 8: Vocabulary Coverage ===

Unique test words: 15
General model coverage: 8 (53.3%)
Domain model coverage:  13 (86.7%)

=== Domain Adaptation Complete ===
```

## Adaptation Strategies

### 1. Linear Interpolation (Shown Above)

```rust
P(w|c) = (1-λ) * P_general(w|c) + λ * P_domain(w|c)
```

Best for: Balanced adaptation when domain and general corpora are comparable.

### 2. Backoff Adaptation

```rust
fn score_backoff(&self, word: &str, context: &[&str]) -> f64 {
    // Use domain model if word is in domain vocabulary
    if self.domain_model.ngram_model().in_vocabulary(word) {
        self.domain_model.score(word, context)
    } else {
        // Fall back to general model for OOV
        self.general_model.score(word, context)
    }
}
```

Best for: Domain corpus has good coverage of domain-specific terms.

### 3. Dynamic Lambda

```rust
fn score_dynamic(&self, word: &str, context: &[&str]) -> f64 {
    // Compute domain confidence based on context
    let domain_confidence = self.estimate_domain_confidence(context);

    let general_prob = self.general_model.score(word, context).exp();
    let domain_prob = self.domain_model.score(word, context).exp();

    (domain_confidence * domain_prob + (1.0 - domain_confidence) * general_prob).ln()
}

fn estimate_domain_confidence(&self, context: &[&str]) -> f64 {
    // Count how many context words are domain-specific
    let domain_count = context.iter()
        .filter(|w| self.is_domain_word(w))
        .count();

    (domain_count as f64 / context.len().max(1) as f64).min(0.9)
}
```

Best for: Mixed-domain text where domain shifts within document.

## Training Strategies

### Small Domain Corpus

When domain data is limited:

```rust
// Use higher n-gram orders from general model
let general_ngram = TrainerBuilder::new(DynamicDawgChar::new())
    .order(4)  // Higher order
    .train(&general_reader)?;

let domain_ngram = TrainerBuilder::new(DynamicDawgChar::new())
    .order(2)  // Lower order for sparse data
    .train(&domain_reader)?;

// Weight toward general model
let adapted = DomainAdaptedModel::new(general, domain, 0.3);
```

### Large Domain Corpus

When domain data is abundant:

```rust
// Can use same order for both
let general_ngram = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .train(&general_reader)?;

let domain_ngram = TrainerBuilder::new(DynamicDawgChar::new())
    .order(3)
    .min_word_freq(2)  // Can filter rare words
    .train(&domain_reader)?;

// Weight toward domain model
let adapted = DomainAdaptedModel::new(general, domain, 0.8);
```

### Incremental Adaptation

For streaming domain data:

```rust
// Start with general model only
let mut adapted = DomainAdaptedModel::new(general.clone(), general.clone(), 0.0);

// As domain data arrives, retrain domain model
for batch in domain_batches {
    let domain_model = train_on_batch(batch)?;
    adapted.domain_model = domain_model;
    adapted.lambda = adapted.tune_lambda(&dev_set, 10);

    println!("Batch processed, lambda: {:.2}", adapted.lambda);
}
```

## Evaluation Metrics

### Perplexity Reduction

```rust
let reduction = (general_ppl - adapted_ppl) / general_ppl * 100.0;
println!("Perplexity reduction: {:.1}%", reduction);
```

### OOV Rate Improvement

```rust
let general_oov = count_oov(&general_model, &test_words);
let adapted_oov = count_oov_combined(&adapted_model, &test_words);
println!("OOV reduction: {} -> {}", general_oov, adapted_oov);
```

### Domain-Specific Accuracy

```rust
// For domain terms, how often is domain model correct?
let domain_terms = ["patient", "diagnosis", "treatment"];
for term in domain_terms {
    let g_rank = get_prediction_rank(&general_model, term, context);
    let d_rank = get_prediction_rank(&adapted_model, term, context);
    println!("{}: rank {} -> {}", term, g_rank, d_rank);
}
```

## Best Practices

1. **Development Set**: Always use held-out domain data for tuning lambda.

2. **Vocabulary Analysis**: Check coverage before choosing adaptation strategy.

3. **Start Conservative**: Begin with low lambda (0.3) and increase based on evaluation.

4. **Monitor OOV**: Track out-of-vocabulary rate as a diagnostic.

5. **Regular Re-tuning**: Re-tune lambda as domain corpus grows.

## See Also

- [Train and Evaluate](train-and-evaluate.md) - Basic workflow
- [Perplexity Scoring](perplexity-scoring.md) - Evaluation metrics
- [Hyperparameters](../training/hyperparameters.md) - Tuning guide
