# Hyperparameter Tuning Guide

This guide covers how to tune hyperparameters for optimal model performance.

## N-gram Model Parameters

### N-gram Order

The order determines maximum context length.

| Order | Context | Memory | Quality |
|-------|---------|--------|---------|
| 2 | 1 word | Low | Low |
| 3 | 2 words | Medium | Medium |
| 5 | 4 words | High | High |
| 7 | 6 words | Very High | Diminishing returns |

**Tuning approach:**
```rust
fn find_best_order(corpus: &impl CorpusReader, dev: &impl CorpusReader) -> usize {
    let mut best_order = 3;
    let mut best_ppl = f64::INFINITY;

    for order in 2..=7 {
        let model = TrainerBuilder::new(DynamicDawgChar::new())
            .order(order)
            .train(corpus)?;

        let ppl = evaluate_perplexity(&model, dev);
        println!("Order {}: perplexity = {:.2}", order, ppl);

        if ppl < best_ppl {
            best_ppl = ppl;
            best_order = order;
        }
    }

    best_order
}
```

**Guidelines:**
- Start with order 5
- Higher orders need more training data
- Diminishing returns beyond 5-7

### Minimum Word Frequency

Filter rare words to reduce model size.

| Value | Vocabulary | Memory | Coverage |
|-------|------------|--------|----------|
| 1 | Large | High | 100% |
| 5 | Medium | Medium | ~95% |
| 10 | Small | Low | ~90% |

**Trade-off:** Lower values keep more words but increase memory and may add noise.

## Embedding Parameters

### Dimension

Controls vector size and expressiveness.

| Dimension | Quality | Memory | Training Time |
|-----------|---------|--------|---------------|
| 50 | Lower | Small | Fast |
| 100 | Good | Medium | Medium |
| 200 | Better | Large | Slow |
| 300 | Best | Very Large | Very Slow |

**Guidelines:**
- 100 for most use cases
- 300 for large corpora (100M+ words)
- 50 for memory-constrained applications

### Window Size

Context window for skip-gram training.

| Size | Relationship Type | Training Time |
|------|-------------------|---------------|
| 2 | Syntactic (same POS) | Fast |
| 5 | Semantic (related meaning) | Medium |
| 10 | Topical (same domain) | Slow |

**Guidelines:**
- 5 is a good default
- Smaller for syntactic tasks (POS tagging)
- Larger for semantic tasks (similarity)

### Number of Epochs

Training passes over the corpus.

| Epochs | Corpus Size | Quality |
|--------|-------------|---------|
| 15-20 | < 1M words | Needed |
| 5-10 | 1-10M words | Sufficient |
| 1-3 | > 100M words | Enough |

**Guidelines:**
- More epochs for smaller corpora
- Watch for overfitting on small data
- Diminishing returns after 5-10 epochs

### Negative Samples

Negative samples per positive sample.

| Count | Quality | Training Time |
|-------|---------|---------------|
| 2 | Lower | Fast |
| 5 | Good | Medium |
| 10 | Better | Slow |
| 20 | Best | Very Slow |

**Guidelines:**
- 5 is standard
- Increase for small vocabularies
- Decrease for faster training

### Learning Rate

Initial learning rate (decays linearly).

| Rate | Convergence | Stability |
|------|-------------|-----------|
| 0.01 | Slow | Very Stable |
| 0.025 | Medium | Stable |
| 0.05 | Fast | Good |
| 0.1 | Very Fast | May diverge |

**Guidelines:**
- 0.05 is default
- Decrease if training is unstable
- Increase if convergence is too slow

## Hybrid Model Parameters

### Interpolation Weight (Alpha)

Balance between n-gram and embedding.

| Alpha | N-gram Weight | When to Use |
|-------|---------------|-------------|
| 0.9 | 90% | High-quality n-gram, rare OOV |
| 0.7 | 70% | Balanced (default) |
| 0.5 | 50% | Equal weighting |
| 0.3 | 30% | Small n-gram corpus |

**Tuning approach:**
```rust
fn tune_alpha(hybrid_components: &(NgramModel<D>, SubwordEmbedding), dev: &impl CorpusReader) -> f64 {
    let (ngram, embedding) = hybrid_components;
    let mut best_alpha = 0.5;
    let mut best_ppl = f64::INFINITY;

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let config = HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha },
            ..Default::default()
        };
        let hybrid = HybridLanguageModel::new(ngram.clone(), embedding.clone(), config);
        let ppl = evaluate_perplexity(&hybrid, dev);

        if ppl < best_ppl {
            best_ppl = ppl;
            best_alpha = alpha;
        }
    }

    best_alpha
}
```

### Temperature

Controls embedding probability sharpness.

| Temperature | Effect |
|-------------|--------|
| 0.5 | Sharp, confident |
| 1.0 | Neutral (default) |
| 2.0 | Smooth, uncertain |

**Guidelines:**
- Start with 1.0
- Lower for more decisive predictions
- Higher for more diversity

## Systematic Tuning Process

### 1. Grid Search

Exhaustively search parameter combinations:

```rust
fn grid_search(
    corpus: &impl CorpusReader,
    dev: &impl CorpusReader,
) -> (usize, usize, f64) {  // (order, dim, alpha)
    let mut best_params = (5, 100, 0.7);
    let mut best_ppl = f64::INFINITY;

    for order in [3, 5, 7] {
        for dim in [50, 100, 200] {
            for alpha in [0.3, 0.5, 0.7, 0.9] {
                // Train models
                let ngram = train_ngram(corpus, order)?;
                let embedding = train_embedding(corpus, dim)?;

                let config = HybridConfig {
                    strategy: InterpolationStrategy::Linear { alpha },
                    ..Default::default()
                };
                let hybrid = HybridLanguageModel::new(ngram, embedding, config);

                let ppl = evaluate_perplexity(&hybrid, dev);
                println!("order={}, dim={}, α={:.1}: ppl={:.2}", order, dim, alpha, ppl);

                if ppl < best_ppl {
                    best_ppl = ppl;
                    best_params = (order, dim, alpha);
                }
            }
        }
    }

    best_params
}
```

### 2. Bayesian Optimization

For large search spaces, use optimization libraries:

```rust
// Pseudo-code for Bayesian optimization
fn bayesian_optimize() {
    let optimizer = BayesianOptimizer::new()
        .add_param("order", 2..=7)
        .add_param("dim", 50..=300)
        .add_param("alpha", 0.1..=0.9)
        .add_param("window", 2..=10);

    for _ in 0..50 {  // 50 iterations
        let params = optimizer.suggest();
        let score = evaluate_with_params(&params);
        optimizer.observe(params, score);
    }

    optimizer.best_params()
}
```

### 3. Cross-Validation

For robust evaluation:

```rust
fn cross_validate(corpus: &[String], k: usize, params: &Params) -> f64 {
    let fold_size = corpus.len() / k;
    let mut scores = Vec::new();

    for i in 0..k {
        let dev_start = i * fold_size;
        let dev_end = dev_start + fold_size;

        let train: Vec<_> = corpus[..dev_start].iter()
            .chain(corpus[dev_end..].iter())
            .cloned()
            .collect();
        let dev = &corpus[dev_start..dev_end];

        let score = train_and_evaluate(&train, dev, params);
        scores.push(score);
    }

    scores.iter().sum::<f64>() / k as f64
}
```

## Recommended Defaults

### Small Corpus (< 1M words)

```rust
// N-gram
.order(3)
.min_word_freq(2)

// Embedding
.dim(50)
.window_size(5)
.min_count(2)
.epochs(15)

// Hybrid
.alpha(0.5)  // Equal weight
```

### Medium Corpus (1-10M words)

```rust
// N-gram
.order(5)
.min_word_freq(5)

// Embedding
.dim(100)
.window_size(5)
.min_count(5)
.epochs(5)

// Hybrid
.alpha(0.7)  // Favor n-gram
```

### Large Corpus (> 100M words)

```rust
// N-gram
.order(5)
.min_word_freq(10)

// Embedding
.dim(300)
.window_size(5)
.min_count(10)
.epochs(3)

// Hybrid
.alpha(0.8)  // Strong n-gram
```

## Common Pitfalls

### Overfitting

**Symptoms:** Low training perplexity, high dev perplexity

**Solutions:**
- Increase min_word_freq
- Decrease order (n-gram)
- Decrease epochs (embedding)
- Use more training data

### Underfitting

**Symptoms:** High perplexity on both train and dev

**Solutions:**
- Increase order (n-gram)
- Increase dim (embedding)
- Increase epochs
- Decrease min_count

### Memory Issues

**Solutions:**
- Decrease order
- Increase min_word_freq
- Decrease dim
- Use streaming corpus reader

## Evaluation Metrics

### Perplexity

Lower is better. Measures how well the model predicts held-out data.

```rust
let ppl = (-log_prob / n_words).exp();
```

### Accuracy (for classification)

```rust
fn classification_accuracy(model: &HybridLanguageModel<D>, test_cases: &[(Vec<&str>, &str)]) -> f64 {
    let correct = test_cases.iter()
        .filter(|(context, expected)| {
            let predicted = model.predict_next(context, &vocabulary);
            predicted.0 == *expected
        })
        .count();

    correct as f64 / test_cases.len() as f64
}
```

### Word Similarity Correlation

For embeddings, correlate with human judgments:

```rust
fn similarity_correlation(model: &SubwordEmbedding, benchmark: &[(String, String, f32)]) -> f64 {
    spearman_correlation(
        &benchmark.iter().map(|(w1, w2, _)| model.similarity(w1, w2)).collect(),
        &benchmark.iter().map(|(_, _, score)| *score).collect()
    )
}
```

## See Also

- [N-gram Training](ngram.md) - N-gram training details
- [Embedding Training](embedding.md) - Embedding training details
- [Large Corpora](large-corpora.md) - Memory optimization
