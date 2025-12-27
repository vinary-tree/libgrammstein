# Pipeline Usage

This document provides complete examples of using libgrammstein in lling-llang correction pipelines.

## Correction Pipeline Concept

A correction pipeline is a sequence of layers that transform a lattice:

```
Input Text: "teh quikc brwon fox"
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Tokenization                                                        │
│   Convert text to initial lattice with one path                             │
│   ["teh", "quikc", "brwon", "fox"]                                          │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Spelling Correction (liblevenshtein)                               │
│   Add fuzzy matching alternatives at each position                          │
│                                                                             │
│   Position 0: teh → [the, tea, ten, ...]                                   │
│   Position 1: quikc → [quick, quack, ...]                                  │
│   Position 2: brwon → [brown, brawn, ...]                                  │
│   Position 3: fox → [fox] (correct)                                        │
│                                                                             │
│   Lattice now has multiple paths                                            │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Grammar Filter (optional)                                          │
│   Remove paths that violate grammar rules                                   │
│   e.g., "the quack brown fox" might be filtered                            │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Language Model (libgrammstein)                                     │
│   Rescore paths based on fluency                                            │
│                                                                             │
│   P("the quick brown fox") = -12.3 (high)                                  │
│   P("tea quick brown fox") = -18.5 (lower)                                 │
│   P("the quick brawn fox") = -16.2 (medium)                                │
│                                                                             │
│   Best path: "the quick brown fox"                                          │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
Output: "the quick brown fox"
```

## Complete Example: Spelling Correction

```rust
use lling_llang::prelude::*;
use lling_llang::layers::{
    LayerPipelineBuilder,
    LanguageModelLayer,
    SpellingCorrectionLayer,
};
use lling_llang::backend::HashMapBackend;
use lling_llang::semiring::TropicalWeight;
use liblevenshtein::dictionary::DoubleArrayTrieChar;
use libgrammstein::HybridLanguageModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Step 1: Load dictionary for spelling correction
    let words = load_dictionary("words.txt")?;
    let dictionary = DoubleArrayTrieChar::from_terms(&words);

    // Step 2: Load language model
    let lm: HybridLanguageModel<_> = HybridLanguageModel::load("model.bin")?;

    // Step 3: Build correction pipeline
    let pipeline = LayerPipelineBuilder::<TropicalWeight, HashMapBackend>::new()
        .add_layer(SpellingCorrectionLayer::new(dictionary, 2))  // max distance 2
        .add_layer(LanguageModelLayer::new(Box::new(lm), 1.0))
        .build();

    // Step 4: Process input
    let input = "teh quikc brwon fox";
    let tokens: Vec<&str> = input.split_whitespace().collect();

    // Create initial lattice
    let mut builder = LatticeBuilder::new(HashMapBackend::new());
    for (i, token) in tokens.iter().enumerate() {
        builder.add_token(i, i + 1, token, TropicalWeight::one());
    }
    let lattice = builder.build(tokens.len());

    // Apply pipeline
    let result = pipeline.apply(&lattice)?;

    // Extract best path
    let best_path = viterbi(&mut result);
    println!("Corrected: {}", best_path.labels.join(" "));
    // Output: "the quick brown fox"

    Ok(())
}
```

## Pipeline with Grammar Filtering

```rust
use lling_llang::layers::CfgFilterLayer;
use lling_llang::grammar::{Grammar, GrammarBuilder};

fn build_grammar_pipeline() -> Result<LayerPipeline<TropicalWeight, HashMapBackend>, Error> {
    // Load resources
    let dictionary = load_dictionary()?;
    let lm = HybridLanguageModel::load("model.bin")?;

    // Define simple grammar
    let grammar = GrammarBuilder::new()
        .add_rule("S", &["NP", "VP"])
        .add_rule("NP", &["DET", "ADJ", "N"])
        .add_rule("NP", &["DET", "N"])
        .add_rule("VP", &["V", "NP"])
        .add_rule("VP", &["V"])
        .add_terminals("DET", &["the", "a", "an"])
        .add_terminals("ADJ", &["quick", "brown", "lazy"])
        .add_terminals("N", &["fox", "dog", "cat"])
        .add_terminals("V", &["jumps", "runs", "walks"])
        .build()?;

    // Build pipeline
    let pipeline = LayerPipelineBuilder::new()
        .add_layer(SpellingCorrectionLayer::new(dictionary, 2))
        .add_layer(CfgFilterLayer::new(&grammar))  // Filter ungrammatical paths
        .add_layer(LanguageModelLayer::new(Box::new(lm), 1.0))
        .build();

    Ok(pipeline)
}
```

## N-best Output

Extract multiple correction candidates:

```rust
use lling_llang::algorithms::nbest;

fn get_correction_candidates(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    input: &str,
    n: usize,
) -> Vec<(String, f64)> {
    let tokens: Vec<&str> = input.split_whitespace().collect();
    let lattice = tokens_to_lattice(&tokens);

    let result = pipeline.apply(&lattice).expect("Pipeline failed");

    // Extract N-best paths
    nbest(&mut result, n)
        .map(|path| {
            let text = path.labels.join(" ");
            let score = path.weight.value();
            (text, score)
        })
        .collect()
}

fn main() {
    let pipeline = build_pipeline()?;

    let candidates = get_correction_candidates(&pipeline, "teh quikc brwon fox", 5);

    for (i, (text, score)) in candidates.iter().enumerate() {
        println!("{}: {} (score: {:.3})", i + 1, text, score);
    }
    // Output:
    // 1: the quick brown fox (score: 12.345)
    // 2: tea quick brown fox (score: 15.678)
    // 3: the quick brawn fox (score: 16.234)
    // ...
}
```

## Confidence Scoring

Compute confidence based on score differences:

```rust
fn correction_with_confidence(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    input: &str,
) -> (String, f64) {
    let candidates = get_correction_candidates(pipeline, input, 2);

    if candidates.len() < 2 {
        return (candidates[0].0.clone(), 1.0);
    }

    let (best_text, best_score) = &candidates[0];
    let (_, second_score) = &candidates[1];

    // Confidence based on score gap
    // Larger gap = more confident
    let score_gap = second_score - best_score;  // Tropical: lower is better
    let confidence = 1.0 - (-score_gap).exp();  // Convert to [0, 1]

    (best_text.clone(), confidence.max(0.0).min(1.0))
}
```

## Streaming Correction

Process text streams efficiently:

```rust
use std::io::{BufRead, BufReader, Write};

fn stream_correction<R: BufRead, W: Write>(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    input: R,
    mut output: W,
) -> Result<(), Error> {
    for line in input.lines() {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens.is_empty() {
            writeln!(output)?;
            continue;
        }

        let lattice = tokens_to_lattice(&tokens);
        let result = pipeline.apply(&lattice)?;
        let best = viterbi(&mut result);

        writeln!(output, "{}", best.labels.join(" "))?;
    }

    Ok(())
}

// Usage
fn main() {
    let pipeline = build_pipeline()?;
    let stdin = BufReader::new(std::io::stdin());
    let stdout = std::io::stdout();

    stream_correction(&pipeline, stdin, stdout)?;
}
```

## Batch Processing

Process multiple inputs in parallel:

```rust
use rayon::prelude::*;

fn batch_correction(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    inputs: &[&str],
) -> Vec<String> {
    inputs
        .par_iter()
        .map(|input| {
            let tokens: Vec<&str> = input.split_whitespace().collect();
            let lattice = tokens_to_lattice(&tokens);
            let result = pipeline.apply(&lattice).expect("Pipeline failed");
            let best = viterbi(&mut result);
            best.labels.join(" ")
        })
        .collect()
}
```

## Custom Weight Tuning

Adjust layer weights for different use cases:

```rust
fn build_tuned_pipeline(
    spelling_weight: f64,  // Weight for spelling distance
    lm_weight: f64,        // Weight for language model
) -> LayerPipeline<TropicalWeight, HashMapBackend> {
    let dictionary = load_dictionary()?;
    let lm = HybridLanguageModel::load("model.bin")?;

    LayerPipelineBuilder::new()
        .add_layer(SpellingCorrectionLayer::with_weight(dictionary, 2, spelling_weight))
        .add_layer(LanguageModelLayer::new(Box::new(lm), lm_weight))
        .build()
}

// High LM weight: Prefer fluent corrections even if edit distance is higher
let fluent_pipeline = build_tuned_pipeline(1.0, 3.0);

// High spelling weight: Prefer minimal edits
let minimal_edit_pipeline = build_tuned_pipeline(2.0, 1.0);
```

## Domain-Specific Models

Use different models for different domains:

```rust
struct DomainPipelines {
    general: LayerPipeline<TropicalWeight, HashMapBackend>,
    medical: LayerPipeline<TropicalWeight, HashMapBackend>,
    legal: LayerPipeline<TropicalWeight, HashMapBackend>,
}

impl DomainPipelines {
    fn new() -> Result<Self, Error> {
        let general_lm = HybridLanguageModel::load("general.bin")?;
        let medical_lm = HybridLanguageModel::load("medical.bin")?;
        let legal_lm = HybridLanguageModel::load("legal.bin")?;

        let general_dict = load_dictionary("general_words.txt")?;
        let medical_dict = load_dictionary("medical_terms.txt")?;
        let legal_dict = load_dictionary("legal_terms.txt")?;

        Ok(Self {
            general: build_pipeline(general_dict, general_lm),
            medical: build_pipeline(medical_dict, medical_lm),
            legal: build_pipeline(legal_dict, legal_lm),
        })
    }

    fn correct(&self, text: &str, domain: Domain) -> String {
        let pipeline = match domain {
            Domain::General => &self.general,
            Domain::Medical => &self.medical,
            Domain::Legal => &self.legal,
        };

        apply_pipeline(pipeline, text)
    }
}
```

## Hybrid N-gram + Embedding Scoring

Leverage both models:

```rust
use libgrammstein::{HybridLanguageModel, HybridConfig, OovStrategy};

fn build_hybrid_pipeline() -> LayerPipeline<TropicalWeight, HashMapBackend> {
    // Load N-gram model
    let ngram = NgramModel::load("ngram.bin")?;

    // Load embedding model
    let embedding = SubwordEmbedding::load("embedding.bin")?;

    // Configure hybrid
    let config = HybridConfig {
        ngram_weight: 0.8,       // 80% N-gram
        embedding_weight: 0.2,   // 20% embedding
        cache_size: 50_000,
        oov_strategy: OovStrategy::BackoffWithEmbedding,
    };

    let hybrid = HybridLanguageModel::new(ngram, embedding, config);

    LayerPipelineBuilder::new()
        .add_layer(SpellingCorrectionLayer::new(dictionary, 2))
        .add_layer(LanguageModelLayer::new(Box::new(hybrid), 1.0))
        .build()
}
```

## Error Handling

Handle pipeline errors gracefully:

```rust
use lling_llang::layers::LayerError;

fn safe_correction(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    input: &str,
) -> Result<String, CorrectionError> {
    let tokens: Vec<&str> = input.split_whitespace().collect();

    if tokens.is_empty() {
        return Ok(String::new());
    }

    let lattice = tokens_to_lattice(&tokens);

    match pipeline.apply(&lattice) {
        Ok(result) => {
            let best = viterbi(&mut result);
            Ok(best.labels.join(" "))
        }
        Err(LayerError::NoValidPaths) => {
            // All paths filtered out, return original
            Ok(input.to_string())
        }
        Err(LayerError::Layer(msg)) => {
            Err(CorrectionError::LayerFailed(msg))
        }
        Err(e) => {
            Err(CorrectionError::Unknown(e.to_string()))
        }
    }
}
```

## Performance Monitoring

Track pipeline performance:

```rust
use std::time::Instant;

struct PipelineMetrics {
    total_inputs: u64,
    total_tokens: u64,
    total_time_ms: u64,
    cache_hits: u64,
    cache_misses: u64,
}

fn correction_with_metrics(
    pipeline: &LayerPipeline<TropicalWeight, HashMapBackend>,
    input: &str,
    metrics: &mut PipelineMetrics,
) -> String {
    let start = Instant::now();

    let tokens: Vec<&str> = input.split_whitespace().collect();
    let result = apply_pipeline(pipeline, &tokens);

    metrics.total_inputs += 1;
    metrics.total_tokens += tokens.len() as u64;
    metrics.total_time_ms += start.elapsed().as_millis() as u64;

    result
}

impl PipelineMetrics {
    fn report(&self) {
        println!("Pipeline Metrics:");
        println!("  Total inputs: {}", self.total_inputs);
        println!("  Total tokens: {}", self.total_tokens);
        println!("  Avg time per input: {:.2}ms",
            self.total_time_ms as f64 / self.total_inputs as f64);
        println!("  Tokens per second: {:.0}",
            self.total_tokens as f64 / (self.total_time_ms as f64 / 1000.0));
    }
}
```

## Integration Test

Complete integration test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pipeline() {
        // Setup
        let dictionary = DoubleArrayTrieChar::from_terms(&[
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        ]);

        let lm = create_test_language_model();

        let pipeline = LayerPipelineBuilder::new()
            .add_layer(SpellingCorrectionLayer::new(dictionary, 2))
            .add_layer(LanguageModelLayer::new(Box::new(lm), 1.0))
            .build();

        // Test cases
        let test_cases = vec![
            ("the quick brown fox", "the quick brown fox"),  // No change
            ("teh quick brown fox", "the quick brown fox"),  // One error
            ("teh quikc brwon fox", "the quick brown fox"),  // Multiple errors
        ];

        for (input, expected) in test_cases {
            let tokens: Vec<&str> = input.split_whitespace().collect();
            let lattice = tokens_to_lattice(&tokens);
            let result = pipeline.apply(&lattice).unwrap();
            let best = viterbi(&mut result);
            let output = best.labels.join(" ");

            assert_eq!(output, expected, "Failed for input: {}", input);
        }
    }
}
```

## Next Steps

- [Overview](overview.md): Integration architecture
- [PathMap Synergy](pathmap-synergy.md): Shared infrastructure
- [liblevenshtein Integration](../liblevenshtein/overview.md): Dictionary backends
- [Hybrid Model](../../components/hybrid/overview.md): Model details
