# Mode-Aware N-gram Models

The LaTeX n-gram module provides mode-aware language model scoring, with separate models trained for different LaTeX contexts.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 LaTeX N-gram Scorer                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: Token sequence with modes                       │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────┐                                    │
│  │  Mode Router    │                                    │
│  └────────┬────────┘                                    │
│           │                                             │
│  ┌────────┴────────┬─────────────┬─────────────┐       │
│  │                 │             │             │       │
│  ▼                 ▼             ▼             ▼       │
│  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐        │
│  │ Text  │   │ Math  │   │Command│   │ Mixed │        │
│  │ Model │   │ Model │   │ Model │   │ Model │        │
│  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘        │
│      │           │           │           │            │
│      └───────────┴───────────┴───────────┘            │
│                       │                                │
│                       ▼                                │
│              ┌─────────────────┐                       │
│              │ Score Combiner  │                       │
│              └────────┬────────┘                       │
│                       │                                │
│                       ▼                                │
│                 Final Score                            │
│                                                        │
└─────────────────────────────────────────────────────────┘
```

## Model Types

### Text Model

Trained on `\text{}` blocks and text-mode content:

```rust
// Scores sequences like:
// "the" "quick" "brown" "fox"
// "according" "to" "theorem" "3.1"
```

### Math Model

Trained on math-mode content:

```rust
// Scores sequences like:
// "alpha" "+" "beta" "=" "gamma"
// "sum" "_{" "i" "=" "1" "}^{" "n" "}"
```

### Command Model

Trained on command sequences:

```rust
// Scores sequences like:
// "begin" "{equation}" "label" "{eq:1}"
// "frac" "{" "1" "}" "{" "2" "}"
```

## Basic Usage

```rust
use libgrammstein::latex::{LaTeXNgramModel, NgramConfig};

let config = NgramConfig {
    text_model: "models/latex_text.5gram",
    math_model: "models/latex_math.5gram",
    command_model: "models/latex_cmd.5gram",
    order: 5,
    smoothing: Smoothing::KneserNey { discount: 0.75 },
};

let model = LaTeXNgramModel::load(&config)?;
```

## Scoring Tokens

```rust
use libgrammstein::latex::{LaTeXTokenizer, TokenMode};

let tokenizer = LaTeXTokenizer::new();
let model = LaTeXNgramModel::load(&config)?;

let input = r"\begin{equation} \alpha + \beta \end{equation}";
let tokens = tokenizer.tokenize(input);

let score = model.score(&tokens)?;
println!("Log probability: {:.4}", score);
```

## Mode-Weighted Scoring

Scores are computed per-mode and combined:

```rust
impl LaTeXNgramModel {
    pub fn score(&self, tokens: &[LaTeXToken]) -> f64 {
        let mut total_score = 0.0;
        let mut mode_counts: HashMap<TokenMode, usize> = HashMap::new();

        // Group tokens by mode
        let groups = self.group_by_mode(tokens);

        for (mode, mode_tokens) in groups {
            let model = match mode {
                TokenMode::Text => &self.text_model,
                TokenMode::InlineMath | TokenMode::DisplayMath => &self.math_model,
                TokenMode::Command => &self.command_model,
                _ => &self.mixed_model,
            };

            let mode_score = model.log_prob(&mode_tokens);
            total_score += mode_score;
            *mode_counts.entry(mode).or_default() += mode_tokens.len();
        }

        // Normalize by total tokens
        total_score / tokens.len() as f64
    }
}
```

## N-gram Order

Different orders for different contexts:

| Context | Recommended Order | Rationale |
|---------|------------------|-----------|
| Text | 5 | Natural language patterns |
| Math | 3 | Shorter formulaic patterns |
| Command | 5 | Command + argument patterns |

## Smoothing

Kneser-Ney smoothing is applied by default:

```rust
pub enum Smoothing {
    /// No smoothing (raw counts)
    None,
    /// Add-k smoothing
    AddK { k: f64 },
    /// Kneser-Ney smoothing (recommended)
    KneserNey { discount: f64 },
    /// Modified Kneser-Ney with multiple discounts
    ModifiedKneserNey {
        d1: f64,
        d2: f64,
        d3: f64,
    },
}
```

## Candidate Scoring

Score multiple correction candidates:

```rust
let candidates = vec![
    r"\begin{equation}",
    r"\bgegin{equation}",  // typo
    r"\begin{equaiton}",   // typo
];

let scores: Vec<f64> = candidates
    .iter()
    .map(|c| model.score(&tokenizer.tokenize(c)))
    .collect();

// Higher score = more probable
let best_idx = scores.iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(i, _)| i)
    .unwrap();
```

## Perplexity Computation

```rust
impl LaTeXNgramModel {
    /// Compute perplexity (lower = better)
    pub fn perplexity(&self, tokens: &[LaTeXToken]) -> f64 {
        let log_prob = self.score(tokens);
        let n = tokens.len() as f64;
        (-log_prob / n).exp()
    }
}
```

## Training

### Data Preparation

```rust
use libgrammstein::latex::NgramTrainer;

let trainer = NgramTrainer::new(NgramTrainerConfig {
    order: 5,
    min_count: 2,
    smoothing: Smoothing::KneserNey { discount: 0.75 },
});

// Process arXiv corpus
for paper in arxiv_corpus.iter() {
    let tokens = tokenizer.tokenize(&paper.content);
    trainer.add_document(&tokens);
}

let model = trainer.build()?;
model.save("latex_model.5gram")?;
```

### Mode-Specific Training

```rust
// Train separate models for each mode
let text_trainer = NgramTrainer::new(config.clone());
let math_trainer = NgramTrainer::new(config.clone());
let cmd_trainer = NgramTrainer::new(config.clone());

for paper in arxiv_corpus.iter() {
    let tokens = tokenizer.tokenize(&paper.content);

    // Split by mode
    let text_tokens: Vec<_> = tokens.iter()
        .filter(|t| t.mode == TokenMode::Text)
        .collect();
    let math_tokens: Vec<_> = tokens.iter()
        .filter(|t| matches!(t.mode, TokenMode::InlineMath | TokenMode::DisplayMath))
        .collect();
    let cmd_tokens: Vec<_> = tokens.iter()
        .filter(|t| t.kind == LaTeXTokenKind::Command)
        .collect();

    text_trainer.add_document(&text_tokens);
    math_trainer.add_document(&math_tokens);
    cmd_trainer.add_document(&cmd_tokens);
}
```

## Context Window

The model considers context across mode boundaries:

```rust
// Input: "According to equation $\alpha = \beta$"
// Context for '\alpha':
//   Previous: ["equation", "$"]  (cross-mode)
//   Current: ["\alpha"]
//   Next: ["=", "\beta"]
```

## Lazy Loading

For memory efficiency, models can be loaded lazily:

```rust
use libgrammstein::integration::LazyNgramModel;

let model = LazyNgramModel::new("models/latex.5gram");

// Model loaded on first use
let score = model.score(&tokens)?;
```

## Serialization

Models can be serialized for deployment:

```rust
// Binary format (fastest loading)
model.save_binary("model.bin")?;
let model = LaTeXNgramModel::load_binary("model.bin")?;

// ARPA format (interoperable)
model.save_arpa("model.arpa")?;
let model = LaTeXNgramModel::load_arpa("model.arpa")?;
```

## Performance

| Model Size | Load Time | Query Time |
|------------|-----------|------------|
| 100MB | 200ms | 0.3ms |
| 500MB | 800ms | 0.5ms |
| 1GB | 1.5s | 0.8ms |

## Related

- [Tokenizer](./tokenizer.md): Token generation
- [Neural Rescorer](./rescorer.md): Neural reranking
- [Combined Scorer](./scorer.md): Score combination
