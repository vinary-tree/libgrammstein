# Hybrid Interpolation: Statistics ⊕ Semantics

The **hybrid language model** combines a Modified Kneser-Ney n-gram model with a subword
embedding model, so that the razor-sharp *local* statistics of the n-gram and the *semantic
generalization* of the embedding cover each other's blind spots. This document describes the
four interpolation strategies, the mathematics behind each, and how libgrammstein implements
scoring and caching.

> **Scope.** Source of truth: [`src/hybrid/model.rs`](../../../src/hybrid/model.rs). For
> out-of-vocabulary behavior see [OOV Handling](oov-handling.md); for the API surface see the
> [Hybrid API reference](../../api/hybrid.md); for tuning see
> [Hybrid Training](../../training/hybrid.md).

## Why interpolate?

The two component models fail in **complementary** ways:

| Model | Strong when | Weak when |
|---|---|---|
| N-gram (MKN) | the exact local context was seen in training | the word or context is unseen (OOV) |
| Subword embedding | the word is semantically close to known words | precise word order matters |

Because their errors are largely independent, a weighted combination is more robust than either
alone. Let $`\mathbb{P}_n(w \mid h) = \mathbb{P}_{\mathrm{MKN}}(w \mid h)`$ be the n-gram
probability (see [Modified Kneser-Ney](../ngram/modified-kneser-ney.md)) and
$`\mathbb{P}_e(w \mid h)`$ the embedding-derived probability (defined [below](#the-embedding-probability)).
An interpolation strategy is a rule for fusing $`\mathbb{P}_n`$ and $`\mathbb{P}_e`$ into a single
score.

**Notation.** $`w`$ is the candidate word, $`h`$ its context, $`\alpha \in [0,1]`$ the n-gram
mixing weight, $`\tau`$ the temperature, $`V`$ the n-gram vocabulary, and $`\lvert h \rvert`$ the
context length in words.

![Hybrid interpolation scoring flow](../../diagrams/hybrid-scoring.svg)

## The four strategies

libgrammstein exposes the strategies as an `InterpolationStrategy` enum; `HybridConfig::default`
selects `Linear { alpha: 0.8 }`.

### Linear

A convex combination of the two **probabilities** — the classic interpolated estimator
[[1]](#references):

```math
\mathbb{P}(w \mid h) = \alpha\,\mathbb{P}_n(w \mid h) + (1 - \alpha)\,\mathbb{P}_e(w \mid h) \tag{H1}
```

$`\alpha = 1`$ is pure n-gram; $`\alpha = 0`$ is pure embedding. Predictable and a good default.

### Log-Linear

A convex combination in **log space** — a geometric mean, i.e. a product-of-experts
[[2]](#references):

```math
\log \mathbb{P}(w \mid h) = \alpha \log \mathbb{P}_n(w \mid h) + (1 - \alpha) \log \mathbb{P}_e(w \mid h) \tag{H2}
```

This is the right choice when $`\mathbb{P}_n`$ and $`\mathbb{P}_e`$ live on different scales,
because a low probability from *either* expert strongly suppresses the product (an AND-like
combination), which sharpens distinctions and avoids probability extremes.

### N-gram with embedding fallback

A hard switch: trust the n-gram when the word is in-vocabulary, otherwise defer entirely to the
embedding.

```math
\mathbb{P}(w \mid h) = \begin{cases}
\mathbb{P}_n(w \mid h) & \text{if } w \in V \\
\mathbb{P}_e(w \mid h) & \text{otherwise}
\end{cases} \tag{H3}
```

Membership is tested by $`c(w) > 0`$ (a non-zero unigram count). There is no interpolation
overhead for known words — ideal when the n-gram corpus is large and its quality is high.

### Dynamic

Linear interpolation whose weight **grows with the available context**, trusting the n-gram more
as the history lengthens (where local statistics are most reliable):

```math
\alpha(h) = \min\bigl(\alpha_0 + \kappa \cdot \lvert h \rvert,\ \alpha_{\max}\bigr),
\qquad
\mathbb{P}(w \mid h) = \alpha(h)\,\mathbb{P}_n + (1 - \alpha(h))\,\mathbb{P}_e \tag{H4}
```

The three parameters are the struct fields `base_alpha` ($`\alpha_0`$), `alpha_per_context`
($`\kappa`$), and `max_alpha` ($`\alpha_{\max}`$).

## The embedding probability

The embedding side converts geometric similarity into a probability. For a non-empty context,
libgrammstein forms the context vector $`v_h`$ as the mean of the context words' subword vectors
(`SubwordEmbedding::sentence_vector`), takes the **cosine similarity** to the candidate's vector
$`v_w`$, scales by temperature $`\tau`$, and works in log space:

```math
\cos(v_w, v_h) = \frac{v_w \cdot v_h}{\lVert v_w \rVert\,\lVert v_h \rVert},
\qquad
\log \mathbb{P}_e(w \mid h) \approx \frac{\cos(v_w, v_h)}{\tau} - 1 \tag{H5}
```

The result is floored at $`\log(\varepsilon)`$ with $`\varepsilon =`$ `embedding_smoothing`
(default $`10^{-8}`$). For an **empty** context the embedding falls back to the uniform
$`\log \mathbb{P}_e = -\log \lvert V_e \rvert`$.

> **Honest approximation.** The $`-1`$ term in $`(\mathrm{H5})`$ is a cheap stand-in for the true
> softmax normalizer $`\log \sum_{w'} \exp(\cos(v_{w'}, v_h)/\tau)`$, which would require a sum
> over the whole vocabulary per query. $`\mathbb{P}_e`$ is therefore an *unnormalized* score used
> for **ranking and interpolation**, not a calibrated probability. This is intentional: the
> n-gram side supplies calibration, the embedding side supplies OOV coverage and semantic tie-breaking.

## Scoring, literately

The following mirrors [`HybridLanguageModel::score`](../../../src/hybrid/model.rs). Log-probs
from either expert are clamped to $`\geq -50`$ before combination so a single $`-\infty`$ cannot
annihilate the score.

```
function score(w, h):
    if (w, h) in cache: return cache[(w, h)]          ▸ lock-free DashMap probe (no lock)
    s <- match strategy:
          Linear{alpha}      -> score_linear(w, h, alpha)
          LogLinear{alpha}   -> alpha*clamp(log Pn) + (1-alpha)*clamp(log Pe)      ▸ (H2)
          Fallback           -> clamp(log Pn) if count(w) > 0 else clamp(log Pe)   ▸ (H3)
          Dynamic{a0,k,amax} -> score_linear(w, h, min(a0 + k*|h|, amax))          ▸ (H4)
    cache.insert((w, h), s)                           ▸ lock taken only on LRU eviction
    return s

function score_linear(w, h, alpha):                   ▸ clamp(x) = max(x, -50)
    pn <- exp(clamp(log Pn(w, h)))
    pe <- exp(clamp(log Pe(w, h)))
    return ln( max( alpha*pn + (1-alpha)*pe,  f64::MIN_POSITIVE ) )   ▸ (H1); MIN_POSITIVE keeps log finite
```

## Engineering

### Lock-free score cache

Scores are memoized in a `ScoreCache` designed for concurrent scoring without blocking the hot
path:

- a `DashMap<u64, f64>` keyed by a `DefaultHasher` digest of $`(w, h)`$ — **lock-free** get and
  insert;
- a `Mutex<VecDeque<u64>>` recording insertion order for **LRU eviction** — the *only* lock, and
  it is taken **only** when the cache is over capacity (`cache_size`, default `50_000`);
- an `AtomicUsize` entry counter for a fast size check.

The map is `Send + Sync`, so a single `HybridLanguageModel` can be scored from many threads at
once. The cache is not serialized; it is reconstructed empty on load.

### Configuration

```rust
pub struct HybridConfig {
    pub strategy: InterpolationStrategy, // default: Linear { alpha: 0.8 }
    pub cache_size: usize,               // default: 50_000
    pub embedding_smoothing: f64,        // default: 1e-8  (the epsilon floor in H5)
    pub temperature: f64,                // default: 1.0   (the tau in H5)
}
```

## Usage

```rust
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};

// `ngram` and `embedding` trained beforehand (see the ngram + embedding docs).
let config = HybridConfig {
    strategy: InterpolationStrategy::LogLinear { alpha: 0.7 },
    ..Default::default()
};
let hybrid = HybridLanguageModel::new(ngram, embedding, config);

// Robust even when "brown" is rare in the n-gram corpus.
let log_p = hybrid.score("brown", &["the", "quick"]);

// Or take the defaults (Linear alpha = 0.8) with the two-argument constructor:
// let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);
```

Persistence is feature-gated on `serde-extras`: `save`/`load` require a `serde`-able dictionary
backend, while `save_portable`/`load_portable` are backend-agnostic (the latter rebuilds the
n-gram trie through a `dictionary_factory` closure).

## Choosing a strategy

| Scenario | Strategy | Weight |
|---|---|---|
| General purpose, both models reliable | `Linear` | $`\alpha \approx 0.8`$ |
| Components on different score scales | `LogLinear` | $`\alpha \approx 0.6`$–$`0.8`$ |
| Large, high-quality n-gram corpus | `NgramWithEmbeddingFallback` | — |
| High OOV rate / mixed vocabulary | `Dynamic` | $`\alpha_0 = 0.7,\ \alpha_{\max} = 0.95`$ |

See [Hybrid Training](../../training/hybrid.md) for grid-search tuning of $`\alpha`$ against
development-set perplexity.

## References

1. F. Jelinek & R. L. Mercer (1980). *Interpolated estimation of Markov source parameters from
   sparse data.* In *Pattern Recognition in Practice*, 381–397. North-Holland.
2. G. E. Hinton (2002). *Training products of experts by minimizing contrastive divergence.*
   Neural Computation 14(8), 1771–1800.
   [doi:10.1162/089976602760128018](https://doi.org/10.1162/089976602760128018)
3. P. Bojanowski, E. Grave, A. Joulin & T. Mikolov (2017). *Enriching word vectors with subword
   information* (FastText). TACL 5, 135–146.
   [doi:10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)

## See also

- [Hybrid Overview](overview.md) — architecture of the combined model
- [OOV Handling](oov-handling.md) — out-of-vocabulary strategies in depth
- [Modified Kneser-Ney](../ngram/modified-kneser-ney.md) — the n-gram expert
- [Subword Embeddings](../embedding/overview.md) — the embedding expert
