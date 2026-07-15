# HybridLanguageModel API Reference

`HybridLanguageModel<D>` fuses the two experts: the razor-sharp *local* statistics of the
Modified Kneser-Ney n-gram model and the *semantic generalization* of the subword embedding.
Their errors are largely independent — the n-gram fails on unseen words, the embedding fails on
word order — so a weighted combination is more robust than either alone, and an out-of-vocabulary
word still receives a sane score.

> **Scope.** Source of truth: [`src/hybrid/model.rs`](../../src/hybrid/model.rs),
> [`src/hybrid/mod.rs`](../../src/hybrid/mod.rs), [`src/hybrid/oov.rs`](../../src/hybrid/oov.rs).
> For the mathematics of the four strategies and the score cache, see
> [Hybrid Interpolation](../components/hybrid/interpolation.md); for OOV behavior see
> [OOV Handling](../components/hybrid/oov-handling.md); for tuning $`\alpha`$ see the
> [Hybrid Training guide](../training/hybrid.md).

## Exports

```rust
use libgrammstein::hybrid::{
    HybridLanguageModel, HybridConfig, InterpolationStrategy,
    OovHandler, OovStrategy,                    // standalone helpers — see the note below
    SerializableHybridModel, PathMapHybridModel, // type aliases
};

#[cfg(feature = "serde-extras")]
use libgrammstein::hybrid::PortableHybridModel;
```

These types are **not** in the crate prelude; import them from `libgrammstein::hybrid`.

## The type parameter `D`, and its bounds

`D` is the n-gram model's dictionary backend. As with `NgramModel`, the bound tightens with what
you ask for:

| Operation | Required bound on `D` |
|---|---|
| Construct, `score`, `perplexity`, `predict_next`, … | `MappedDictionary<Value = NgramEntry> + Send + Sync` |
| `to_portable`, `save_portable` | `+ IterableDictionary` |
| `load_portable` | `MutableMappedDictionary<Value = NgramEntry> + Send + Sync` |
| `save`, `load` (direct bincode) | `+ Serialize + DeserializeOwned` |

```rust
pub type SerializableHybridModel = HybridLanguageModel<DynamicDawgChar<NgramEntry>>;
pub type PathMapHybridModel      = HybridLanguageModel<PathMapDictionary<NgramEntry>>;
```

## Construction

There is **no trainer** for the hybrid model — you train the two components separately and
compose them. The constructor takes **three** arguments (or two, for defaults):

```rust
impl<D: MappedDictionary<Value = NgramEntry> + Send + Sync> HybridLanguageModel<D> {
    pub fn new(ngram: NgramModel<D>, embedding: SubwordEmbedding, config: HybridConfig) -> Self;
    pub fn with_defaults(ngram: NgramModel<D>, embedding: SubwordEmbedding) -> Self;
}
```

Both **take ownership** of the components. Borrow them back with `ngram_model()` and
`embedding_model()`.

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

let ngram = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(5)
    .train(PlaintextReader::from_file("corpus.txt")?)?;

let embedding = EmbeddingTrainerBuilder::new()
    .dim(100)
    .epochs(5)
    .train(PlaintextReader::from_file("corpus.txt")?)?;   // a FRESH reader: the first was moved

// Explicit configuration …
let config = HybridConfig {
    strategy: InterpolationStrategy::LogLinear { alpha: 0.7 },
    ..Default::default()
};
let hybrid = HybridLanguageModel::new(ngram, embedding, config);

// … or the defaults (Linear { alpha: 0.8 }):
// let hybrid = HybridLanguageModel::with_defaults(ngram, embedding);
# Ok::<(), libgrammstein::Error>(())
```

## Configuration

```rust
pub struct HybridConfig {
    pub strategy: InterpolationStrategy,  // default: Linear { alpha: 0.8 }
    pub cache_size: usize,                // default: 50_000
    pub embedding_smoothing: f64,         // default: 1e-8
    pub temperature: f64,                 // default: 1.0
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `strategy` | `InterpolationStrategy` | `Linear { alpha: 0.8 }` | How the two experts are fused. |
| `cache_size` | `usize` | `50_000` | Capacity of the LRU score cache (clamped to $`\geq 1`$). |
| `embedding_smoothing` | `f64` | `1e-8` | $`\varepsilon`$ — the floor on the embedding log-probability. |
| `temperature` | `f64` | `1.0` | $`\tau`$ — divides the cosine similarity before it becomes a log-probability. Lower sharpens. |

### `InterpolationStrategy`

With $`\mathbb{P}_n`$ the n-gram probability, $`\mathbb{P}_e`$ the embedding-derived probability,
$`\alpha`$ the n-gram weight, and $`\lvert h \rvert`$ the context length:

```rust
pub enum InterpolationStrategy {
    Linear { alpha: f64 },
    LogLinear { alpha: f64 },
    NgramWithEmbeddingFallback,
    Dynamic { base_alpha: f64, alpha_per_context: f64, max_alpha: f64 },
}
```

**`Linear`** — a convex combination of the *probabilities*; predictable, and the default:

```math
\begin{array}{lr}
\displaystyle \mathbb{P}(w \mid h) = \alpha\,\mathbb{P}_n(w \mid h) + (1 - \alpha)\,\mathbb{P}_e(w \mid h) & \text{(H1)}
\end{array}
```

**`LogLinear`** — a convex combination in *log space* (a geometric mean / product-of-experts).
Use it when the two experts live on different scales: a low probability from *either* strongly
suppresses the product.

```math
\begin{array}{lr}
\displaystyle \log \mathbb{P}(w \mid h) = \alpha \log \mathbb{P}_n(w \mid h) + (1 - \alpha) \log \mathbb{P}_e(w \mid h) & \text{(H2)}
\end{array}
```

**`NgramWithEmbeddingFallback`** — a hard switch, tested by `count(&[w]) > 0`. No interpolation
overhead for known words:

```math
\begin{array}{lr}
\displaystyle \mathbb{P}(w \mid h) = \begin{cases}
\mathbb{P}_n(w \mid h) & \text{if } c(w) > 0 \\
\mathbb{P}_e(w \mid h) & \text{otherwise}
\end{cases} & \text{(H3)}
\end{array}
```

**`Dynamic`** — linear interpolation whose weight *grows with the available context*, trusting
the n-gram more as the history lengthens:

```math
\begin{array}{lr}
\displaystyle \alpha(h) = \min\bigl(\alpha_0 + \kappa \cdot \lvert h \rvert,\ \alpha_{\max}\bigr) & \text{(H4)}
\end{array}
```

with `base_alpha` = $`\alpha_0`$, `alpha_per_context` = $`\kappa`$, `max_alpha` = $`\alpha_{\max}`$.
`Dynamic` has no `Default`; supply all three fields, e.g.
`Dynamic { base_alpha: 0.5, alpha_per_context: 0.1, max_alpha: 0.9 }`.

| Scenario | Strategy | Typical weight |
|---|---|---|
| General purpose, both models reliable | `Linear` | $`\alpha \approx 0.8`$ |
| Components on different score scales | `LogLinear` | $`\alpha \approx 0.6`$–$`0.8`$ |
| Large, high-quality n-gram corpus | `NgramWithEmbeddingFallback` | — |
| High OOV rate / mixed vocabulary | `Dynamic` | $`\alpha_0 = 0.7,\ \alpha_{\max} = 0.95`$ |

## Scoring

```rust
impl<D: MappedDictionary<Value = NgramEntry> + Send + Sync> HybridLanguageModel<D> {
    pub fn score(&self, word: &str, context: &[&str]) -> f64;
    pub fn sentence_log_prob(&self, words: &[&str]) -> f64;
    pub fn perplexity(&self, words: &[&str]) -> f64;
    pub fn predict_next(&self, context: &[&str], candidates: &[&str]) -> Option<(String, f64)>;
    pub fn clear_cache(&self);
    pub fn ngram_model(&self) -> &NgramModel<D>;
    pub fn embedding_model(&self) -> &SubwordEmbedding;
    pub fn config(&self) -> &HybridConfig;
}
```

| Method | Returns | Description |
|---|---|---|
| `score(word, context)` | `f64` | The interpolated **log**-probability, per the configured strategy. Memoized. |
| `sentence_log_prob(words)` | `f64` | $`\sum_i \texttt{score}(w_i, h_i)`$, sliding a window of up to `order - 1` context words. `0.0` for an empty slice. |
| `perplexity(words)` | `f64` | $`\exp(-\texttt{sentence\_log\_prob} / N)`$. **`f64::INFINITY`** for an empty slice. |
| `predict_next(context, candidates)` | `Option<(String, f64)>` | The highest-scoring candidate and its score. `None` **only** if `candidates` is empty. |
| `clear_cache()` | — | Empties the score cache. Takes `&self` (the cache is interior-mutable). |

![Hybrid interpolation scoring flow](../diagrams/hybrid-scoring.svg)

```rust
// Robust even when "brown" is rare in the n-gram corpus.
let log_p = hybrid.score("brown", &["the", "quick"]);

// An OOV word still gets a finite, meaningful score via the subword vectors.
let oov = hybrid.score("xyzzy", &["magic", "word"]);

// Rank candidate corrections in context.
let candidates = ["their", "there", "they're"];
if let Some((best, score)) = hybrid.predict_next(&["put", "it", "over"], &candidates) {
    println!("best = {best} ({score:.4})");
}

let ppl = hybrid.perplexity(&["the", "quick", "brown", "fox"]);
```

### How the embedding becomes a probability

For a non-empty context the model averages the context words' subword vectors into $`v_h`$,
takes the cosine to the candidate's vector $`v_w`$, scales by $`\tau`$, and works in log space:

```math
\begin{array}{lr}
\displaystyle \log \mathbb{P}_e(w \mid h) \;\approx\; \frac{\cos(v_w, v_h)}{\tau} - 1,
\qquad \text{floored at } \log \varepsilon & \text{(H5)}
\end{array}
```

For an **empty** context it falls back to the uniform $`-\log \lvert V_e \rvert`$ over the
*embedding* vocabulary.

> **$`\mathbb{P}_e`$ is unnormalized.** The $`-1`$ in $`(\mathrm{H5})`$ is a cheap stand-in for
> the true softmax normalizer, which would cost a sum over the whole vocabulary per query. It is
> a *ranking* score, not a calibrated probability — the n-gram side supplies calibration, the
> embedding side supplies OOV coverage and semantic tie-breaking. See
> [Hybrid Interpolation](../components/hybrid/interpolation.md) for the full argument.

### Guards that keep the score finite

Both experts' log-probabilities are **clamped to $`\geq -50`$** before combination, so a single
catastrophically-low term cannot annihilate the score; and `Linear` floors the combined
probability at `f64::MIN_POSITIVE` before taking its log. `score` therefore always returns a
finite `f64`.

### The score cache

Scores are memoized in a lock-free cache keyed by a hash of $`(w, h)`$: a `DashMap` for the
entries (lock-free get and insert), an `AtomicUsize` counter, and a `Mutex<VecDeque>` recording
insertion order for LRU eviction — the **only** lock, taken **only** when the cache is over
`cache_size`. The model is `Send + Sync`, so one instance can be scored from many threads at
once. The cache is never serialized; it is reconstructed empty on `load`.

> **Cache invalidation is your job.** The key is $`(w, h)`$ only — it does not incorporate the
> configuration. The config is immutable after construction, so this is sound in normal use, but
> if you mutate the underlying components through any other path, call `clear_cache()`.

## Persistence (feature `serde-extras`)

| Method | Bound on `D` | Notes |
|---|---|---|
| `save(path)` / `load(path)` | `+ Serialize + DeserializeOwned` | Direct bincode. In practice: `DynamicDawgChar` only. |
| `to_portable()` / `save_portable(path)` | `+ IterableDictionary` | Backend-agnostic: `PortableHybridModel { ngram, embedding, config }`. |
| `load_portable(path, factory)` | `MutableMappedDictionary` | Rebuilds the n-gram trie through the `FnOnce() -> D` factory. |

```rust
use libgrammstein::hybrid::HybridLanguageModel;
use libgrammstein::ngram::NgramEntry;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

hybrid.save("hybrid.bin")?;
let hybrid: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load("hybrid.bin")?;

// Portable — the factory supplies the empty backend to rebuild into.
hybrid.save_portable("hybrid.portable.bin")?;
let hybrid: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load_portable("hybrid.portable.bin", DynamicDawgChar::new)?;
# Ok::<(), libgrammstein::Error>(())
```

## `OovHandler` and `OovStrategy`

`libgrammstein::hybrid` also exports a **standalone** OOV utility:

```rust
pub enum OovStrategy {
    SubwordEmbedding,                    // default: cosine of the subword-composed vector
    FixedProbability { log_prob: f64 },
    Uniform,                             // -ln(vocab_size)
    SimilarWords { k: usize },           // estimate from the k nearest in-vocabulary words
}

pub struct OovHandler<'a> { /* borrows a &'a SubwordEmbedding */ }
impl<'a> OovHandler<'a> {
    pub fn new(embedding: &'a SubwordEmbedding, strategy: OovStrategy) -> Self;
    pub fn estimate_log_prob(&self, word: &str, context: &[&str]) -> f64;
}
```

> **`HybridLanguageModel::score` does not route through `OovHandler`.** The model inlines its own
> embedding path ($`(\mathrm{H5})`$, with `temperature` and `embedding_smoothing`), and never
> consults `OovStrategy`. `OovHandler` is a separate, composable helper for callers who want an
> OOV estimate — in particular `SimilarWords`, which the model itself does not offer. Configuring
> an `OovStrategy` has **no effect** on the hybrid model's scores.

## Complete workflow

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

fn main() -> libgrammstein::Result<()> {
    // 1. Train the two experts (each trainer moves its reader — build a fresh one).
    let ngram = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(5)
        .train(PlaintextReader::from_file("corpus.txt")?)?;

    let embedding = EmbeddingTrainerBuilder::new()
        .dim(100)
        .epochs(5)
        .train(PlaintextReader::from_file("corpus.txt")?)?;

    // 2. Fuse them.
    let hybrid = HybridLanguageModel::new(
        ngram,
        embedding,
        HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha: 0.7 },
            ..Default::default()
        },
    );

    // 3. Score.
    let sentence = ["the", "quick", "brown", "fox"];
    println!("log P   = {:.4}", hybrid.sentence_log_prob(&sentence));
    println!("PP      = {:.2}", hybrid.perplexity(&sentence));
    println!("OOV     = {:.4}", hybrid.score("xyzzy", &["magic", "word"]));

    // 4. Persist (feature: serde-extras).
    hybrid.save("hybrid.bin")?;
    Ok(())
}
```

## Notes and caveats

| Caveat | Detail |
|---|---|
| **No corpus-level perplexity helper** | [`scoring::Perplexity`](scoring.md) is generic over `NgramModel<D>` only. To evaluate a hybrid model over a corpus, loop the reader's sentences and accumulate `sentence_log_prob` and the token count yourself. |
| **`perplexity(&[])` is `INFINITY`** | Not `NaN`, not an error. Guard empty input if you aggregate. |
| **`predict_next` scans linearly** | It scores every candidate. It does not search the vocabulary — you supply the candidate set. |
| **Cache size is clamped to $`\geq 1`$** | `cache_size: 0` becomes `1`, not "disabled". |

## References

1. F. Jelinek & R. L. Mercer (1980). *Interpolated estimation of Markov source parameters from
   sparse data.* In *Pattern Recognition in Practice*, 381–397. North-Holland.
2. G. E. Hinton (2002). *Training products of experts by minimizing contrastive divergence.*
   Neural Computation 14(8), 1771–1800.
   [doi:10.1162/089976602760128018](https://doi.org/10.1162/089976602760128018)
3. P. Bojanowski, E. Grave, A. Joulin & T. Mikolov (2017). *Enriching word vectors with subword
   information* (FastText). Transactions of the ACL 5, 135–146.
   [doi:10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)

## See also

- [Hybrid Interpolation](../components/hybrid/interpolation.md) — the four strategies in depth
- [Hybrid overview](../components/hybrid/overview.md) — architecture of the combined model
- [OOV Handling](../components/hybrid/oov-handling.md) — out-of-vocabulary strategies
- [NgramModel API](ngram.md) — the statistical expert
- [SubwordEmbedding API](embedding.md) — the semantic expert
- [Scoring API](scoring.md) — perplexity and ranking (n-gram models)
- [Hybrid Training guide](../training/hybrid.md) — grid-searching $`\alpha`$
