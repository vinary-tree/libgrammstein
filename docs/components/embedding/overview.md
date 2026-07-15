# Subword Embeddings

The **embedding model** is libgrammstein's semantic expert: it maps every word — including
words never seen in training — to a dense vector so that words used in similar contexts land
near one another in space. It follows the **FastText** recipe [[2]](#references): a word's vector
is built from the vectors of its **character n-grams** (subwords), which is what gives the model
its out-of-vocabulary (OOV) coverage. This document explains *what* subword embeddings are, the
*mathematics* of how a word vector is composed, and *how libgrammstein implements it* — including
the two places where the shipped code deliberately departs from canonical FastText.

> **Scope.** Source of truth: [`src/embedding/model.rs`](../../../src/embedding/model.rs),
> [`src/embedding/mod.rs`](../../../src/embedding/mod.rs), and
> [`src/embedding/bpe.rs`](../../../src/embedding/bpe.rs). Training is covered in
> [Skip-gram Training](skip-gram.md); the subword mechanics in [BPE & Subword Extraction](bpe.md);
> ranking in [Similarity](similarity.md). This model is the embedding half of the
> [Hybrid Model](../hybrid/interpolation.md).

## Notation

Every symbol is defined before it is used in a formula.

| Symbol | Meaning |
|---|---|
| $`w`$ | the word (token) whose vector is being computed |
| $`V`$ | the in-vocabulary word set; $`\lvert V \rvert`$ its size |
| $`d`$ | the embedding dimension (`dim`; default $`100`$) |
| $`B`$ | the number of subword hash buckets (`bucket_count`; default $`2\,000\,000`$) |
| $`G(w)`$ | the set of boundary-marked character n-grams of $`w`$; $`\lvert G(w) \rvert`$ its size |
| $`g`$ | a single subword (character n-gram) in $`G(w)`$ |
| $`h(g)`$ | the FNV-1a bucket index of subword $`g`$, in $`\{0, \dots, B-1\}`$ |
| $`E_{\mathrm{word}}`$ | the word-embedding matrix, shape $`\lvert V \rvert \times d`$ |
| $`E_{\mathrm{sub}}`$ | the subword-embedding matrix, shape $`B \times d`$ |
| $`v_{\mathrm{word}}(w)`$ | the word-embedding row of $`w`$ (only when $`w \in V`$) |
| $`v_{\mathrm{sub}}(w)`$ | the mean of $`w`$'s subword bucket rows |
| $`\cos(a,b)`$ | cosine similarity between vectors $`a`$ and $`b`$ |

**Acronyms.** *OOV* — Out-Of-Vocabulary; *FNV* — Fowler-Noll-Vo (a hash function);
*BPE* — Byte-Pair Encoding.

## What & why: from one-hot to subwords

A naive representation gives each word a **one-hot** vector of length $`\lvert V \rvert`$ — all
zeros but a single one. One-hot vectors are enormous, and any two distinct words are exactly
equidistant, so they carry *no* notion of similarity. **Embeddings** replace them with short dense
vectors learned so that geometric closeness reflects distributional similarity ("a word is known
by the company it keeps").

| Property | One-hot | Dense embedding |
|---|---|---|
| Size per word | $`O(\lvert V \rvert)`$ | $`O(d)`$, $`d \approx 100`$ |
| Similarity | none (all pairs equidistant) | cosine captures semantic closeness |
| OOV word | inexpressible | approximated from its subwords |

Classic word embeddings still have one fatal gap: a word absent from the training vocabulary has
**no** vector at all. FastText closes the gap by representing a word through its **character
n-grams**. Because *running*, *runner*, and *runs* share the subwords $`\texttt{run}`$,
$`\texttt{<ru}`$, $`\texttt{unn}`$, they end up with related vectors even if one of them never
appeared in training. The same mechanism lets a fresh OOV word such as *fastly* inherit meaning
from the subwords it shares with *fast* and *quickly*.

## Theory: composing a word vector

### Step 1 — subword decomposition

libgrammstein wraps the word in boundary markers $`\texttt{<}\,w\,\texttt{>}`$ and enumerates
every character n-gram of length $`n \in [\,\texttt{min\_subword\_len},\ \texttt{max\_subword\_len}\,]`$
(default $`[3, 6]`$). The markers let the model distinguish a prefix/suffix from an interior
occurrence — $`\texttt{<he}`$ (word start) differs from $`\texttt{he}`$ inside *the*. For
$`w = \texttt{hello}`$ the marked form is $`\texttt{<hello>}`$ and:

| $`n`$ | subwords $`g \in G(\texttt{hello})`$ |
|---|---|
| 3 | $`\texttt{<he}`$, $`\texttt{hel}`$, $`\texttt{ell}`$, $`\texttt{llo}`$, $`\texttt{lo>}`$ |
| 4 | $`\texttt{<hel}`$, $`\texttt{hell}`$, $`\texttt{ello}`$, $`\texttt{llo>}`$ |
| 5 | $`\texttt{<hell}`$, $`\texttt{hello}`$, $`\texttt{ello>}`$ |
| 6 | $`\texttt{<hello}`$, $`\texttt{hello>}`$ |

This is [`extract_subwords`](bpe.md#character-n-gram-subwords).

### Step 2 — hashing to a fixed bucket table

Storing a row per distinct subword is unbounded, so each subword is hashed to one of $`B`$
buckets with 64-bit **FNV-1a**, and buckets share their embedding rows:

```math
\begin{array}{lr}
\displaystyle h(g) = \mathrm{FNV\text{-}1a}(g) \bmod B & \text{(E1)}
\end{array}
```

Collisions are tolerated: with $`B = 2 \times 10^{6}`$ they are rare, and the skip-gram objective
averages out the few that occur. The bucket table $`E_{\mathrm{sub}}`$ is the model's dominant
memory cost.

### Step 3 — the subword vector and the word vector

The **subword vector** is the *mean* of the bucket rows of the word's n-grams:

```math
\begin{array}{lr}
\displaystyle v_{\mathrm{sub}}(w) = \frac{1}{\lvert G(w) \rvert} \sum_{g \in G(w)} E_{\mathrm{sub}}\bigl[h(g)\bigr] & \text{(E2)}
\end{array}
```

The final **word vector** averages the word row with the subword vector for a known word, and
falls back to the subword vector alone for an OOV word — so *every* string yields a vector:

```math
\begin{array}{lr}
\displaystyle \mathrm{word\_vector}(w) =
\begin{cases}
\tfrac{1}{2}\bigl(v_{\mathrm{word}}(w) + v_{\mathrm{sub}}(w)\bigr) & w \in V \\[2pt]
v_{\mathrm{sub}}(w) & w \notin V
\end{cases} & \text{(E3)}
\end{array}
```

> **Two honest departures from canonical FastText.** Canonical FastText [[2]](#references)
> represents a word as the **sum** $`\sum_{g} z_g`$ of its subword vectors (plus a dedicated
> whole-word row) and does not renormalize. libgrammstein instead (i) **averages** the subword
> rows in $`(\mathrm{E2})`$ and (ii) **averages** that with the word row in $`(\mathrm{E3})`$.
> Averaging keeps a word's vector norm roughly independent of its length (long words have many
> subwords), which stabilizes the cosine comparisons in [Similarity](similarity.md); the trade-off
> is that raw additive compositionality is weaker than in the summed formulation. Note also that
> `word_vector` does **not** L2-normalize its output — cosine routines normalize on demand.

A **sentence vector** is simply the mean of its word vectors (used as the context vector by the
hybrid model — see [Hybrid Interpolation](../hybrid/interpolation.md)):

```math
\begin{array}{lr}
\displaystyle \mathrm{sentence\_vector}(w_1 \dots w_m) = \frac{1}{m} \sum_{i=1}^{m} \mathrm{word\_vector}(w_i) & \text{(E4)}
\end{array}
```

![Subword composition of a word vector](../../diagrams/embedding-subword.svg)

## The algorithm, literately

The following mirrors [`SubwordEmbedding::word_vector`](../../../src/embedding/model.rs) and its
private helper `subword_vector`. `⟨…⟩` names a refinement expanded below; `▸` marks a
side-comment. Inside pseudocode all operators are ASCII.

```
function word_vector(w):                          ▸ public, memoized entry point
    if w in cache: return cache[w]                ▸ lock-free DashMap probe
    v_sub <- subword_vector(w)
    if w in V:                                     ▸ known word
        v_word <- word_embeddings[ index_of(w) ]
        v <- (v_word + v_sub) / 2                  ▸ (E3), in-vocab branch
    else:                                          ▸ OOV: subwords carry all the signal
        v <- v_sub                                 ▸ (E3), OOV branch
    if size(cache) < max_cache_size: cache[w] <- v ▸ bounded insert (default 100_000)
    return v

function subword_vector(w):                        ▸ (E2)
    G <- extract_subwords(w, min_subword_len, max_subword_len)
    if G is empty: return zeros(d)                 ▸ e.g. a word shorter than min_n
    s <- zeros(d)
    for g in G:
        b <- hash_subword(g, B)                    ▸ FNV-1a mod B, per (E1)
        s <- s + subword_embeddings[b]
    return s / |G|                                 ▸ MEAN of bucket rows
```

`word_vector_uncached` is the identical computation without the cache probe/insert; training uses
it to avoid populating the cache with transient values.

## Engineering

### The `SubwordEmbedding` struct

Fields are private; the two dense matrices are `ndarray::Array2<f32>` in row-major (C) order, so a
word's $`d`$ values are contiguous.

```rust
use ndarray::{Array1, Array2};
use dashmap::DashMap;
use std::{collections::HashMap, sync::Arc};

pub struct SubwordEmbedding {
    word_embeddings: Array2<f32>,     // [ |V| x d ]  E_word
    subword_embeddings: Array2<f32>,  // [ B  x d ]  E_sub  (dominant memory cost)
    word_to_idx: HashMap<String, usize>,
    idx_to_word: Vec<String>,         // index-ordered vocabulary; keeps rows aligned
    dim: usize,                       // d
    bucket_count: usize,              // B
    min_subword_len: usize,           // default 3
    max_subword_len: usize,           // default 6
    tokenizer: Option<BpeTokenizer>,  // optional BPE segmenter (see bpe.md)
    cache: Arc<DashMap<String, Array1<f32>>>, // #[serde(skip)] — rebuilt empty on load
    max_cache_size: usize,            // default 100_000
}
```

The exported defaults are `DEFAULT_EMBEDDING_DIM = 100`, `DEFAULT_BUCKET_COUNT = 2_000_000`,
`DEFAULT_MIN_SUBWORD_LEN = 3`, and `DEFAULT_MAX_SUBWORD_LEN = 6`.

### Concurrency and persistence

- The two weight matrices are **immutable after training**, so any number of threads may read
  vectors concurrently.
- The vector `cache` is an `Arc<DashMap<..>>` — **lock-free** concurrent get/insert. It is bounded
  by `max_cache_size` (a simple length gate, not strict LRU), marked `#[serde(skip)]`, and
  `Clone` deliberately gives the clone a **fresh empty** cache.
- Persistence is feature-gated on `serde-extras`: `save`/`load` use `bincode` to write/read the
  whole model (`SubwordEmbedding::save`, `SubwordEmbedding::load`).

### Complexity

Let $`s = \lvert G(w) \rvert`$ be the subword count of $`w`$ (bounded by word length and the
n-gram range).

| Operation | Cost | Notes |
|---|---|---|
| `word_vector` (cache hit) | $`O(d)`$ | DashMap probe + clone |
| `word_vector` (miss) | $`O(s\,d)`$ | one bucket row summed per subword |
| `similarity` | $`O(d)`$ | two vectors + one dot product |
| `most_similar` | $`O(\lvert V \rvert\,d)`$ | full scan of $`E_{\mathrm{word}}`$ (see [Similarity](similarity.md)) |

### Memory layout

With the defaults $`d = 100`$, $`B = 2 \times 10^{6}`$, and $`\lvert V \rvert = 2 \times 10^{5}`$
(and $`4`$ bytes per `f32`):

```math
\begin{array}{lr}
\displaystyle \underbrace{2{\times}10^{5} \cdot 100 \cdot 4}_{E_{\mathrm{word}} \approx 80\ \text{MB}}
\;+\;
\underbrace{2{\times}10^{6} \cdot 100 \cdot 4}_{E_{\mathrm{sub}} \approx 800\ \text{MB}} & \text{(E5)}
\end{array}
```

The subword table dominates; shrinking $`B`$ trades memory for a higher subword collision rate.

## Usage

Train a model, then query it through the real API. `word_vector` never fails — an OOV word still
returns a subword-derived vector.

```rust
use libgrammstein::embedding::{EmbeddingTrainerBuilder, SubwordEmbedding};
use libgrammstein::corpus::PlaintextReader;

// Train FastText-style subword embeddings from a plaintext corpus.
let reader = PlaintextReader::from_file("corpus.txt")?;
let model: SubwordEmbedding = EmbeddingTrainerBuilder::new()
    .dim(100)
    .window_size(5)
    .min_count(5)
    .epochs(5)
    .train(reader)?;

// A known word: word row averaged with its subword vector.
let v = model.word_vector("running");
assert_eq!(v.len(), model.dim());

// An OOV word still resolves, via shared subwords with the vocabulary.
let oov = model.word_vector("fastly");
assert_eq!(oov.len(), model.dim());

// Nearest neighbours and a sentence (context) vector.
let neighbours = model.most_similar("king", 10);   // Vec<(String, f32)>
let ctx = model.sentence_vector(&["the", "quick", "brown"]);
println!("vocab = {}, neighbours[0] = {:?}", model.vocab_size(), neighbours.first());
# Ok::<(), libgrammstein::Error>(())
```

## Hyperparameters

These are the fields of [`EmbeddingConfig`](../../../src/embedding/trainer.rs) with their shipped
defaults; the builder in the snippet above sets a subset. See [Skip-gram Training](skip-gram.md)
for the training-specific ones and [Hyperparameters](../../training/hyperparameters.md) for tuning.

| Parameter | Default | Effect |
|---|---|---|
| `dim` | $`100`$ | vector width — higher is more expressive, more memory |
| `window_size` | $`5`$ | context radius during training |
| `min_count` | $`5`$ | drop words rarer than this from $`V`$ |
| `neg_samples` | $`5`$ | negative samples per positive (see skip-gram) |
| `epochs` | $`5`$ | passes over the corpus |
| `learning_rate` | $`0.05`$ | initial step size (linearly decayed per epoch) |
| `bucket_count` | $`2\,000\,000`$ | subword hash table size $`B`$ |
| `min_subword_len` | $`3`$ | shortest character n-gram |
| `max_subword_len` | $`6`$ | longest character n-gram |
| `subsample_threshold` | $`10^{-4}`$ | frequent-word down-sampling threshold |
| `batch_size` | $`10\,000`$ | prefetch/streaming batch size |

## References

1. T. Mikolov, I. Sutskever, K. Chen, G. Corrado & J. Dean (2013). *Distributed representations of
   words and phrases and their compositionality* (word2vec / skip-gram with negative sampling).
   NeurIPS 26. [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
2. P. Bojanowski, E. Grave, A. Joulin & T. Mikolov (2017). *Enriching word vectors with subword
   information* (FastText). TACL 5, 135–146.
   [doi:10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)
3. R. Sennrich, B. Haddow & A. Birch (2016). *Neural machine translation of rare words with subword
   units* (BPE). ACL 2016, 1715–1725.
   [doi:10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)

## See also

- [BPE & Subword Extraction](bpe.md) — `extract_subwords`, `hash_subword`, and the BPE tokenizer
- [Skip-gram Training](skip-gram.md) — how $`E_{\mathrm{word}}`$ and $`E_{\mathrm{sub}}`$ are learned
- [Similarity](similarity.md) — cosine, `most_similar`, and analogies
- [Phonetic Embeddings](phonetic.md) — sound-aware similarity on top of this model
- [Hybrid Interpolation](../hybrid/interpolation.md) — fusing this model with the n-gram expert
- [Embedding API reference](../../api/embedding.md) — the full method surface
