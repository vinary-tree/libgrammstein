# Modified Kneser-Ney Smoothing

**Modified Kneser-Ney (MKN)** is the smoothing algorithm at the statistical core of
libgrammstein. It is the most accurate count-based smoother known for n-gram language models
[[1]](#references)[[2]](#references), and it is the crate's always-on default. This document
explains *what* problem it solves, *how* the mathematics works, and *how libgrammstein
implements it* — including the places where the shipped code deliberately simplifies the
textbook formula.

> **Scope.** Source of truth: [`src/ngram/smoothing/kneser_ney.rs`](../../../src/ngram/smoothing/kneser_ney.rs)
> and [`src/ngram/entry.rs`](../../../src/ngram/entry.rs). For the surrounding query surface see
> [Query API](query-api.md); for storage see [Trie Storage](trie-storage.md).

## Notation

Every symbol below is defined before it is used in a formula.

| Symbol | Meaning |
|---|---|
| $`w`$ | the word (token) whose probability is being estimated |
| $`h`$ | the *history* (context) — the words preceding $`w`$ |
| $`h'`$ | the *backed-off* history — $`h`$ with its **oldest** (leftmost) word removed |
| $`c(h\,w)`$ | raw training count of the n-gram formed by appending $`w`$ to $`h`$ |
| $`c(h)`$ | raw training count of the context $`h`$ |
| $`D(c)`$ | the absolute discount applied to an n-gram of count $`c`$ |
| $`D_1, D_2, D_{3+}`$ | the three MKN discounts, for counts $`1`$, $`2`$, and $`\geq 3`$ |
| $`\gamma(h)`$ | the backoff weight for history $`h`$ (mass redistributed to lower orders) |
| $`N_{1+}(\bullet, w)`$ | *continuation count* — number of **distinct** contexts in which $`w`$ appears |
| $`N_{1+}(h, \bullet)`$ | number of **distinct** words that follow $`h`$ |
| $`N_{1+}(\bullet, \bullet)`$ | total number of distinct bigram types, $`\sum_w N_{1+}(\bullet, w)`$ |
| $`\lvert V \rvert`$ | vocabulary size |
| $`[x]^{+}`$ | $`\max(x, 0)`$ |

**Acronyms.** *MLE* — Maximum-Likelihood Estimate; *MKN* — Modified Kneser-Ney; *OOV* —
Out-Of-Vocabulary.

## The problem: unseen n-grams

An n-gram model estimates $`\mathbb{P}(w \mid h)`$, the probability of the next word given a
history. The naïve **Maximum-Likelihood Estimate** simply divides counts:

```math
\begin{array}{lr}
\displaystyle \mathbb{P}_{\mathrm{MLE}}(w \mid h) = \frac{c(h\,w)}{c(h)} & \text{(M1)}
\end{array}
```

$`(\mathrm{M1})`$ assigns **zero** probability to any n-gram never seen in training. Because a
sentence's log-probability is a *sum* of per-token log-probabilities, a single unseen n-gram
sends $`\log \mathbb{P} \to -\infty`$ for the whole sentence. If *"the quick purple"* never
occurred, then $`\mathbb{P}_{\mathrm{MLE}}(\text{purple} \mid \text{the quick}) = 0`$ and the
sentence is deemed impossible. *Smoothing* repairs this by stealing a little probability mass
from seen events and redistributing it to unseen ones. Two questions define a smoother:

1. **How much** mass to steal? — MKN answers with *absolute discounting*.
2. **How to redistribute** it? — MKN answers with *continuation counts* and *backoff*.

## Theory

### Absolute discounting with three discounts

Kneser-Ney subtracts a fixed **absolute discount** $`D`$ from every non-zero count before
normalizing, reserving the removed mass for a lower-order *backoff* distribution. The *Modified*
variant of Chen & Goodman [[2]](#references) observed that the optimal discount depends on the
count, and so uses **three** discounts — one for singletons, one for count-2 n-grams, and one
for the rest:

```math
\begin{array}{lr}
\displaystyle D(c) = \begin{cases}
0 & c = 0 \\
D_1 & c = 1 \\
D_2 & c = 2 \\
D_{3+} & c \geq 3
\end{cases} & \text{(M2a)}
\end{array}
```

The discounts are estimated from the corpus's *count-of-counts* — $`n_i`$ is the number of
n-grams occurring exactly $`i`$ times:

```math
\begin{array}{lr}
\displaystyle Y = \frac{n_1}{n_1 + 2\,n_2}, \qquad
D_1 = 1 - 2Y\frac{n_2}{n_1}, \qquad
D_2 = 2 - 3Y\frac{n_3}{n_2}, \qquad
D_{3+} = 3 - 4Y\frac{n_4}{n_3} & \text{(M2b)}
\end{array}
```

libgrammstein clamps each discount to its natural range ($`D_1 \in [0,1]`$, $`D_2 \in [0,2]`$,
$`D_{3+} \in [0,3]`$) for numerical safety, and when count statistics are unavailable it falls
back to fixed defaults $`D_1 = 0.75,\ D_2 = 0.85,\ D_{3+} = 0.95`$
(`KneserNeySmoothing::default_discounts`).

### The interpolated recursion

At the highest order, MKN interpolates a discounted higher-order estimate with a recursive
lower-order term:

```math
\begin{array}{lr}
\displaystyle \mathbb{P}_{\mathrm{MKN}}(w \mid h) =
\frac{\bigl[\,c(h\,w) - D(c(h\,w))\,\bigr]^{+}}{c(h)}
\;+\; \gamma(h)\,\mathbb{P}_{\mathrm{MKN}}(w \mid h') & \text{(M3)}
\end{array}
```

The backoff weight $`\gamma(h)`$ is chosen to be exactly the mass removed by discounting, which
guarantees the estimate is a proper distribution — $`\sum_w \mathbb{P}_{\mathrm{MKN}}(w \mid h) = 1`$
at every order. In canonical MKN it aggregates the three discounts over the counts of the
words following $`h`$:

```math
\begin{array}{lr}
\displaystyle \gamma(h) = \frac{D_1\,N_1(h) + D_2\,N_2(h) + D_{3+}\,N_{3+}(h)}{c(h)} & \text{(M4)}
\end{array}
```

where $`N_i(h)`$ is the number of distinct words that follow $`h`$ exactly $`i`$ times.

### Continuation counts: measuring versatility

The lower-order terms do **not** use raw counts. They use **continuation counts** — *how many
distinct contexts a word completes* — which measure a word's *versatility* rather than its raw
frequency:

```math
\begin{array}{lr}
\displaystyle \mathbb{P}_{\mathrm{cont}}(w) = \frac{N_{1+}(\bullet, w)}{N_{1+}(\bullet, \bullet)},
\qquad N_{1+}(\bullet, w) = \bigl\lvert \{\, v : c(v\,w) > 0 \,\} \bigr\rvert & \text{(M5)}
\end{array}
```

This is the **"San Francisco" intuition**: the word *Francisco* is frequent, but it follows
essentially only *San*, so its continuation count is $`\approx 1`$ and it earns almost no
lower-order mass. *City* follows many words, so it earns a high fallback probability.

| Word | Raw count | Continuation count $`N_{1+}(\bullet, w)`$ |
|---|---|---|
| Francisco | high | $`\approx 1`$ (only follows *San*) |
| city | moderate | large (follows many words) |

Under raw counts, *Francisco* would wrongly outrank *city* as a fallback; under continuation
counts, *city* correctly wins.

![Modified Kneser-Ney backoff recursion](../../diagrams/mkn-backoff.svg)

## The algorithm, literately

Scoring is a single recursion from the longest matching context down to a unigram base case.
The following mirrors [`KneserNeySmoothing::prob_recursive`](../../../src/ngram/smoothing/kneser_ney.rs);
`⟨…⟩` names a refinement expanded below.

```
function MKN_log_prob(w, h):                          ▸ public entry point; returns a log-probability
    return ln( prob(w, h, is_highest_order = true) )

function prob(w, h, is_highest_order):
    if h is empty:                                    ▸ recursion bottoms out at the unigram
        return ⟨Unigram base case⟩
    c_hw <- count(h ++ w)                             ▸ raw count of the full n-gram
    c_h  <- count(h)                                  ▸ raw count of the context
    if c_h == 0:                                      ▸ context unseen at this order ...
        return prob(w, h', is_highest_order = false)  ▸ ... so back off: drop the oldest word of h
    ⟨Discounted higher-order term⟩
    ⟨Backoff weight lambda(h)⟩
    p_low <- prob(w, h', is_highest_order = false)    ▸ recursive lower-order probability
    p <- p_high + lambda * p_low
    return p if p > 0 else p_low                      ▸ finite-log guard (see Engineering)

⟨Discounted higher-order term⟩ ≡
    D      <- discount(c_hw)                          ▸ D1 / D2 / D3+  per (M2a)
    p_high <- max(c_hw - D, 0) / c_h                  ▸ the discounted mass, per (M3)

⟨Backoff weight lambda(h)⟩ ≡                          ▸ single-discount form; see Engineering
    D_lam <- d1 if c_hw == 0 else discount(c_hw)      ▸ reserve mass even for an unseen target
    Nh    <- max(unique_continuations(h), 1)          ▸ N1+(h, .) = distinct words following h
    lambda <- D_lam * Nh / c_h

⟨Unigram base case⟩ ≡
    if is_highest_order:                              ▸ top-level unigram uses raw frequency
        cnt <- count(w)
        return  1 / |V|  if cnt == 0  else  cnt / total_count
    else:                                             ▸ backed-off unigram uses continuation prob
        cc <- continuation_count(w)                   ▸ N1+(., w) = distinct preceding contexts
        return  1 / |V|  if cc == 0  else  cc / total_bigram_types   ▸ (M5); OOV backs off to 1/|V|
```

**Backoff drops the oldest word.** $`h' = h[1{:}]`$ — the recursion peels the *leftmost*
(oldest) word, so a 5-gram context shrinks 5→4→3→2→1.

## Engineering

### `NgramEntry`: lock-free atomic statistics

Each n-gram stores three statistics, held as **atomics** so parallel corpus workers can update
them without locks (see [Threading Model](../../architecture/threading.md)). Fields are private;
access is via methods.

```rust
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct NgramEntry {
    count: AtomicU64,                 // raw corpus count  c(h·w)
    continuation_count: AtomicU32,    // N1+(., ngram): distinct preceding contexts
    unique_continuations: AtomicU32,  // N1+(ngram, .): distinct following words
}

impl NgramEntry {
    pub fn count(&self) -> u64 { self.count.load(Ordering::Relaxed) }
    pub fn continuation_count(&self) -> u32 { self.continuation_count.load(Ordering::Relaxed) }
    pub fn unique_continuations(&self) -> u32 { self.unique_continuations.load(Ordering::Relaxed) }
    pub fn increment(&self) { self.count.fetch_add(1, Ordering::Relaxed); }
    // …increment_continuation / increment_unique_continuations / set_* elsewhere
}
```

A `Clone`/`Copy` [`NgramEntrySnapshot`](../../../src/ngram/entry.rs) with plain (non-atomic)
fields is provided for crossing thread boundaries and for serialization.

### The implementation's backoff weight is the single-discount form

The shipped code does **not** evaluate the three-count $`\gamma(h)`$ of $`(\mathrm{M4})`$.
Instead it uses the *single-discount* interpolation weight

```math
\begin{array}{lr}
\displaystyle \lambda(h) = \frac{D \cdot N_{1+}(h, \bullet)}{c(h)} & \text{(M4$'$)}
\end{array}
```

where $`N_{1+}(h, \bullet)`$ is `unique_continuations` and $`D`$ is the per-count discount of
the queried n-gram. This is the classic Kneser-Ney backoff weight; it is cheaper (one field,
one multiply) and empirically robust, at the cost of the finer per-count aggregation. **Two
guards keep the log-probability finite:**

1. **Unseen-target mass.** If the queried n-gram is unseen ($`c(h\,w) = 0`$), then
   $`D(0) = 0`$ would zero *both* the discounted term and $`\lambda`$, giving probability $`0`$
   and $`\log \mathbb{P} = -\infty`$. The code substitutes $`D_1`$ for $`D`$ in
   $`(\mathrm{M4}')`$ in this case, so mass is still reserved for the always-positive backoff.
2. **Degenerate-zero fallback.** If the interpolated $`p = p_{\text{high}} + \lambda\,p_{\text{low}}`$
   still evaluates to $`\leq 0`$, the code returns the strictly-positive backoff $`p_{\text{low}}`$.

The lower-order continuation denominator $`N_{1+}(\bullet, \bullet)`$ (`total_bigram_types`) is
computed once at training time. Models serialized before this statistic was tracked store $`0`$;
the code then falls back to $`\lvert V \rvert`$ as the denominator, so old models still load.

### Log-space, and why probabilities never underflow

Sentence scoring sums per-token log-probabilities (never multiplies raw probabilities), so long
sequences cannot underflow. Every path through the recursion terminates at the unigram base
case, whose OOV branch returns the strictly-positive $`1 / \lvert V \rvert`$. Hence
$`\mathbb{P}_{\mathrm{MKN}}(w \mid h) > 0`$ always, and $`\log \mathbb{P}`$ is always finite.

### Complexity

A single $`\mathbb{P}_{\mathrm{MKN}}(w \mid h)`$ query performs at most $`n`$ trie look-ups (one
per backoff level) for an order-$`n`$ model — $`O(n)`$ look-ups, each linear in the key length.
In practice this is $`\approx 100`$ ns for a 5-gram model over a varint-indexed trie.

## Usage

```rust
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::corpus::PlaintextReader;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

// Train a 5-gram Modified Kneser-Ney model over a serializable trie backend.
let reader = PlaintextReader::from_file("corpus.txt")?;
let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(5)
    .train(reader)?;

// log P(fox | quick brown) under Modified Kneser-Ney.
let log_p = model.log_prob("fox", &["quick", "brown"]);
println!("log P = {log_p:.3}");
# Ok::<(), libgrammstein::Error>(())
```

The discounts themselves can be inspected or constructed directly:

```rust
use libgrammstein::ngram::smoothing::KneserNeySmoothing;

// From corpus count-of-counts (n1, n2, n3, n4):
let kn = KneserNeySmoothing::from_counts(1_000_000, 250_000, 100_000, 50_000);
// …or the fixed fallback discounts D1=0.75, D2=0.85, D3+=0.95:
let default_kn = KneserNeySmoothing::default_discounts();
```

## Comparison with other smoothers

| Method | Idea | Trade-off |
|---|---|---|
| Add-$`k`$ (Laplace) | add a constant to every count | simple; systematically over-smooths |
| Good-Turing | re-estimate mass from count-of-counts | theoretically motivated; fiddly to implement |
| Witten-Bell | discount by observed novelty rate | intuitive; weaker than Kneser-Ney |
| Kneser-Ney | absolute discount + continuation counts | strong; single discount |
| **Modified Kneser-Ney** | three count-dependent discounts | **best empirical accuracy**; most parameters |

## References

1. R. Kneser & H. Ney (1995). *Improved backing-off for M-gram language modeling.* ICASSP '95,
   181–184. [doi:10.1109/ICASSP.1995.479394](https://doi.org/10.1109/ICASSP.1995.479394)
2. S. F. Chen & J. Goodman (1999). *An empirical study of smoothing techniques for language
   modeling.* Computer Speech & Language 13(4), 359–393.
   [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)

## See also

- [N-gram Overview](overview.md) — higher-level concepts and the query surface
- [Trie Storage](trie-storage.md) — how n-grams and vocabulary are stored
- [Query API](query-api.md) — the probability-query interface
- [Hybrid Interpolation](../hybrid/interpolation.md) — combining this model with embeddings
