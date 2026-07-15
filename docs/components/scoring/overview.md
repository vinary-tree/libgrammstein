# Scoring: Perplexity and Sentence Ranking

The **scoring** module is libgrammstein's evaluation surface: it answers *"how good is this
model?"* (perplexity over a held-out corpus) and *"which of these sentences is most likely?"*
(sentence ranking). It computes no probabilities of its own — every number it reports comes
from the n-gram model's Modified Kneser-Ney query path — so what it measures is exactly the
model you ship. This document explains the information-theoretic basis of perplexity, the
length-bias problem that sentence scoring must confront, and how the shipped code realizes
both.

> **Scope.** Source of truth: [`src/scoring/mod.rs`](../../../src/scoring/mod.rs),
> [`src/scoring/perplexity.rs`](../../../src/scoring/perplexity.rs), and
> [`src/scoring/sentence.rs`](../../../src/scoring/sentence.rs). For the exact signatures see
> the [Scoring API reference](../../api/scoring.md); for the probabilities being summed see
> [Modified Kneser-Ney](../ngram/modified-kneser-ney.md).

## Notation

Every symbol is defined before it is used.

| Symbol | Meaning |
|---|---|
| $`W = w_1 w_2 \ldots w_N`$ | the token sequence being scored (a sentence, or a whole corpus) |
| $`N`$ | the number of tokens in $`W`$ |
| $`h_i`$ | the *history* (context) of token $`w_i`$ — the preceding words available to the model |
| $`n`$ | the model's n-gram order, so $`\lvert h_i \rvert \leq n - 1`$ |
| $`\mathbb{P}(w_i \mid h_i)`$ | the model's probability of $`w_i`$ given $`h_i`$ (here, Modified Kneser-Ney) |
| $`\lvert V \rvert`$ | vocabulary size (number of distinct unigrams seen in training) |
| $`H(W)`$ | the model's per-token **cross-entropy** on $`W`$, in nats |
| $`\mathrm{PP}(W)`$ | the **perplexity** of the model on $`W`$ |
| $`\log`$ | the natural logarithm (base $`e`$); libgrammstein works in **nats** throughout |

**Acronyms.** *PP* — Perplexity; *MKN* — Modified Kneser-Ney; *OOV* — Out-Of-Vocabulary;
*LM* — Language Model.

## What perplexity measures

A language model is a probability distribution over token sequences. The natural way to grade
one is to ask how much probability mass it assigns to text it has never seen: a good model is
*unsurprised* by real language. **Cross-entropy** makes "surprise" precise — it is the average
number of nats needed to encode each token under the model [[1]](#references):

```math
H(W) = -\frac{1}{N}\sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i) \tag{S1}
```

**Perplexity** is cross-entropy exponentiated back out of log space [[2]](#references):

```math
\mathrm{PP}(W) \;=\; \exp\bigl(H(W)\bigr) \;=\; \exp\!\Bigl(-\frac{1}{N}\sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i)\Bigr) \tag{S2}
```

Equivalently — and this is the most useful way to *read* it — perplexity is the geometric mean
of the model's inverse per-token probabilities:

```math
\mathrm{PP}(W) \;=\; \Bigl(\prod_{i=1}^{N} \frac{1}{\mathbb{P}(w_i \mid h_i)}\Bigr)^{1/N} \tag{S3}
```

### The branching-factor intuition

$`(\mathrm{S3})`$ says: *if the model were forced to guess each next token uniformly among
$`k`$ equally-likely candidates, $`k`$ is the perplexity.* Perplexity is the model's **average
branching factor** — the effective size of the choice it faces at each position.

Two anchors make the scale concrete. Suppose the model were uniform, $`\mathbb{P}(w_i \mid h_i) = 1/\lvert V \rvert`$
for every token. Substituting into $`(\mathrm{S3})`$ gives

```math
\mathrm{PP}(W) = \Bigl(\prod_{i=1}^{N} \lvert V \rvert\Bigr)^{1/N} = \lvert V \rvert \tag{S4}
```

so **uniform guessing scores exactly $`\lvert V \rvert`$** — the worst any sane model should do.
At the other extreme a model that is certain and correct at every position has
$`\mathbb{P} = 1`$ everywhere and scores $`\mathrm{PP} = 1`$. Hence $`1 \leq \mathrm{PP}(W) \leq \lvert V \rvert`$
for any model that never assigns less than uniform mass, and **lower is better**.

> **Base matters, and libgrammstein uses nats.** $`(\mathrm{S1})`$–$`(\mathrm{S3})`$ use the
> natural logarithm, matching `NgramModel::log_prob`, which returns $`\log_e`$. Literature that
> quotes cross-entropy *in bits* uses $`H_2 = H / \log 2`$; perplexity itself is
> **base-independent**, because $`\exp(H) = 2^{H_2}`$. So a perplexity number is directly
> comparable with a published one, while a raw cross-entropy number is not until you fix the
> base.

### Why probabilities never underflow

$`(\mathrm{S3})`$ multiplies $`N`$ probabilities; for a corpus of millions of tokens that
product underflows any float. libgrammstein therefore never forms it: it accumulates the
**sum of logs** in $`(\mathrm{S1})`$ and exponentiates once, at the end. This is safe because
Modified Kneser-Ney guarantees $`\mathbb{P}(w \mid h) > 0`$ for every query — every backoff path
terminates in the strictly-positive uniform base case $`1/\lvert V \rvert`$ — so every
$`\log \mathbb{P}`$ is finite (see [Modified Kneser-Ney](../ngram/modified-kneser-ney.md),
*Log-space, and why probabilities never underflow*).

## Corpus perplexity

`Perplexity::corpus_perplexity` streams a held-out corpus, scores every token in its sentence
context, and reduces the stream to a single `PerplexityResult`.

![Corpus perplexity evaluation pipeline](../../diagrams/scoring-perplexity.svg)

The context of token $`i`$ is the window of up to $`n - 1`$ preceding tokens **within the same
sentence**, truncated at the sentence start:

```math
h_i = w_{\max(1,\; i-n+1)} \ldots w_{i-1} \tag{S5}
```

Sentences are scored independently — no context crosses a sentence boundary — which is why
`total_log_prob` is a sum of per-sentence sums, and why `total_tokens` is the sum of sentence
lengths. The reported perplexity is $`(\mathrm{S2})`$ evaluated over *all* tokens of *all*
sentences at once (a token-weighted, not sentence-weighted, average).

### The OOV rate, and what it actually detects

`PerplexityResult` also reports an OOV count. The model has no explicit `<unk>` token, so
"OOV" is *detected*, not looked up: a token is counted as out-of-vocabulary when its
log-probability sinks to the uniform floor that MKN reserves for unknown words,

```math
\log \mathbb{P}(w_i \mid h_i) \;\leq\; \texttt{oov_log_prob} \;=\; -\log \lvert V \rvert \tag{S6}
```

which by $`(\mathrm{S4})`$ is exactly "this token did no better than uniform guessing".

> **Honest caveat.** $`(\mathrm{S6})`$ is a *sufficient* signal, not an exact one. It fires for
> every genuinely unseen word (whose unigram backoff returns exactly $`1/\lvert V \rvert`$), but
> it will *also* fire for a known-but-desperately-improbable token whose smoothed probability
> happens to fall at or below the floor. Read `oov_rate` as "fraction of tokens the model could
> not do better than guess on" — which is the quantity you actually care about when deciding
> whether a corpus is in-domain — rather than as a exact vocabulary-membership tally. For an
> exact tally, test `NgramModel::in_vocabulary` per token.

## Sentence scoring, and the length-bias trap

`SentenceScorer` exposes one sentence's probability three ways. The distinction is not
cosmetic — choosing wrong silently corrupts a ranking.

![Sentence scoring and length normalization](../../diagrams/scoring-sentence.svg)

The raw score is the **sum** of $`(\mathrm{S1})`$'s summands, un-normalized:

```math
\ell(W) \;=\; \sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i) \tag{S7}
```

Because every $`\log \mathbb{P}(w_i \mid h_i) < 0`$, appending a word can only *decrease*
$`\ell`$. So $`\ell`$ is **length-biased**: across candidates of unequal length it
systematically prefers the shortest, regardless of fluency. Dividing the bias out gives the
per-token score and its exponential:

```math
\bar{\ell}(W) = \frac{\ell(W)}{N},
\qquad
\mathrm{PP}(W) = \exp\bigl(-\bar{\ell}(W)\bigr) \tag{S8}
```

$`\bar{\ell}`$ (`normalized_log_prob`) and $`\mathrm{PP}`$ (`perplexity`) are length-invariant
and are the right scores to compare unequal-length candidates with.

> **API gotcha.** `rank_sentences` and `best_sentence` rank by the **raw** $`\ell`$ of
> $`(\mathrm{S7})`$, not by $`\bar{\ell}`$. That is the correct default for the common case —
> ranking equal-length rewrites of one sentence, e.g. competing spelling corrections, where the
> $`1/N`$ factor is a shared constant and cannot change the order. When your candidates differ
> in length, do **not** use those helpers directly: map `normalized_log_prob` (or `perplexity`)
> over the candidates and sort on that instead. The [Usage](#usage) section shows both.

## The algorithm, literately

The following mirrors [`Perplexity::corpus_perplexity`](../../../src/scoring/perplexity.rs) and
[`Perplexity::sentence_log_prob_with_oov`](../../../src/scoring/perplexity.rs); `⟨…⟩` names a
refinement expanded below.

```
function corpus_perplexity(reader):                    ▸ returns Result<PerplexityResult>
    sum_log_p  <- 0.0                                  ▸ accumulates (S1)'s numerator, in nats
    n_tokens   <- 0
    n_oov      <- 0
    n_sentence <- 0

    for sentence in reader.sentences():                ▸ streaming; the corpus is never buffered
        tokens <- sentence.split_whitespace()          ▸ NOTE: not the training Tokenizer (see Engineering)
        if tokens is empty: continue                   ▸ blank lines contribute nothing

        (log_p, oov) <- ⟨Score one sentence⟩
        sum_log_p  <- sum_log_p + log_p
        n_tokens   <- n_tokens + len(tokens)
        n_oov      <- n_oov + oov
        n_sentence <- n_sentence + 1

    if n_tokens == 0:                                  ▸ nothing was scorable ...
        return Err(EmptyCorpus)                        ▸ ... so PP is undefined, not infinite

    avg_log_p  <- sum_log_p / n_tokens                 ▸ = -H(W), per (S1)
    perplexity <- exp(-avg_log_p)                      ▸ = PP(W),  per (S2)
    return Ok(PerplexityResult{ perplexity, sum_log_p, n_tokens,
                                n_oov, n_oov / n_tokens, n_sentence })

⟨Score one sentence⟩ ≡
    log_p <- 0.0
    oov   <- 0
    n     <- model.order()
    for i in 0 .. len(tokens):
        context <- tokens[max(0, i - (n-1)) .. i]      ▸ the window h_i of (S5)
        p       <- model.log_prob(tokens[i], context)  ▸ the ONLY probability computation
        if p <= model.oov_log_prob():                  ▸ the floor test of (S6)
            oov <- oov + 1
        log_p <- log_p + p                             ▸ sum of logs, never a product
    return (log_p, oov)
```

`SentenceScorer` is the same accumulation exposed without the corpus loop: `log_prob` is
$`(\mathrm{S7})`$ (delegated straight to `NgramModel::sentence_log_prob`),
`normalized_log_prob` and `perplexity` are $`(\mathrm{S8})`$, and `rank_sentences` /
`best_sentence` sort / maximize over $`\ell`$.

## Engineering

### Tokenization must match training — and by default it does not

This is the single most common way to get a misleading perplexity number, and it is worth
stating plainly.

| Stage | Tokenizer used | Lowercases | Strips punctuation |
|---|---|---|---|
| **Training** (`NgramTrainer::count_ngrams`) | `Tokenizer::words`, splitting on `[\s\p{P}]+` | yes | **yes** |
| **Evaluation** (`Perplexity::corpus_perplexity`) | `str::split_whitespace` | (inherited from the reader) | **no** |

A `PlaintextReader` lowercases as it emits sentences (its `Tokenizer` defaults to
`lowercase: true`), so case is usually consistent. **Punctuation is not.** Training sees the
type `quick`; whitespace-splitting an evaluation sentence yields the *distinct* type `quick,`,
which the model has never observed. Every punctuation-adjacent token is then charged the OOV
floor, inflating both `oov_rate` and `perplexity` — and the inflation is a property of the
*harness*, not of the model.

Two ways to keep the comparison honest:

1. **Pre-normalize the evaluation corpus** so that whitespace-splitting reproduces the training
   tokenization (strip punctuation before writing the file, e.g. with `TextPreprocessor`), or
2. **Bypass the corpus loop**: tokenize each sentence yourself with the *same* `Tokenizer` the
   trainer used and call `Perplexity::sentence_log_prob` (or `SentenceScorer::perplexity`) on
   the resulting tokens, accumulating $`(\mathrm{S1})`$ yourself.

Perplexity is only ever comparable between models that share a vocabulary *and* a tokenization;
this is an instance of that general rule [[3]](#references).

### The `MutableMappedDictionary` bound is stronger than the code needs

Both `Perplexity<'a, D>` and `SentenceScorer<'a, D>` are declared over
`D: MutableMappedDictionary<Value = NgramEntry>`, yet neither ever writes to the dictionary —
they only call `log_prob`, `sentence_log_prob`, `order`, and `oov_log_prob`, all of which live
on `NgramModel`'s read-only `MappedDictionary` impl. The practical consequence is that the
**static, read-only `DoubleArrayTrieChar` backend cannot be scored by this module**: it
implements `MappedDictionary` (and so backs `NgramModel` and answers `log_prob` perfectly well)
but deliberately does *not* implement `MutableMappedDictionary`, because a Double-Array Trie is
bulk-built and immutable. Scoring a static model therefore means calling
`NgramModel::sentence_log_prob` directly and applying $`(\mathrm{S2})`$ yourself — three lines,
shown in [Usage](#usage). Relaxing the bound to `MappedDictionary` would remove the wart
without changing any behavior.

### Cost and concurrency

Scoring one token is one `log_prob` query: $`O(n)`$ trie look-ups for an order-$`n`$ model (one
per backoff level), each linear in the key length. A corpus of $`N`$ tokens therefore costs
$`O(N \cdot n)`$ look-ups, and the streaming reader keeps memory flat in the corpus size — only
one sentence is materialized at a time.

The accumulation loop itself is **sequential**: `corpus_perplexity` folds `sum_log_p` in a
single thread. That is a deliberate simplicity trade, not a limitation of the model —
`NgramModel` is `Send + Sync` and its query path is lock-free, so a caller who needs throughput
can shard the corpus, score each shard on its own thread, and combine the partial sums.
$`(\mathrm{S1})`$ is a plain sum, so the reduction is exact and order-independent up to
floating-point associativity.

### Scoring a hybrid model

The scoring module is n-gram-only: its type parameter is `NgramModel<D>`. `HybridLanguageModel`
carries its **own** `sentence_log_prob` and `perplexity` (identical formulae $`(\mathrm{S7})`$
and $`(\mathrm{S8})`$, but summing the *interpolated* score rather than the pure MKN one), so a
hybrid model is evaluated through those methods directly — see the
[Hybrid API reference](../../api/hybrid.md). There is no corpus-level perplexity helper for
hybrid models; the loop in [Usage](#usage) is the whole of it.

## Usage

Corpus perplexity over a held-out development set:

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::scoring::Perplexity;
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(5)
    .train(PlaintextReader::from_file("train.txt")?)?;

let dev = PlaintextReader::from_file("dev.txt")?;
let result = Perplexity::new(&model).corpus_perplexity(&dev)?;

// PerplexityResult implements Display:
//   "Perplexity: 142.87 | Tokens: 51234 | OOV: 3.12% | Sentences: 2410"
println!("{result}");
println!("PP = {:.2}, OOV rate = {:.2}%", result.perplexity, result.oov_rate * 100.0);
# Ok::<(), libgrammstein::Error>(())
```

Ranking candidates — the equal-length case (safe) and the unequal-length case (normalize):

```rust
use libgrammstein::scoring::SentenceScorer;

let scorer = SentenceScorer::new(&model);

// Equal length: rank_sentences is exactly right — 1/N is a shared constant.
let a: &[&str] = &["the", "quick", "brown", "fox"];
let b: &[&str] = &["the", "quick", "brown", "fax"];
if let Some((best, score)) = scorer.best_sentence(&[a, b]) {
    println!("best = {best:?} (log P = {score:.3})");
}

// Unequal length: rank on the length-invariant score instead.
let short: &[&str] = &["the", "fox"];
let long: &[&str] = &["the", "quick", "brown", "fox", "jumped"];
let mut ranked: Vec<(&[&str], f64)> = [short, long]
    .into_iter()
    .map(|s| (s, scorer.normalized_log_prob(s)))
    .collect();
ranked.sort_by(|x, y| y.1.total_cmp(&x.1));   // descending: least surprising first
```

Scoring a **static** (`DoubleArrayTrieChar`-backed) model, which the module's bound excludes —
apply $`(\mathrm{S2})`$ directly:

```rust
// `static_model: NgramModel<DoubleArrayTrieChar<NgramEntry>>`, loaded via load_static_portable.
let tokens = ["the", "quick", "brown", "fox"];
let log_prob = static_model.sentence_log_prob(&tokens);          // (S7)
let perplexity = (-log_prob / tokens.len() as f64).exp();        // (S2)
println!("PP = {perplexity:.2}");
```

## Interpreting the number

| Observation | Reading |
|---|---|
| $`\mathrm{PP} \approx \lvert V \rvert`$ | the model is barely better than uniform — check tokenization and training |
| $`\mathrm{PP}`$ far lower on train than on dev | over-fitting; the model memorized the training corpus |
| $`\mathrm{PP}`$ rises with `oov_rate` | the dev set is out-of-domain (or tokenized differently) |
| $`\mathrm{PP}`$ drops as `order` rises, then flattens | the corpus has exhausted its usable context depth |

Two comparisons are **invalid** and worth guarding against: perplexities computed over different
vocabularies (a smaller $`\lvert V \rvert`$ mechanically lowers $`\mathrm{PP}`$ by
$`(\mathrm{S4})`$), and perplexities computed under different tokenizations. And even a valid
comparison is only a proxy: perplexity correlates with, but does not determine, downstream task
quality — the correlation with word-error rate is real yet loose, and a model with lower
perplexity can lose on the task you actually care about [[4]](#references). Treat perplexity as
the fast inner-loop signal, and the end task as the arbiter.

## References

1. C. E. Shannon (1948). *A mathematical theory of communication.* Bell System Technical
   Journal 27(3), 379–423.
   [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
2. F. Jelinek, R. L. Mercer, L. R. Bahl & J. K. Baker (1977). *Perplexity — a measure of the
   difficulty of speech recognition tasks.* Journal of the Acoustical Society of America
   62(S1), S63. [doi:10.1121/1.2016299](https://doi.org/10.1121/1.2016299)
3. P. F. Brown, S. A. Della Pietra, V. J. Della Pietra, J. C. Lai & R. L. Mercer (1992).
   *An estimate of an upper bound for the entropy of English.* Computational Linguistics 18(1),
   31–40. [ACL Anthology J92-1002](https://aclanthology.org/J92-1002/)
4. D. Klakow & J. Peters (2002). *Testing the correlation of word error rate and perplexity.*
   Speech Communication 38(1–2), 19–28.
   [doi:10.1016/S0167-6393(01)00041-3](https://doi.org/10.1016/S0167-6393%2801%2900041-3)
5. S. F. Chen & J. Goodman (1999). *An empirical study of smoothing techniques for language
   modeling.* Computer Speech & Language 13(4), 359–393.
   [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)

## See also

- [Scoring API reference](../../api/scoring.md) — the exact `Perplexity` / `SentenceScorer` signatures
- [Modified Kneser-Ney](../ngram/modified-kneser-ney.md) — the probabilities being summed
- [N-gram Query API](../ngram/query-api.md) — `log_prob`, `sentence_log_prob`, `oov_log_prob`
- [Hybrid Interpolation](../hybrid/interpolation.md) — scoring an interpolated model instead
- [Text Generation](../generation/text-generation.md) — the other consumer of `log_prob`
- [Perplexity Scoring example](../../examples/perplexity-scoring.md) — end-to-end quality filtering
