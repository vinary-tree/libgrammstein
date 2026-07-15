# Example: Perplexity Scoring

**Perplexity** is the number libgrammstein uses to answer *"is this model any good?"* and, turned
around, *"is this text any good?"*. This page derives it, shows the exact API that computes it
(`Perplexity` / `PerplexityResult`), explains how out-of-vocabulary tokens are accounted for,
sweeps the n-gram order, compares an n-gram against a hybrid, and finally uses per-sentence
perplexity as a **corpus quality filter**.

> **Scope.** Source of truth: [`src/scoring/perplexity.rs`](../../src/scoring/perplexity.rs),
> with [`src/ngram/model.rs`](../../src/ngram/model.rs) (`log_prob`, `oov_log_prob`) and
> [`src/hybrid/model.rs`](../../src/hybrid/model.rs) (`perplexity`). The training half of the
> story is [Train and Evaluate](train-and-evaluate.md); the smoothing that keeps the sum finite
> is [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md).

## Notation

| Symbol | Meaning |
|---|---|
| $`W`$ | the evaluation corpus as a token sequence $`w_1 w_2 \ldots w_N`$ |
| $`N`$ | the total number of tokens in $`W`$ |
| $`w_i`$ | the $`i`$-th token |
| $`h_i`$ | the history of $`w_i`$ — the at most $`n-1`$ tokens preceding it in its sentence |
| $`n`$ | the model's n-gram order |
| $`\mathbb{P}(w_i \mid h_i)`$ | the probability the model assigns to $`w_i`$ in context |
| $`\lvert V \rvert`$ | vocabulary size — distinct unigrams seen in training |
| $`H(W)`$ | the per-token cross-entropy of the model on $`W`$, in **nats** |
| $`\mathrm{PP}(W)`$ | perplexity |
| $`\tau`$ | a perplexity threshold used for quality filtering (§7) |

**Acronyms.** *PP* — perplexity; *OOV* — Out-Of-Vocabulary; *MKN* — Modified Kneser-Ney;
*nat* — the unit of information when logarithms are natural ($`1\ \text{nat} = \log_2 e \approx 1.443`$ bits).

## 1. What perplexity measures

A language model defines a distribution over token sequences. Its **cross-entropy** on a held-out
corpus is the average number of nats it spends encoding each token — that is, how *surprised* it
is, on average:

```math
\begin{array}{lr}
\displaystyle H(W) \;=\; -\frac{1}{N} \sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i) & \text{(P1)}
\end{array}
```

Perplexity is that entropy exponentiated back out of log space:

```math
\begin{array}{lr}
\displaystyle \mathrm{PP}(W) \;=\; \exp\bigl(H(W)\bigr)
            \;=\; \exp\!\Bigl(-\tfrac{1}{N} \sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i)\Bigr) & \text{(P2)}
\end{array}
```

Equivalently, $`\mathrm{PP}`$ is the **geometric mean of the inverse probabilities**:

```math
\begin{array}{lr}
\displaystyle \mathrm{PP}(W) \;=\; \Bigl(\prod_{i=1}^{N} \frac{1}{\mathbb{P}(w_i \mid h_i)}\Bigr)^{1/N} & \text{(P3)}
\end{array}
```

**The branching-factor intuition.** A model that spread its mass uniformly over $`k`$ words at
every step would assign $`\mathbb{P} = 1/k`$ to each token, giving $`\mathrm{PP} = k`$ exactly. So
$`\mathrm{PP} = 120`$ means: *this model is as uncertain as someone guessing uniformly among 120
words at each position.* Lower is better; $`\mathrm{PP} \geq 1`$ always, with equality only for a
model that is certain and right about every token.

libgrammstein works in **natural** logarithms throughout, so $`(\mathrm{P2})`$ uses $`\exp`$. Some
literature reports $`2^{H_2}`$ with $`H_2`$ in bits; the two differ only by the base and the
values coincide: $`\exp(H) = 2^{H / \ln 2}`$.

| Reported | Meaning |
|---|---|
| $`\mathrm{PP} \approx 1`$ | the model is essentially certain (usually a sign of evaluating on training data) |
| $`\mathrm{PP} \approx \lvert V \rvert`$ | the model is no better than uniform guessing |
| $`\mathrm{PP} > \lvert V \rvert`$ | worse than uniform — the model is *confidently wrong*, typically an OOV or tokenization mismatch |

## 2. The evaluator API

```rust
use libgrammstein::scoring::Perplexity;   // also re-exported from libgrammstein::prelude

let evaluator = Perplexity::new(&model);                       // borrows an NgramModel<D>
let result    = evaluator.corpus_perplexity(&test_reader)?;    // one streaming pass
let log_p     = evaluator.sentence_log_prob(&["the", "quick", "brown", "fox"]);
# Ok::<(), libgrammstein::Error>(())
```

`corpus_perplexity` takes any `&R where R: CorpusReader` — `PlaintextReader`, `GutenbergReader`,
`WikipediaReader` — **by reference** (unlike training, which consumes its reader). It returns a
`PerplexityResult`:

| Field | Type | Meaning |
|---|---|---|
| `perplexity` | `f64` | $`\mathrm{PP}(W)`$ from $`(\mathrm{P2})`$ |
| `total_log_prob` | `f64` | $`\sum_i \log \mathbb{P}(w_i \mid h_i)`$ — the corpus log-likelihood |
| `total_tokens` | `usize` | $`N`$ |
| `oov_count` | `usize` | tokens scored at or below the OOV floor (§3) |
| `oov_rate` | `f64` | `oov_count / total_tokens` |
| `sentence_count` | `usize` | non-empty sentences evaluated |

It also implements `Display`:

```rust
println!("{result}");
// Perplexity: 118.42 | Tokens: 15234 | OOV: 3.10% | Sentences: 812
```

> **The result type is reachable but not nameable.** `PerplexityResult` lives in a private module
> and only `Perplexity` is re-exported from `libgrammstein::scoring`. Bind the value with `let`
> and use its public fields or `Display` — do **not** try to `use` the type or write it in a
> signature. If you need to pass it around, destructure it or copy the fields you need.

An empty corpus is an error, not a $`\mathrm{PP}`$ of infinity: `corpus_perplexity` returns
`Err(Error::EmptyCorpus)` when it sees zero tokens.

## 3. The evaluation loop

![The perplexity evaluation loop. The test corpus is streamed through reader.sentences(), which lowercases and splits on terminal punctuation; each sentence is split on whitespace into tokens. For each token index i, the context h-i is the at most n-1 preceding tokens, clamped at the sentence start; model.log_prob(w-i, h-i) returns a finite negative log-probability under Modified Kneser-Ney backoff. The accumulator adds the log-probability to a running sum, increments the token count, and increments the OOV counter when the log-probability is at or below the uniform floor log of one over vocabulary size. After the last token, a single exponentiation produces the PerplexityResult: perplexity, out-of-vocabulary count and rate, token and sentence counts, and the total log-probability.](../diagrams/example-perplexity-eval.svg)

**Figure 1.** One streaming pass, accumulation in log space, exactly one $`\exp`$ at the end.
*(Rendered from `docs/diagrams/example-perplexity-eval.puml`.)*

The following mirrors `Perplexity::corpus_perplexity` and its private helper
`sentence_log_prob_with_oov`; `⟨…⟩` names a refinement expanded below.

```
function corpus_perplexity(reader):                   ▸ returns PerplexityResult
    sum_log_p <- 0 ; N <- 0 ; oov <- 0 ; sentences <- 0
    for sentence in reader.sentences():               ▸ streams; nothing is buffered
        tokens <- sentence.split_whitespace()
        if tokens is empty: continue
        (log_p, oov_here) <- ⟨score one sentence⟩
        sum_log_p <- sum_log_p + log_p
        N         <- N + |tokens|
        oov       <- oov + oov_here
        sentences <- sentences + 1
    if N == 0: return Err(EmptyCorpus)                ▸ an empty corpus is an error, not PP = ∞
    return PerplexityResult {
        perplexity     = exp(-sum_log_p / N),         ▸ (P2) — the only exp() in the whole pass
        total_log_prob = sum_log_p,
        total_tokens   = N,
        oov_count      = oov,
        oov_rate       = oov / N,
        sentence_count = sentences,
    }

⟨score one sentence⟩ ≡
    log_p <- 0 ; oov_here <- 0
    for i in 0 .. |tokens|:
        start   <- max(0, i - (order - 1))            ▸ the history never crosses a sentence boundary
        context <- tokens[start .. i]                 ▸ empty for the first token: a unigram query
        lp      <- model.log_prob(tokens[i], context) ▸ MKN backoff; always finite, always < 0
        if lp <= model.oov_log_prob():                ▸ ⟨the OOV test⟩
            oov_here <- oov_here + 1
        log_p <- log_p + lp                           ▸ SUM logs — never multiply probabilities
    return (log_p, oov_here)

⟨the OOV test⟩ ≡                                      ▸ oov_log_prob() = log(1 / |V|)
    A token is counted OOV when its log-probability has sunk to (or below) the
    uniform floor that MKN hands to a word it has never seen. This is a
    *behavioral* test, not a dictionary lookup: a known-but-hopeless word in a
    hostile context can also land on the floor. Use model.in_vocabulary(w) when
    you want strict lexical membership instead.
```

**Why the sum never underflows.** $`\mathbb{P}`$ for a 20-token sentence can be $`10^{-40}`$;
$`\log \mathbb{P}`$ is merely $`-92`$. Summing logs (never multiplying probabilities) is what lets
a million-token corpus be scored in `f64` without underflow. And because the MKN recursion
terminates in the strictly positive unigram term $`1/\lvert V \rvert`$ with non-negative
interpolation weights, every $`\mathbb{P}(w_i \mid h_i)`$ is strictly positive — no term is ever
$`-\infty`$, so $`\mathrm{PP}`$ is always finite.

**Why the OOV test is $`\leq`$ and not $`=`$.** The backoff weight $`\lambda(h) < 1`$ multiplies
the uniform term, so an unseen word's final probability is $`\lambda(h)/\lvert V \rvert`$ —
*below* the nominal floor $`1/\lvert V \rvert`$, not exactly on it. Testing
$`\log \mathbb{P} \leq \log(1/\lvert V \rvert)`$ therefore catches it, while an equality test would
catch nothing.

## 4. Choosing the order

Higher order is not automatically better: a 5-gram model has vastly more parameters, and on a
small corpus most 5-grams are singletons that smoothing must discount away. **Measure it.**

```rust
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::scoring::Perplexity;

println!("{:<8} {:>12} {:>10} {:>12}", "order", "perplexity", "OOV %", "n-grams");
for order in 2..=5 {
    let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(order)
        .train(PlaintextReader::from_file("train.txt")?)?;   // a fresh reader every time

    let report = Perplexity::new(&model).corpus_perplexity(&PlaintextReader::from_file("test.txt")?)?;

    println!(
        "{:<8} {:>12.2} {:>9.1}% {:>12}",
        format!("{order}-gram"),
        report.perplexity,
        report.oov_rate * 100.0,
        model.ngram_count()
    );
}
# Ok::<(), libgrammstein::Error>(())
```

Read the sweep like this:

- **PP falls, then flattens.** Stop at the knee; the extra orders are buying memory, not accuracy.
- **PP falls, then *rises*.** The corpus cannot support the higher order.
- **The OOV rate does not move.** It is a property of the *vocabulary*, not the order — the same
  unigrams are seen whatever $`n`$ is. If it moves, your tokenization changed (§5).
- **`ngram_count` explodes.** Storage grows roughly linearly in $`n`$ for a fixed corpus, since
  every order up to $`n`$ is stored.

## 5. The tokenization contract

Perplexity compares models only if they see the *same tokens*. libgrammstein has one asymmetry you
must know about, because it silently inflates OOV rates:

| Stage | Splitter | Punctuation | Case |
|---|---|---|---|
| **Training** (`Tokenizer::words`) | `[\s\p{P}]+` | **stripped** — `p{P}` is a separator | lowercased |
| **Evaluation** (`corpus_perplexity`) | `str::split_whitespace` | **kept** — attached to the token | lowercased upstream, by `sentences()` |

So a training corpus teaches the model the token `fox`, while a test sentence ending
`… the lazy fox.` presents the token `fox.` — a *different string*, hence OOV, hence a spuriously
high $`\mathrm{PP}`$. Three ways to make the two agree:

```rust
use libgrammstein::corpus::{CorpusReader, PlaintextReader, Tokenizer};

// (a) Pre-normalize the evaluation corpus once, offline: strip punctuation with the very
//     tokenizer that training uses, then write the result back out, one sentence per line.
let tokenizer = Tokenizer::new();
let source = PlaintextReader::from_file("test.raw.txt")?;
let cleaned: String = source
    .sentences()
    .map(|sentence| tokenizer.words(&sentence).collect::<Vec<_>>().join(" "))
    .filter(|line| !line.is_empty())
    .collect::<Vec<_>>()
    .join("\n");
std::fs::write("test.txt", cleaned)?;

// (b) …then evaluate against the cleaned file, whose whitespace split is now exact.
# Ok::<(), libgrammstein::Error>(())
```

```rust
// (c) Or bypass the corpus API entirely and drive the model with tokens you control.
let tokens: Vec<&str> = "the quick brown fox".split_whitespace().collect();
let pp = (-model.sentence_log_prob(&tokens) / tokens.len() as f64).exp();
```

**Rule of thumb.** A sudden OOV rate in the tens of percent on clean, in-domain text almost always
means a tokenization mismatch, not a vocabulary gap. Check `model.in_vocabulary("fox")` versus
`model.in_vocabulary("fox.")` before you blame the model.

## 6. N-gram versus hybrid

`Perplexity` is typed against `NgramModel`, so it cannot score a hybrid. Aggregate the hybrid's
own per-sentence `perplexity` — or, better, aggregate its per-sentence *log-probabilities* and
exponentiate once, which is $`(\mathrm{P2})`$ applied to the whole corpus rather than an average
of per-sentence perplexities (they are not the same number, and only the former is $`\mathrm{PP}(W)`$):

```rust
use libdictenstein::MappedDictionary;
use libgrammstein::corpus::CorpusReader;
use libgrammstein::hybrid::HybridLanguageModel;
use libgrammstein::ngram::NgramEntry;

/// Corpus perplexity of a hybrid model — the (P2) analogue of `Perplexity::corpus_perplexity`.
fn hybrid_corpus_perplexity<D, R>(model: &HybridLanguageModel<D>, reader: &R) -> f64
where
    D: MappedDictionary<Value = NgramEntry> + Send + Sync,
    R: CorpusReader,
{
    let mut total_log_prob = 0.0;
    let mut total_tokens = 0usize;

    for sentence in reader.sentences() {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        total_log_prob += model.sentence_log_prob(&tokens);  // same context rule as the n-gram
        total_tokens += tokens.len();
    }

    if total_tokens == 0 {
        return f64::INFINITY;
    }
    (-total_log_prob / total_tokens as f64).exp()
}
```

Sweeping the interpolation weight $`\alpha`$ needs one hybrid per $`\alpha`$, and
`HybridConfig::strategy` is fixed at construction. Since `SubwordEmbedding` is **not** `Clone` and
the hybrid takes ownership of it, materialize a fresh embedding per candidate by round-tripping it
through disk (feature `serde-extras`) — this is the honest, compiling way to do a sweep:

```rust
use libgrammstein::embedding::SubwordEmbedding;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};

// `ngram` and `embedding` were trained in Train and Evaluate §5–§6.
embedding.save("embedding.bin")?;   // write it once …

for alpha in [0.9, 0.8, 0.7, 0.5, 0.3] {
    let embedding = SubwordEmbedding::load("embedding.bin")?;  // … and reload a fresh copy per α
    let hybrid = HybridLanguageModel::new(
        ngram.clone(),                                          // NgramModel *is* Clone
        embedding,
        HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha },
            ..Default::default()
        },
    );

    let reader = libgrammstein::corpus::PlaintextReader::from_file("test.txt")?;
    println!("α = {alpha:.1} → PP {:.2}", hybrid_corpus_perplexity(&hybrid, &reader));
}
# Ok::<(), libgrammstein::Error>(())
```

**Interpreting the comparison — honestly.** The embedding expert's $`\mathbb{P}_e`$ is
*unnormalized*: it converts cosine similarity into a pseudo-probability without the
vocabulary-wide softmax denominator ([Hybrid Interpolation](../components/hybrid/interpolation.md)).
Consequences:

- Comparing **hybrid $`\alpha`$ against hybrid $`\alpha'`$** is sound — same scale, one knob.
- Comparing **hybrid against pure n-gram** by perplexity is *indicative, not decisive*: the
  hybrid's mass need not sum to one over the vocabulary, so its "perplexity" is a monotone score,
  not a calibrated branching factor. Judge the hybrid on the **downstream task** (correction
  accuracy, ranking quality) and on **OOV behavior**, and use perplexity to tune $`\alpha`$ within
  the hybrid family.

## 7. Perplexity as a text-quality filter

Turn the metric around. Freeze a model you trust, and score *candidate text* with it: a sentence
whose per-token perplexity is wildly high is, from the model's point of view, not the language it
was trained on — boilerplate, OCR garbage, markup, a different language, or a corrupted line. This
is the cheapest useful corpus filter there is.

Fix a threshold $`\tau`$ and keep sentence $`s`$ when

```math
\begin{array}{lr}
\displaystyle \mathrm{PP}(s) \;=\; \exp\!\Bigl(-\tfrac{1}{\lvert s \rvert} \sum_{i=1}^{\lvert s \rvert}
                      \log \mathbb{P}(w_i \mid h_i)\Bigr) \;\leq\; \tau & \text{(P4)}
\end{array}
```

Note the **per-token** normalization: without it, long sentences would be penalized purely for
being long.

```rust
use libgrammstein::corpus::{CorpusReader, PlaintextReader};
use libgrammstein::scoring::Perplexity;

/// Keep only sentences the model finds plausible. `tau` is a perplexity ceiling.
fn filter_by_perplexity<D, R>(evaluator: &Perplexity<'_, D>, reader: &R, tau: f64) -> Vec<String>
where
    D: libdictenstein::MutableMappedDictionary<Value = libgrammstein::ngram::NgramEntry>,
    R: CorpusReader,
{
    reader
        .sentences()
        .filter(|sentence| {
            let tokens: Vec<&str> = sentence.split_whitespace().collect();
            if tokens.is_empty() {
                return false;
            }
            let pp = (-evaluator.sentence_log_prob(&tokens) / tokens.len() as f64).exp();
            pp <= tau
        })
        .collect()
}

// Calibrate tau on data you trust, then apply it to data you do not.
let evaluator = Perplexity::new(&model);
let clean = filter_by_perplexity(&evaluator, &PlaintextReader::from_file("scraped.txt")?, 500.0);
println!("kept {} sentences", clean.len());
# Ok::<(), libgrammstein::Error>(())
```

**Calibrating $`\tau`$.** Never pick it out of thin air: score a corpus you *know* is good, take a
high quantile of its per-sentence perplexity distribution (the 95th is a reasonable start), and use
that as $`\tau`$. Re-calibrate whenever the model or the tokenizer changes — $`\tau`$ is a property
of the pair, not of the language.

> **Do not filter the training corpus with a model trained on it.** That is circular: the filter
> would keep exactly what the model already believes and quietly destroy the corpus's diversity.
> Filter *new* text with a model trained on *trusted* text.

libgrammstein also ships a rule-based [`QualityFilter`](../components/corpus/overview.md)
(`libgrammstein::corpus::QualityFilterBuilder`) for the cheap structural checks — length,
character-class ratios, repetition. Use it *before* the perplexity filter: it is far cheaper, and
it removes the garbage that would otherwise skew your $`\tau`$ calibration.

## 8. From the CLI

With the `cli` feature, no Rust is required:

```sh
# Perplexity of a saved model on a test corpus (add --per-sentence for a per-line breakdown).
grammstein eval perplexity model.bin test.txt --format plaintext --output report.json

# Rank two or more models on the same corpus, side by side.
grammstein eval compare test.txt model-3gram.bin model-4gram.bin model-5gram.bin
```

See the [CLI guide](../cli/README.md).

## 9. Pitfalls

| Pitfall | Why it bites | Fix |
|---|---|---|
| Comparing PP across **different vocabularies** | $`\mathrm{PP}`$ depends on $`\lvert V \rvert`$ through the OOV floor $`1/\lvert V \rvert`$; a smaller vocabulary flatters itself | only compare models trained on the same corpus with the same tokenizer, or report OOV rate alongside |
| Comparing PP across **different tokenizations** | fewer, longer tokens ⇒ lower $`N`$ ⇒ different scale entirely | fix the tokenizer first (§5) |
| Evaluating on the **training corpus** | $`\mathrm{PP}`$ collapses toward 1 and means nothing | always hold out a disjoint test set |
| Averaging **per-sentence perplexities** | the mean of exponentials is not the exponential of the mean | accumulate log-probabilities and exponentiate once, as in $`(\mathrm{P2})`$ |
| A high OOV rate on clean text | almost always punctuation, not vocabulary | compare `in_vocabulary("fox")` with `in_vocabulary("fox.")` (§5) |
| Treating hybrid PP as calibrated | $`\mathbb{P}_e`$ is unnormalized | tune $`\alpha`$ with it; judge the model with a downstream task (§6) |
| `Err(EmptyCorpus)` | every sentence was shorter than `Tokenizer`'s 10-character minimum | `with_min_sentence_length(0)`, or check the file is not empty |

## References

1. F. Jelinek, R. L. Mercer, L. R. Bahl & J. K. Baker (1977). *Perplexity — a measure of the
   difficulty of speech recognition tasks.* JASA 62(S1), S63.
   [doi:10.1121/1.2016299](https://doi.org/10.1121/1.2016299)
2. C. E. Shannon (1948). *A mathematical theory of communication.* Bell System Technical Journal
   27(3), 379–423.
   [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
3. S. F. Chen & J. Goodman (1999). *An empirical study of smoothing techniques for language
   modeling.* Computer Speech & Language 13(4), 359–393.
   [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
4. J. T. Goodman (2001). *A bit of progress in language modeling.* Computer Speech & Language
   15(4), 403–434. [doi:10.1006/csla.2001.0174](https://doi.org/10.1006/csla.2001.0174)
5. P. F. Brown, S. A. Della Pietra, V. J. Della Pietra, J. C. Lai & R. L. Mercer (1992). *An
   estimate of an upper bound for the entropy of English.* Computational Linguistics 18(1), 31–40.
   [aclanthology.org/J92-1002](https://aclanthology.org/J92-1002/)

## See also

- [Train and Evaluate](train-and-evaluate.md) — where the models being scored here come from
- [Domain Adaptation](domain-adaptation.md) — perplexity as the objective a mixture weight is fitted against
- [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md) — why $`\log \mathbb{P}`$ is always finite
- [Hybrid Interpolation](../components/hybrid/interpolation.md) — what $`\mathbb{P}_e`$ is, and why it is not calibrated
- [Corpus Overview](../components/corpus/overview.md) — readers, normalizers, quality filters
- [Hyperparameters](../training/hyperparameters.md) — the knobs a perplexity sweep should move
