# Example: Domain Adaptation

A model trained on news is confidently wrong about radiology reports. **Domain adaptation** fixes
that without throwing away the broad model you already paid for: train a second, small model on
in-domain text and **mix** the two, with a single weight $`\lambda`$ fitted on held-out in-domain
data. This page gives the four adaptation rules, *proves* that the mixture weight can be fitted to
global optimality by a one-dimensional search (Lemma 1), and assembles a complete, compiling
program that does it.

> **Scope.** Source of truth: [`src/ngram/trainer.rs`](../../src/ngram/trainer.rs),
> [`src/ngram/model.rs`](../../src/ngram/model.rs),
> [`src/embedding/trainer.rs`](../../src/embedding/trainer.rs) (`train_continued`) and
> [`src/scoring/perplexity.rs`](../../src/scoring/perplexity.rs). The mixture itself is
> *your* code — libgrammstein deliberately exposes `log_prob` / `score` rather than baking one
> adaptation policy in. Prerequisites: [Train and Evaluate](train-and-evaluate.md) and
> [Perplexity Scoring](perplexity-scoring.md).

## Notation

| Symbol | Meaning |
|---|---|
| $`w`$ | a word (token) |
| $`h`$ | its history (context) |
| $`\mathbb{P}_g(w \mid h)`$ | the **general** expert's probability (broad corpus) |
| $`\mathbb{P}_d(w \mid h)`$ | the **domain** expert's probability (in-domain corpus) |
| $`\mathbb{P}_\lambda(w \mid h)`$ | the adapted (mixed) probability |
| $`\lambda \in [0,1]`$ | the mixture weight — the mass given to the **domain** expert |
| $`\mathcal{D}`$ | the held-out in-domain **development** set, $`N`$ tokens $`w_1 \ldots w_N`$ |
| $`p_i`$ | shorthand for $`\mathbb{P}_g(w_i \mid h_i)`$ on the dev set |
| $`q_i`$ | shorthand for $`\mathbb{P}_d(w_i \mid h_i)`$ on the dev set |
| $`\mathrm{LL}(\lambda)`$ | dev-set log-likelihood of the mixture, $`\sum_i \log \mathbb{P}_\lambda(w_i \mid h_i)`$ |
| $`\mathrm{PP}(\lambda)`$ | dev-set perplexity of the mixture |
| $`V_g,\ V_d`$ | the general and domain **vocabularies** (words seen at least once) |
| $`\lambda^{*}`$ | the fitted weight, $`\arg\min_\lambda \mathrm{PP}(\lambda)`$ |

**Acronyms.** *EM* — Expectation-Maximization; *PP* — perplexity; *OOV* — Out-Of-Vocabulary;
*MKN* — Modified Kneser-Ney.

## 1. The problem

| Domain | What breaks a general model |
|---|---|
| Medical | drug names, anatomy, dosage syntax (*"5 mg PO q8h"*) |
| Legal | Latin phrases, archaic constructions, citation forms |
| Software | identifiers, API names, code-switching between prose and symbols |
| Social | slang, elision, hashtags, non-standard orthography |

Two failure modes hide behind one symptom (a bad perplexity):

1. **Coverage.** The domain word is simply absent — $`w \notin V_g`$ — so the general model floors
   it at $`1/\lvert V_g \rvert`$. A bigger general corpus helps a little; a domain corpus helps a lot.
2. **Priors.** The word exists but the *context statistics* are wrong: in general English,
   *"the patient presented"* is bizarre; in a clinical note it is the most ordinary sentence there is.

The domain corpus fixes both — but it is small, so a model trained on it *alone* is sparse and
brittle. That tension is exactly what a mixture resolves.

## 2. Three ways to adapt

Let both experts be trained with the **same order and the same tokenizer** (§4 explains why this is
not optional).

### Linear mixture (the default)

A convex combination in **probability** space [[1]](#references):

```math
\mathbb{P}_\lambda(w \mid h) \;=\; (1 - \lambda)\,\mathbb{P}_g(w \mid h) \;+\; \lambda\,\mathbb{P}_d(w \mid h)
\tag{D1}
```

$`\lambda = 0`$ is the general model, $`\lambda = 1`$ is the domain model. Because both experts are
proper distributions, so is the mixture — it sums to one over the vocabulary for every
$`\lambda`$. This is the rule to reach for first, and the only one whose weight can be fitted to
*provable* optimality (§3).

### Log-linear mixture (product of experts)

A convex combination in **log** space:

```math
\log \tilde{\mathbb{P}}_\lambda(w \mid h) \;=\; (1 - \lambda)\log \mathbb{P}_g(w \mid h)
                                        \;+\; \lambda \log \mathbb{P}_d(w \mid h)
\tag{D2}
```

This is a geometric mean: a low probability from *either* expert strongly suppresses the product
(an AND-like combination), which sharpens the model when the two experts disagree.

> **$`(\mathrm{D2})`$ is unnormalized.** The tilde is not decoration: a geometric mean of two
> distributions does not sum to one, and renormalizing would require a sum over the entire
> vocabulary at every query. Use $`(\mathrm{D2})`$ for **ranking** (which candidate is better?),
> and $`(\mathrm{D1})`$ when you need a probability — including whenever you intend to report a
> perplexity. Note the identical trade-off inside the hybrid model
> ([Hybrid Interpolation](../components/hybrid/interpolation.md)).

### Vocabulary-gated backoff

A hard switch on domain-vocabulary membership:

```math
\mathbb{P}(w \mid h) \;=\;
\begin{cases}
\mathbb{P}_d(w \mid h) & \text{if } w \in V_d \\
\mathbb{P}_g(w \mid h) & \text{otherwise}
\end{cases}
\tag{D3}
```

Cheap (no second query for in-domain words) and effective when the domain corpus covers its own
jargon well. It is also *brittle*: a word that appears once in the domain corpus is trusted
entirely to a model that has seen it once. Prefer $`(\mathrm{D1})`$ unless you have measured
$`(\mathrm{D3})`$ to be better.

### Context-dependent weight

Let the data decide how "in-domain" the current context is, and lean on the domain expert
proportionally:

```math
\lambda(h) \;=\; \min\Bigl(\lambda_{\max},\;
   \frac{1}{\lvert h \rvert}\sum_{v \in h} \mathbb{1}\bigl[v \in V_d \setminus V_g\bigr]\Bigr),
\qquad
\mathbb{P}(w \mid h) \;=\; \bigl(1 - \lambda(h)\bigr)\mathbb{P}_g + \lambda(h)\,\mathbb{P}_d
\tag{D4}
```

where $`\mathbb{1}[\cdot]`$ is the indicator function (1 if the condition holds, else 0), and
$`V_d \setminus V_g`$ is the set of words the domain corpus has and the general corpus does not —
the *domain markers*. The cap $`\lambda_{\max} < 1`$ keeps the general model in play even in the
most jargon-dense context. Use this for **mixed-domain documents** (a clinical note quoting a
patient's own words), where a single global $`\lambda`$ is a compromise between two regimes rather
than a fit to either.

## 3. Fitting $`\lambda`$

![Domain adaptation as a two-expert mixture. A broad general corpus trains the general expert P-g; a narrow in-domain corpus trains the domain expert P-d with the same order and tokenizer. The mixture combines them as one minus lambda times P-g plus lambda times P-d, where lambda zero is the general model and lambda one is the domain model. A held-out in-domain development set drives the fit: the mixture is evaluated on the dev set, and because the log-likelihood is concave in lambda a grid or ternary search — or EM — finds the global optimum lambda-star, which is fed back into the mixture. A disjoint in-domain test set then produces the final report: perplexity of the general model, the domain model, and the fitted mixture, alongside vocabulary coverage.](../diagrams/example-domain-adaptation.svg)

**Figure 1.** The mixture and its tuning loop. The dev set fits $`\lambda`$; a **disjoint** test
set reports the result. *(Rendered from `docs/diagrams/example-domain-adaptation.puml`.)*

### The objective

Fit $`\lambda`$ by minimizing dev-set perplexity — equivalently, by maximizing dev-set
log-likelihood, since $`\mathrm{PP}(\lambda) = \exp(-\mathrm{LL}(\lambda)/N)`$ and $`\exp(-x/N)`$
is strictly decreasing:

```math
\lambda^{*} \;=\; \arg\min_{\lambda \in [0,1]} \mathrm{PP}(\lambda)
            \;=\; \arg\max_{\lambda \in [0,1]} \mathrm{LL}(\lambda),
\qquad
\mathrm{LL}(\lambda) = \sum_{i=1}^{N} \log\bigl((1-\lambda)\,p_i + \lambda\,q_i\bigr)
\tag{D5}
```

The $`p_i`$ and $`q_i`$ **do not depend on $`\lambda`$**. Compute them once — one pass of each
expert over the dev set — and every subsequent evaluation of $`\mathrm{LL}(\lambda)`$ costs
$`O(N)`$ arithmetic with no model queries at all. This single observation makes the search
essentially free.

### Lemma 1 (concavity)

> **Lemma 1.** Let $`p_i > 0`$ and $`q_i > 0`$ for $`i = 1, \ldots, N`$. Then
> $`\mathrm{LL}(\lambda) = \sum_{i=1}^{N} \log\bigl((1-\lambda)p_i + \lambda q_i\bigr)`$ is concave
> on $`[0,1]`$. It is *strictly* concave unless $`p_i = q_i`$ for every $`i`$, in which case
> $`\mathrm{LL}`$ is constant.

**Proof.** Fix $`i`$ and write $`a_i(\lambda) = (1-\lambda)p_i + \lambda q_i = p_i + \lambda(q_i - p_i)`$.

1. *$`a_i`$ is affine in $`\lambda`$ and strictly positive on $`[0,1]`$.* Affine is immediate from
   the displayed form. Positivity: for $`\lambda \in [0,1]`$, $`a_i(\lambda)`$ is a convex
   combination of $`p_i > 0`$ and $`q_i > 0`$, and a convex combination of two strictly positive
   numbers is strictly positive.
2. *Each term is concave.* $`g_i(\lambda) = \log a_i(\lambda)`$ is twice differentiable on
   $`[0,1]`$ by step 1, with
   ```math
   g_i'(\lambda) = \frac{q_i - p_i}{a_i(\lambda)},
   \qquad
   g_i''(\lambda) = -\frac{(q_i - p_i)^2}{a_i(\lambda)^2} \;\leq\; 0 .
   ```
   A twice-differentiable function with non-positive second derivative on an interval is concave
   there. (Equivalently, and without derivatives: $`\log`$ is concave on $`(0,\infty)`$, and the
   composition of a concave function with an affine map is concave [[5, §3.2.2]](#references).)
3. *The sum is concave.* $`\mathrm{LL} = \sum_i g_i`$ is a finite sum of concave functions, hence
   concave [[5, §3.2.1]](#references), with
   ```math
   \mathrm{LL}''(\lambda) \;=\; -\sum_{i=1}^{N} \frac{(q_i - p_i)^2}{a_i(\lambda)^2} \;\leq\; 0 .
   ```
4. *Strictness.* $`\mathrm{LL}''(\lambda) = 0`$ for some $`\lambda`$ forces every summand to vanish,
   i.e. $`q_i = p_i`$ for all $`i`$; then $`a_i(\lambda) = p_i`$ is constant in $`\lambda`$ and
   $`\mathrm{LL}`$ is a constant function. Otherwise at least one summand is strictly negative for
   every $`\lambda \in [0,1]`$, so $`\mathrm{LL}'' < 0`$ and $`\mathrm{LL}`$ is strictly concave. $`\blacksquare`$

**The hypothesis $`p_i, q_i > 0`$ is not an assumption — it is a guarantee.** Modified Kneser-Ney
never returns zero: the recursion terminates in the strictly positive unigram term
$`1/\lvert V \rvert`$, and every interpolation weight along the way is non-negative, so the
interpolated probability is strictly positive for **every** word, seen or unseen
([Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md)). Lemma 1 therefore applies
unconditionally — and the same fact is what keeps the EM denominator below away from zero.

> **Corollary 1 (the search is safe).** By Lemma 1 the maximizer set of $`\mathrm{LL}`$ on
> $`[0,1]`$ is a non-empty closed interval — a single point unless the two experts agree on every
> dev token. Since $`\mathrm{PP}`$ is a strictly decreasing function of $`\mathrm{LL}`$, that set
> is exactly $`\arg\min \mathrm{PP}`$. Consequently **any** local optimum found by a
> one-dimensional search is global, a ternary search converges to it, and no restart strategy or
> multi-start heuristic is needed. The optimum may lie at an endpoint ($`\lambda^{*} = 0`$ or
> $`1`$); concavity permits this and ternary search finds it.

### Two ways to search

**Ternary search** — bisection for concave functions. Each iteration shrinks the bracket by a
factor $`2/3`$, so $`60`$ iterations pin $`\lambda^{*}`$ to well below `f64` resolution:

```rust
/// Dev-set log-likelihood of the mixture (D5), from the cached component probabilities.
fn log_likelihood(components: &[(f64, f64)], lambda: f64) -> f64 {
    components
        .iter()
        .map(|(p, q)| ((1.0 - lambda) * p + lambda * q).ln())
        .sum()
}

/// Ternary search on [0, 1]. Sound because LL is concave (Lemma 1) — Corollary 1.
fn fit_lambda_ternary(components: &[(f64, f64)], iterations: usize) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..iterations {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if log_likelihood(components, m1) < log_likelihood(components, m2) {
            lo = m1; // the maximum cannot lie left of m1
        } else {
            hi = m2; // the maximum cannot lie right of m2
        }
    }
    0.5 * (lo + hi)
}
```

**EM** — the classical two-component mixture update [[3]](#references). Introduce a latent variable
$`z_i \in \{g, d\}`$ saying which expert generated token $`i`$. The E-step computes the posterior
("responsibility") that the *domain* expert did, and the M-step sets $`\lambda`$ to its mean:

```math
r_i(\lambda) \;=\; \frac{\lambda\,q_i}{(1-\lambda)\,p_i + \lambda\,q_i},
\qquad
\lambda' \;=\; \frac{1}{N}\sum_{i=1}^{N} r_i(\lambda)
\tag{D6}
```

Each iteration is guaranteed not to decrease $`\mathrm{LL}`$ [[3]](#references); combined with
Lemma 1, it converges to the global optimum from any interior start.

```rust
/// EM for the two-component mixture (D6). Monotone in LL; converges to the global optimum.
fn fit_lambda_em(components: &[(f64, f64)], iterations: usize) -> f64 {
    let mut lambda = 0.5;
    for _ in 0..iterations {
        // E-step: posterior responsibility of the domain expert for each dev token.
        // The denominator is strictly positive because MKN probabilities are (Lemma 1).
        let responsibility: f64 = components
            .iter()
            .map(|(p, q)| lambda * q / ((1.0 - lambda) * p + lambda * q))
            .sum();
        // M-step: the new weight is the mean responsibility.
        lambda = responsibility / components.len() as f64;
    }
    lambda
}
```

Both land on the same $`\lambda^{*}`$. Ternary search is easier to reason about and needs no
initialization; EM converges in a handful of iterations and generalizes directly to more than two
experts. A coarse grid ($`\lambda \in \{0, 0.1, \ldots, 1\}`$) is fine for a first look and is what
you will want to *plot*, but do not ship it as the fit: it is a discretization of an objective you
can optimize exactly.

## 4. What the two experts must share

| Must match | Why |
|---|---|
| **Tokenizer** | $`p_i`$ and $`q_i`$ must be probabilities *of the same token sequence*. Different tokenizers make $`(\mathrm{D1})`$ a category error. Both models default to `Tokenizer::new()` — keep it that way, or set the same custom tokenizer on both. |
| **Order** (recommended) | Not required by $`(\mathrm{D1})`$ — the mixture is over probabilities, not counts — but differing orders mean the two experts condition on different amounts of context, which muddies the interpretation of $`\lambda`$. Deliberately lowering the *domain* order is a legitimate response to a sparse domain corpus; do it knowingly. |
| **Evaluation corpus** | $`\lambda^{*}`$ is fitted on the dev set and reported on a **disjoint** test set. Fitting and reporting on the same text is self-congratulation, not measurement. |

**They need not share a vocabulary.** Each expert floors an unknown word at its own
$`1/\lvert V \rvert`$; the general model's larger vocabulary gives it a *lower* floor, which the
mixture handles correctly because it mixes probabilities, not counts.

> **Do not train these models in vocabulary-indexed mode.** `TrainerBuilder::with_vocabulary` /
> `with_vocabulary_path` switch the trie to **PUA-character** keys (each word becomes one Private
> Use Area codepoint), whereas `log_prob`, `sentence_log_prob`, `in_vocabulary` and `Perplexity`
> all build **legacy pipe-separated** keys (`"the|quick|brown"`). A vocabulary-indexed model queried
> through `log_prob` therefore misses *every* key and silently backs off to the uniform floor —
> you get finite, plausible-looking, meaningless numbers. Vocabulary mode exists for the
> Google-Books import path, where lookups go through `encode_ngram_key`. For everything on this
> page, use the default (legacy) mode — which is what `TrainerBuilder::new(dict).order(n)` gives
> you. See [`src/ngram/trie.rs`](../../src/ngram/trie.rs) (`encode_key_legacy`) versus
> [`src/ngram/vocabulary.rs`](../../src/ngram/vocabulary.rs) (`encode_ngram_key`).

## 5. Adapting the embedding

The n-gram side is adapted by mixing. The embedding side has a second option: **continue training**
an existing model on the domain corpus, keeping its weights and vocabulary and resuming the
learning-rate schedule at the right epoch.

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::{EmbeddingTrainerBuilder, SubwordEmbedding};

// A general embedding trained for 5 epochs, saved earlier (feature `serde-extras`).
let general_embedding = SubwordEmbedding::load("general-embedding.bin")?;

// Continue for epochs 5..8 on the domain corpus: same `dim` as the loaded model,
// `epochs` = the TOTAL, `start_epoch` = how many are already done.
let adapted = EmbeddingTrainerBuilder::new()
    .dim(general_embedding.dim())
    .epochs(8)
    .learning_rate(0.01)                    // a gentler rate for fine-tuning
    .train_continued(general_embedding, 5, PlaintextReader::from_file("domain.txt")?)?;

adapted.save("adapted-embedding.bin")?;
# Ok::<(), libgrammstein::Error>(())
```

> **What `train_continued` does and does not do.** It keeps the loaded model's weights **and its
> vocabulary**; it re-derives the ephemeral corpus statistics (the negative-sampling table and the
> sub-sampling counts) against that existing vocabulary so embedding-row indices stay consistent;
> then it runs epochs `start_epoch .. epochs`. It does **not** grow the vocabulary. A domain word
> absent from the general vocabulary therefore gets no row of its own: it remains *scorable*
> (its vector is composed from hashed character n-grams — the FastText property
> [[4]](#references)) but it can never be *returned* by `most_similar`, which ranks over the
> stored vocabulary. If domain terms must be first-class neighbours, train a **fresh** domain
> embedding instead and mix at the hybrid level.

## 6. The complete program

```rust
use std::io::Write;

use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libdictenstein::MappedDictionary;
use libgrammstein::corpus::{CorpusReader, PlaintextReader};
use libgrammstein::ngram::{NgramEntry, NgramModel, TrainerBuilder};
use libgrammstein::scoring::Perplexity;

/// A two-expert linear mixture — equation (D1).
///
/// Generic over the dictionary backend `D`; `MappedDictionary` is exactly what
/// `NgramModel<D>` requires, so no stronger bound is needed here.
struct DomainMixture<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    general: NgramModel<D>,
    domain: NgramModel<D>,
    lambda: f64,
}

impl<D> DomainMixture<D>
where
    D: MappedDictionary<Value = NgramEntry>,
{
    fn new(general: NgramModel<D>, domain: NgramModel<D>, lambda: f64) -> Self {
        Self {
            general,
            domain,
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// log P_λ(w | h) — (D1). Both experts return natural log-probabilities, so we
    /// exponentiate, mix in probability space, and take the log back.
    fn log_prob(&self, word: &str, context: &[&str]) -> f64 {
        let p = self.general.log_prob(word, context).exp();
        let q = self.domain.log_prob(word, context).exp();
        ((1.0 - self.lambda) * p + self.lambda * q).ln()
    }

    /// The (p_i, q_i) pairs of (D5) for one corpus — computed ONCE, reused for every λ.
    fn components<R: CorpusReader>(&self, reader: &R) -> Vec<(f64, f64)> {
        let order = self.general.order().max(self.domain.order());
        let mut out = Vec::new();

        for sentence in reader.sentences() {
            let tokens: Vec<&str> = sentence.split_whitespace().collect();
            for i in 0..tokens.len() {
                let start = i.saturating_sub(order - 1);
                let context = &tokens[start..i];
                out.push((
                    self.general.log_prob(tokens[i], context).exp(),
                    self.domain.log_prob(tokens[i], context).exp(),
                ));
            }
        }
        out
    }

    /// Corpus perplexity of the mixture at the current λ — (P2) of the perplexity page.
    fn corpus_perplexity<R: CorpusReader>(&self, reader: &R) -> f64 {
        let components = self.components(reader);
        perplexity_at(&components, self.lambda)
    }
}

/// Dev-set log-likelihood (D5) from cached components — no model queries.
fn log_likelihood(components: &[(f64, f64)], lambda: f64) -> f64 {
    components
        .iter()
        .map(|(p, q)| ((1.0 - lambda) * p + lambda * q).ln())
        .sum()
}

/// Dev-set perplexity at λ.
fn perplexity_at(components: &[(f64, f64)], lambda: f64) -> f64 {
    if components.is_empty() {
        return f64::INFINITY;
    }
    (-log_likelihood(components, lambda) / components.len() as f64).exp()
}

/// Ternary search — sound by Lemma 1 / Corollary 1.
fn fit_lambda_ternary(components: &[(f64, f64)], iterations: usize) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..iterations {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if log_likelihood(components, m1) < log_likelihood(components, m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    0.5 * (lo + hi)
}

/// EM for the two-component mixture (D6).
fn fit_lambda_em(components: &[(f64, f64)], iterations: usize) -> f64 {
    let mut lambda = 0.5;
    for _ in 0..iterations {
        let responsibility: f64 = components
            .iter()
            .map(|(p, q)| lambda * q / ((1.0 - lambda) * p + lambda * q))
            .sum();
        lambda = responsibility / components.len() as f64;
    }
    lambda
}

const GENERAL: &str = "\
the company announced quarterly earnings today.
stock prices rose after the news release.
the market showed strong performance this week.
analysts predict favorable conditions ahead.
the board approved the new strategic plan.
the patient waited for the meeting to end.
";

const DOMAIN: &str = "\
the patient presented with acute symptoms.
blood pressure was elevated at admission.
laboratory results indicated elevated glucose levels.
the diagnosis confirmed type 2 diabetes.
treatment protocol includes insulin therapy.
the patient responded well to medication.
";

const DEV: &str = "\
blood glucose levels decreased after treatment.
the patient reported reduced symptoms today.
";

const TEST: &str = "\
the patient blood pressure normalized after admission.
laboratory results improved with the treatment protocol.
";

fn train(
    corpus: &std::path::Path,
    order: usize,
) -> libgrammstein::Result<NgramModel<DynamicDawgChar<NgramEntry>>> {
    TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(order)
        .train(PlaintextReader::from_file(corpus)?)
}

fn write(dir: &std::path::Path, name: &str, body: &str, repeats: usize) -> std::io::Result<std::path::PathBuf> {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path)?;
    for _ in 0..repeats {
        file.write_all(body.as_bytes())?;
    }
    Ok(path)
}

fn main() -> libgrammstein::Result<()> {
    let dir = tempfile::TempDir::new()?;
    // Repeat the training corpora so counts exceed the MKN discounts; dev/test are read once.
    let general_path = write(dir.path(), "general.txt", GENERAL, 20)?;
    let domain_path = write(dir.path(), "domain.txt", DOMAIN, 20)?;
    let dev_path = write(dir.path(), "dev.txt", DEV, 1)?;
    let test_path = write(dir.path(), "test.txt", TEST, 1)?;

    // ---- 1. two experts, same order, same (default) tokenizer -------------------------
    let general = train(&general_path, 3)?;
    let domain = train(&domain_path, 3)?;
    println!(
        "general |V| = {} · domain |V| = {}",
        general.vocab_size(),
        domain.vocab_size()
    );

    // ---- 2. baselines on the in-domain TEST set ---------------------------------------
    let general_pp = Perplexity::new(&general)
        .corpus_perplexity(&PlaintextReader::from_file(&test_path)?)?
        .perplexity;
    let domain_pp = Perplexity::new(&domain)
        .corpus_perplexity(&PlaintextReader::from_file(&test_path)?)?
        .perplexity;

    // ---- 3. fit λ on the DEV set (components cached once) ------------------------------
    let mut mixture = DomainMixture::new(general.clone(), domain.clone(), 0.5);
    let dev_components = mixture.components(&PlaintextReader::from_file(&dev_path)?);

    let lambda_ternary = fit_lambda_ternary(&dev_components, 60);
    let lambda_em = fit_lambda_em(&dev_components, 50);
    println!(
        "λ* ternary = {lambda_ternary:.4} · λ* EM = {lambda_em:.4} (they agree — Corollary 1)"
    );

    // Plot the objective: it is concave, so this curve has exactly one summit.
    for step in 0..=10 {
        let lambda = step as f64 / 10.0;
        println!(
            "  λ = {lambda:.1} → dev PP {:.2}",
            perplexity_at(&dev_components, lambda)
        );
    }

    // ---- 4. report on the disjoint TEST set --------------------------------------------
    mixture.lambda = lambda_ternary;
    let mixture_pp = mixture.corpus_perplexity(&PlaintextReader::from_file(&test_path)?);

    println!("\n{:<22} {:>12}", "model", "test PP");
    println!("{:<22} {:>12.2}", "general only", general_pp);
    println!("{:<22} {:>12.2}", "domain only", domain_pp);
    println!(
        "{:<22} {:>12.2}",
        format!("mixture (λ={lambda_ternary:.2})"),
        mixture_pp
    );

    // ---- 5. vocabulary coverage on the test set ----------------------------------------
    let reader = PlaintextReader::from_file(&test_path)?;
    let words: std::collections::HashSet<String> = reader
        .sentences()
        .flat_map(|s| s.split_whitespace().map(str::to_owned).collect::<Vec<_>>())
        .collect();
    let covered_general = words.iter().filter(|w| general.in_vocabulary(w)).count();
    let covered_domain = words.iter().filter(|w| domain.in_vocabulary(w)).count();
    println!(
        "\ncoverage of {} distinct test words: general {covered_general} · domain {covered_domain}",
        words.len()
    );

    Ok(())
}
```

**What it actually prints:**

```
general |V| = 34 · domain |V| = 31
λ* ternary = 0.4579 · λ* EM = 0.4579 (they agree — Corollary 1)
  λ = 0.0 → dev PP 58.14
  λ = 0.1 → dev PP 46.14
  λ = 0.2 → dev PP 42.41
  λ = 0.3 → dev PP 40.64
  λ = 0.4 → dev PP 39.87
  λ = 0.5 → dev PP 39.82
  λ = 0.6 → dev PP 40.44
  λ = 0.7 → dev PP 41.89
  λ = 0.8 → dev PP 44.66
  λ = 0.9 → dev PP 50.41
  λ = 1.0 → dev PP 84.70

model                       test PP
general only                  62.53
domain only                  119.85
mixture (λ=0.46)              38.85

coverage of 13 distinct test words: general 3 · domain 9
```

**How to read it — including the surprise.**

- **The mixture beats both experts, decisively** ($`38.85`$ against $`62.53`$ and $`119.85`$).
  That is the entire thesis of adaptation: two models that are wrong in *different* ways are worth
  more together than either alone.
- **The two fitting routines agree to four decimals.** That is not luck; it is Corollary 1. A
  concave objective has one summit, and both searches climb to it.
- **The $`\lambda`$ sweep is visibly unimodal** — down to a single interior minimum near
  $`\lambda \approx 0.46`$, then up again. You are looking at the concavity proved in Lemma 1,
  rendered as a curve.
- **The surprise: the domain-only model is *worse* than the general-only model** ($`119.85`$ vs
  $`62.53`$) *even though it covers far more of the test vocabulary* (9 of 13 distinct words
  against 3). Coverage is not fit. A six-sentence corpus repeated twenty times produces a sharply
  peaked model: it is *certain* the token after *"the patient"* is *"presented"*, and when the test
  text says *"blood"* instead, that confident wrong prediction costs more than the general model's
  diffuse shrug. The domain expert is still carrying real information — the mixture's win proves
  it — but it cannot stand alone. This is precisely the regime $`(\mathrm{D1})`$ exists for.
- **These are toy corpora.** Six sentences each: the *shape* of the result is the lesson, not the
  magnitudes. On realistic corpora the domain expert usually does beat the general one on in-domain
  text, and $`\lambda^{*}`$ moves higher — often into $`[0.6, 0.9]`$. What does *not* change is the
  method: cache the components, fit $`\lambda`$ on dev, report on test.

## 7. Reading the result

| Observation | Diagnosis | Action |
|---|---|---|
| $`\lambda^{*} \approx 1`$ | the general model adds nothing on this dev set | check the dev set really is in-domain; consider dropping the general expert |
| $`\lambda^{*} \approx 0`$ | the domain model is too sparse or too noisy to help | collect more in-domain text, or lower the *domain* order (a bigram may estimate what a trigram cannot) |
| domain-only PP **worse** than general-only, yet the mixture still wins | the domain corpus is too small to stand alone — sharply peaked and confidently wrong off its few training contexts — but it still carries real information | keep it in the mixture (that is what $`(\mathrm{D1})`$ is for); grow the corpus, or lower the domain order, before promoting it |
| domain-only covers more test words but scores worse | **coverage is not fit**: a word can be in $`V_d`$ and still be badly predicted in context | judge with perplexity, not with vocabulary overlap |
| $`\mathrm{PP}(\lambda^{*})`$ barely below $`\min(\mathrm{PP}_g, \mathrm{PP}_d)`$ | the experts are nearly redundant | the corpora overlap — the "domain" is not a domain |
| dev PP improves, test PP does not | $`\lambda`$ is overfitted to a small dev set | enlarge the dev set; $`\lambda`$ is one parameter, but it is still a parameter |
| domain coverage $`\approx`$ general coverage | the jargon is already in the general vocabulary | the problem is **priors**, not coverage — the mixture is still the right tool |

Report $`\lambda^{*}`$ alongside the perplexities, always. A mixture without its weight is not a
reproducible result.

## 8. Pitfalls

| Pitfall | Fix |
|---|---|
| Fitting and reporting on the same corpus | fit on dev, report on a **disjoint** test set (§4) |
| Mixing in log space and calling it a probability | $`(\mathrm{D2})`$ is unnormalized; report perplexity only for $`(\mathrm{D1})`$ |
| Re-querying the models inside the $`\lambda`$ loop | cache the $`(p_i, q_i)`$ pairs once — they do not depend on $`\lambda`$ (§3) |
| Vocabulary-indexed training | `log_prob` silently backs off to the uniform floor; use the default legacy mode (§4) |
| `SubwordEmbedding::clone` | it is not `Clone`; `save`/`load` a second copy, or reach through `hybrid.embedding_model()` |
| Expecting `train_continued` to learn new words | it keeps the loaded vocabulary; train a fresh embedding if domain terms must be neighbours (§5) |
| Different tokenizers per expert | $`(\mathrm{D1})`$ becomes meaningless; keep `Tokenizer::new()` on both |

## References

1. F. Jelinek & R. L. Mercer (1980). *Interpolated estimation of Markov source parameters from
   sparse data.* In *Pattern Recognition in Practice*, 381–397. North-Holland.
2. J. R. Bellegarda (2004). *Statistical language model adaptation: review and perspectives.*
   Speech Communication 42(1), 93–108.
   [doi:10.1016/j.specom.2003.08.002](https://doi.org/10.1016/j.specom.2003.08.002)
3. A. P. Dempster, N. M. Laird & D. B. Rubin (1977). *Maximum likelihood from incomplete data via
   the EM algorithm.* Journal of the Royal Statistical Society B 39(1), 1–22.
   [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)
4. P. Bojanowski, E. Grave, A. Joulin & T. Mikolov (2017). *Enriching word vectors with subword
   information* (FastText). TACL 5, 135–146.
   [doi:10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)
5. S. Boyd & L. Vandenberghe (2004). *Convex Optimization.* Cambridge University Press. §3.2.1
   (non-negative weighted sums) and §3.2.2 (composition with an affine map).
   [doi:10.1017/CBO9780511804441](https://doi.org/10.1017/CBO9780511804441)
6. S. F. Chen & J. Goodman (1999). *An empirical study of smoothing techniques for language
   modeling.* Computer Speech & Language 13(4), 359–393.
   [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)

## See also

- [Perplexity Scoring](perplexity-scoring.md) — the objective $`\lambda`$ is fitted against
- [Train and Evaluate](train-and-evaluate.md) — how each expert is built
- [Hybrid Interpolation](../components/hybrid/interpolation.md) — the same mixing algebra, applied to n-gram ⊕ embedding
- [Embedding Training](../training/embedding.md) — epochs, learning rate, and what `train_continued` resumes
- [Hyperparameters](../training/hyperparameters.md) — the wider tuning surface
- [Spell Correction](spell-correction.md) — a downstream task that *feels* the adaptation
