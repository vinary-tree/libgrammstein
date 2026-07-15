# Example: Spell Correction

Spelling correction is the **noisy-channel model** made concrete: a Levenshtein automaton proposes
what the writer *might* have meant, and a language model decides which of those candidates they
*did* mean. libgrammstein ships this as `HierarchicalCorrector` — a batteries-included façade over
[lling-llang](https://github.com/f1r3fly-io/lling-llang)'s WFST lattice framework — and this page
walks it end to end, from the three-line version to a custom pipeline you assemble yourself.

> **Scope.** Source of truth: [`src/integration/corrector.rs`](../../src/integration/corrector.rs)
> (`HierarchicalCorrector`, `LevenshteinCorrectionLayer`, `CorrectorConfig`, `EditConfig`) and
> [`src/integration/lling_llang.rs`](../../src/integration/lling_llang.rs)
> (`GrammsteinLanguageModel`). The runnable program is
> [`examples/correct_sentence.rs`](../../examples/correct_sentence.rs). Everything here needs the
> **`lling-llang-integration`** feature. For the theory in full — the $`N`$-dimensional view,
> the semiring algebra, WFST composition — read
> [Hierarchical Correction](../integration/lling-llang/hierarchical-correction.md).

## Notation

| Symbol | Meaning |
|---|---|
| $`x`$ | the **observed** (possibly corrupted) input — the tokens the user typed |
| $`w`$ | a **candidate** correction — a token sequence the corrector might output |
| $`\hat{w}`$ | the corrector's chosen output |
| $`h`$ | the history (context) preceding a token |
| $`\mathbb{P}(w)`$ | the **prior**: how plausible $`w`$ is as language — supplied by the n-gram / hybrid model |
| $`\mathbb{P}(x \mid w)`$ | the **channel**: how plausibly $`w`$ was corrupted into $`x`$ — supplied by edit distance |
| $`d(x_i, w_i)`$ | the edit distance between the observed token and the candidate |
| $`k`$ | the maximum edit distance the automaton searches (`max_edit_distance`) |
| $`\lambda \in [0,1]`$ | the language-model weight (`lm_weight`) that balances prior against channel |
| $`V`$ | the **spelling vocabulary** — the set of words the corrector may output |

**Acronyms.** *OOV* — Out-Of-Vocabulary; *WFST* — Weighted Finite-State Transducer; *LM* —
Language Model; *CFG* — Context-Free Grammar; *MKN* — Modified Kneser-Ney.

## 1. The noisy channel

Bayes' rule, applied to typing. The writer intended $`w`$; a noisy channel (fingers, keyboard,
OCR, autocorrect) turned it into $`x`$. Recover the intent by maximizing the posterior
[[1]](#references)[[2]](#references):

```math
\hat{w} \;=\; \arg\max_{w} \mathbb{P}(w \mid x)
        \;=\; \arg\max_{w} \; \underbrace{\mathbb{P}(w)}_{\text{prior}} \cdot
                              \underbrace{\mathbb{P}(x \mid w)}_{\text{channel}}
\tag{S1}
```

The denominator $`\mathbb{P}(x)`$ is dropped because it is constant across candidates. Two factors
remain, and libgrammstein supplies one each:

| Factor | Question | Engine |
|---|---|---|
| $`\mathbb{P}(x \mid w)`$ | "how badly mangled is this?" | liblevenshtein automaton $`\cap`$ vocabulary — the **channel** |
| $`\mathbb{P}(w)`$ | "is this even English?" | libgrammstein MKN n-gram (or hybrid) — the **prior** |

**Into the tropical semiring.** Take negative logarithms: products become sums, and $`\arg\max`$
becomes $`\arg\min`$. Maximizing a probability is *identical* to finding a **minimum-cost path**
through a weighted lattice under the tropical semiring $`(\min, +)`$ [[3]](#references):

```math
\hat{w} \;=\; \arg\min_{w} \Bigl[\;\underbrace{-\log \mathbb{P}(x \mid w)}_{\text{channel cost}}
                          \;+\; \underbrace{-\log \mathbb{P}(w)}_{\text{prior cost}}\;\Bigr]
\tag{S2}
```

That is why the corrector is a lattice search and not a loop over candidate strings: the lattice
encodes the *combinatorial product* of per-token alternatives, and Viterbi finds the best whole
sentence — not the concatenation of locally-best words.

**The shipped channel model.** libgrammstein sets $`-\log \mathbb{P}(x_i \mid w_i) \triangleq d(x_i, w_i)`$:
the cost of a correction edge *is* its edit distance, which is equivalent to the channel
$`\mathbb{P}(x_i \mid w_i) \propto e^{-d(x_i, w_i)}`$ — every edit equally likely. This is the
classical Levenshtein channel. A *learned* channel with per-edit costs (so that `a`→`e` is cheaper
than `a`→`q`) is the natural refinement [[4]](#references); it would slot in as a different edge
weight without changing anything else in $`(\mathrm{S2})`$.

**The layer that fuses them.** In the shipped pipeline the language-model layer does not simply add
the two costs — it interpolates them with the weight $`\lambda`$ (`lm_weight`), rewriting each edge
weight $`w`$ as

```math
w' \;=\; (1 - \lambda)\cdot w \;+\; \lambda \cdot \bigl(-\log \mathbb{P}(\text{token} \mid h)\bigr)
\tag{S3}
```

so $`\lambda = 0`$ is pure edit distance (the closest string wins, however nonsensical) and
$`\lambda = 1`$ is pure fluency (the most likely sentence wins, however far from the input).

## 2. The corrector in three lines

If you already have a trained model on disk (a checkpoint directory holding a persistent
`vocabulary` and a bincode `model.bin`), correction is three lines — this path additionally needs
the **`serde-extras`** feature, which deserializes the n-gram model:

```rust
use libgrammstein::integration::corrector::{CorrectorConfig, HierarchicalCorrector};

let corrector = HierarchicalCorrector::from_checkpoint(
    std::path::Path::new("model_dir"),
    CorrectorConfig::default(),
)?;

let best = corrector.correct("teh quikc brwon fox");
assert_eq!(best[0].text, "the quick brown fox");
# Ok::<(), libgrammstein::integration::corrector::CorrectorError>(())
```

The file names inside the checkpoint directory are themselves configuration
(`CorrectorConfig::vocabulary_filename`, default `"vocabulary"`;
`CorrectorConfig::model_filename`, default `"model.bin"`), and a missing artifact surfaces as
`CorrectorError::MissingArtifact` rather than a panic.

## 3. How the pieces fit

![The spell-correction example as a code flow, in two phases. Phase one, assembled once at start-up: a corpus file is opened by PlaintextReader::from_file and fed to TrainerBuilder, producing an NgramModel whose keys use the legacy pipe-separated encoding; separately, create_vocabulary builds a SharedVocabARTrie spelling dictionary into which every correct word is inserted. HierarchicalCorrector::from_ngram_model combines the vocabulary and the model, wrapping the latter as a GrammsteinLanguageModel. Phase two, per input: the text is tokenized into a linear lattice with one zero-cost original edge per token; the LevenshteinCorrectionLayer expands it with correction edges whose cost is the edit distance, drawn by intersecting the automaton with the persistent vocabulary trie in place; an optional CfgFilterLayer prunes ungrammatical paths; the LanguageModelLayer reweights every edge as one minus lambda times the weight plus lambda times the negative log probability of the token given its history; and viterbi or nbest collapses the lattice to a ranked Vec of CorrectionResult, in which a lower score is better.](../diagrams/example-spell-correction.svg)

**Figure 1.** What the program builds once (①) and what happens on every call (②). The dashed
teal edge is the point of the design: the Levenshtein automaton **traverses the persistent
vocabulary trie in place** — no RAM copy of the vocabulary, which is what makes this viable at
Google-Books scale ([Dictionary Backend](../integration/lling-llang/dictionary-backend.md)).
*(Rendered from `docs/diagrams/example-spell-correction.puml`.)*

Three layers, applied in sequence to a lattice, each an implementation of lling-llang's
`CorrectionLayer` trait:

| Stage | Layer | Effect on the lattice | Bayes' role |
|---|---|---|---|
| 1 | `LevenshteinCorrectionLayer` | **expands** — adds one edge per candidate, weighted by $`d(x_i, w_i)`$ | channel $`\mathbb{P}(x \mid w)`$ |
| 2 | `CfgFilterLayer` *(optional)* | **prunes** — drops ungrammatical paths | a hard syntactic constraint |
| 3 | `LanguageModelLayer` | **reweights** — applies $`(\mathrm{S3})`$ to every edge | prior $`\mathbb{P}(w)`$ |
| 4 | `viterbi` / `nbest` | **collapses** — extracts the best path(s) | the $`\arg\min`$ of $`(\mathrm{S2})`$ |

Stage 2 is genuinely optional and *off* by default: there is no general-purpose English CFG to
default to. Attach one with `.with_grammar(grammar)` when you have a grammar worth enforcing (a
command language, a query DSL, a form field).

## 4. The complete program

This is [`examples/correct_sentence.rs`](../../examples/correct_sentence.rs), the one example that
ships pre-registered in `Cargo.toml`. It is entirely self-contained: it writes its own corpus,
trains its own model, seeds its own vocabulary, and corrects a sentence.

```rust
use std::io::Write;

use libdictenstein::pathmap::PathMapDictionary;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::integration::corrector::{CorrectorConfig, HierarchicalCorrector};
use libgrammstein::ngram::vocabulary::create_vocabulary;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};

fn main() {
    // A scratch directory for the corpus file and the persistent vocabulary.
    let dir = tempfile::TempDir::new().expect("create temp dir");

    // 1. Write a tiny training corpus. Repetition strengthens the in-vocabulary
    //    n-gram counts relative to the (unseen) misspellings.
    let sentence = "the quick brown fox jumps over the lazy dog";
    let corpus = format!("{sentence}\n").repeat(20);
    let corpus_path = dir.path().join("corpus.txt");
    {
        let mut file = std::fs::File::create(&corpus_path).expect("create corpus file");
        file.write_all(corpus.as_bytes()).expect("write corpus file");
    }

    // 2. Train a trigram model over a PathMap-backed dictionary. This model supplies
    //    the PRIOR P(w); it is trained in the default (legacy-key) mode — see §7.
    let reader = PlaintextReader::from_file(&corpus_path).expect("open corpus");
    let dictionary = PathMapDictionary::<NgramEntry>::new();
    let model = TrainerBuilder::new(dictionary)
        .order(3)
        .train(reader)
        .expect("train n-gram model");

    // 3. Seed the spelling vocabulary — the set V the corrector may OUTPUT. This is a
    //    separate artifact from the n-gram trie (§7) and drives the CHANNEL.
    let vocabulary = create_vocabulary(&dir.path().join("vocab")).expect("create vocabulary");
    for word in ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"] {
        vocabulary.insert(word).expect("insert vocabulary word");
    }

    // 4. Assemble the corrector and correct a misspelled sentence.
    let corrector =
        HierarchicalCorrector::from_ngram_model(vocabulary, model, CorrectorConfig::default());

    let input = "teh quikc brwon fox";
    let results = corrector.correct(input);

    println!("input:  {input:?}");
    match results.first() {
        Some(best) => println!("output: {:?}  (score {:.4})", best.text, best.score),
        None => println!("output: <no hypothesis>"),
    }
}
```

```sh
cargo run --example correct_sentence --features lling-llang-integration
```

```
input:  "teh quikc brwon fox"
output: "the quick brown fox"  (score 4.9835)
```

That score is the **total path cost** of the winning sentence — the sum, over the four tokens, of
each edge's $`(\mathrm{S3})`$ weight. It is a cost in the tropical semiring, so its magnitude is
not a probability and only its *ordering* against the competing hypotheses is meaningful.

**Why it works on a corpus this small.** `teh`, `quikc` and `brwon` are all absent from $`V`$, so
the spelling layer fires on each. `teh` → `the` is a *single* edit under the default
`Algorithm::Transposition` (adjacent transposition), the most common typo class
[[5]](#references) — not the two substitutions plain Levenshtein would charge. The prior then
prefers `the quick brown fox` over the other reachable strings because that is the only trigram
sequence the corpus ever showed it.

**The score is a cost, not a likelihood.** `CorrectionResult::score` is the tropical path weight:
$`\oplus = \min`$, $`\otimes = +`$. **Lower is better**, and results come back best-first.

## 5. Configuration

`CorrectorConfig` is the façade's single knob-bag; its spelling-relevant fields are projected into
an `EditConfig` for the Levenshtein layer.

| Field | Default | Meaning |
|---|---|---|
| `max_edit_distance` | `2` | the $`k`$ of the automaton. `1` is fast and timid; `3` explodes the candidate set on long words |
| `algorithm` | `Algorithm::Transposition` | Damerau-Levenshtein: adjacent transposition costs **one** edit, not two [[5]](#references). Alternatives: `Standard`, `MergeAndSplit` |
| `max_corrections_per_word` | `8` | cap on edges added per OOV token; candidates are kept by ascending distance (ties lexicographic) |
| `min_word_length` | `2` | tokens shorter than this are never corrected — the 1-character neighbourhood is dense and thrashy |
| `keep_original` | `true` | keep the input token as a competing edge. **Required** to correct real-word errors (§10) |
| `k_best` | `1` | `1` runs `viterbi`; `> 1` runs the lazy n-best enumeration |
| `lm_weight` | `0.75` | the $`\lambda`$ of $`(\mathrm{S3})`$ |
| `vocabulary_filename` | `"vocabulary"` | file name inside a checkpoint directory (`from_checkpoint`) |
| `model_filename` | `"model.bin"` | file name inside a checkpoint directory (`from_checkpoint`) |

```rust
use liblevenshtein::transducer::Algorithm;
use libgrammstein::integration::corrector::{CorrectorConfig, HierarchicalCorrector};

let config = CorrectorConfig {
    max_edit_distance: 1,          // tight: only single-edit typos
    algorithm: Algorithm::Standard,
    max_corrections_per_word: 4,
    lm_weight: 0.9,                // fluency dominates edit distance
    k_best: 5,                     // return a ranked list
    ..Default::default()
};

let corrector = HierarchicalCorrector::from_ngram_model(vocabulary, model, config);
```

**Why `lm_weight = 0.75`?** A correction edge always costs *more* than the original token's edge,
which the input lattice adds at zero cost. For a correction to win, the language-model term of
$`(\mathrm{S3})`$ must overcome that edit-distance handicap. At $`\lambda = 0.75`$ fluency
dominates — which is what spelling correction wants — while edit distance still breaks ties
between comparably fluent candidates. Push $`\lambda`$ toward $`1`$ and the corrector will happily
rewrite text that was never misspelled; pull it toward $`0`$ and it becomes a nearest-string
matcher with no sense of context.

## 6. k-best and confidence

Set `k_best > 1` to get a ranked list instead of a single answer:

```rust
let config = CorrectorConfig { k_best: 5, ..Default::default() };
let corrector = HierarchicalCorrector::from_ngram_model(vocabulary, model, config);

for (rank, candidate) in corrector.correct("teh quikc brwon fox").iter().enumerate() {
    println!("{}. {:?}  cost {:.4}", rank + 1, candidate.text, candidate.score);
}
```

Because the scores are **costs** (lower is better), the gap between the best and the runner-up is a
usable confidence signal: a large gap means the winner is unambiguous, a gap near zero means the
corrector is guessing between two comparable readings.

```rust
/// Map the best-vs-runner-up cost gap into a confidence in [0, 1).
/// `results` must be the best-first output of `HierarchicalCorrector::correct`.
fn confidence(results: &[libgrammstein::integration::corrector::CorrectionResult]) -> Option<(&str, f64)> {
    let best = results.first()?;
    match results.get(1) {
        // Tropical scores are costs: runner_up.score >= best.score, so the gap is >= 0
        // and 1 - exp(-gap) rises from 0 (a tie) toward 1 (an unambiguous win).
        Some(runner_up) => {
            let gap = runner_up.score - best.score;
            Some((best.text.as_str(), 1.0 - (-gap).exp()))
        }
        // Only one hypothesis exists: nothing to be uncertain between.
        None => Some((best.text.as_str(), 1.0)),
    }
}
```

Use it to gate: auto-apply above (say) $`0.8`$, offer a suggestion between $`0.4`$ and $`0.8`$,
stay silent below. Calibrate the thresholds on text you have ground truth for — the mapping above
is a monotone convenience, not a probability.

## 7. Two vocabularies, two jobs

This is the single most common source of confusion, so it gets its own section.

| Artifact | Type | Job | Built by |
|---|---|---|---|
| **spelling vocabulary** $`V`$ | `SharedVocabARTrie` | the set of words the corrector may **output**; the automaton walks it to enumerate candidates | `create_vocabulary(path)` + `insert(word)` |
| **n-gram dictionary** | `PathMapDictionary` / `DynamicDawgChar` / … | stores n-gram **counts**; supplies $`\mathbb{P}(w)`$ | `TrainerBuilder::new(dictionary)` |

They are *different objects with different contents*: one holds words, the other holds n-grams.
`HierarchicalCorrector::from_ngram_model(vocabulary, model, config)` takes both, and the example in
§4 builds them separately for exactly that reason.

> **Do not train the n-gram model in vocabulary-indexed mode.**
> `TrainerBuilder::with_vocabulary` / `with_vocabulary_path` switch the n-gram trie to
> **PUA-character** keys (one Private Use Area codepoint per word), while every query path used by
> the corrector — `NgramModel::log_prob` and `sentence_log_prob`, reached through
> `GrammsteinLanguageModel::score_continuation` / `score_sequence` — builds **legacy
> pipe-separated** keys (`"the|quick|brown"`). A vocabulary-indexed model would miss every lookup
> and silently back off to the uniform floor, leaving the corrector with a language model that has
> no opinion. Vocabulary-indexed mode belongs to the Google-Books import pipeline, whose lookups
> go through `encode_ngram_key`. The default mode — plain `TrainerBuilder::new(dict).order(n)`,
> exactly as in §4 — is the correct one here.

A practical consequence: **$`V`$ is what bounds the output.** A word missing from $`V`$ can never
be produced, no matter how fluent it would be. In production, seed $`V`$ from the same corpus that
trained the model (iterate its vocabulary and insert each word), then add domain terms.

## 8. A hybrid language model in the corrector

`from_ngram_model` is a convenience for the pure n-gram prior. The general constructor,
`from_parts`, accepts any `GrammsteinLanguageModel` — including a hybrid, whose subword embedding
gives the prior an opinion about words the n-gram has never seen
([OOV Handling](../components/hybrid/oov-handling.md)):

```rust
use libgrammstein::integration::{GrammsteinLanguageModel, corrector::{CorrectorConfig, HierarchicalCorrector}};

// `ngram` and `embedding` were trained in Train and Evaluate §5–§6.
// from_components fuses them with HybridConfig::default() (Linear, α = 0.8);
// use from_components_with_config to choose the strategy yourself.
let language_model = GrammsteinLanguageModel::from_components(ngram, embedding);

let corrector =
    HierarchicalCorrector::from_parts(vocabulary, language_model, CorrectorConfig::default());
```

`GrammsteinLanguageModel` is the adapter that makes a libgrammstein model *look like* an
lling-llang language model. It is an enum over the two shapes, and it is where `score_sequence` and
`score_continuation` actually live — they are methods of lling-llang's `LanguageModel` **trait**,
not inherent methods of `HybridLanguageModel`:

```rust
use lling_llang::layers::LanguageModel;   // the trait must be in scope to call its methods
use libgrammstein::integration::GrammsteinLanguageModel;

let lm = GrammsteinLanguageModel::from_ngram(model);

let sequence = lm.score_sequence(&["the", "quick", "brown", "fox"]); // Σ log P over the sentence
let next     = lm.score_continuation(&["the", "quick"], "brown");    // log P(brown | the quick)
let size     = lm.vocab_size();

assert!(lm.ngram_model().is_some());
assert!(!lm.is_hybrid());
```

| Constructor | Prior |
|---|---|
| `from_ngram(model)` / `from_ngram_arc(arc)` | pure MKN n-gram |
| `from_hybrid(model)` / `from_hybrid_arc(arc)` | an already-built `HybridLanguageModel` |
| `from_components(ngram, embedding)` | builds the hybrid for you with `HybridConfig::default()` |
| `from_components_with_config(ngram, embedding, config)` | …with a strategy you choose |

Cloning a `GrammsteinLanguageModel` is an `Arc` bump, so one model can be shared across correction
threads without duplication.

## 9. Building your own pipeline

The façade fixes three choices: the tropical semiring, the `HashMapBackend` lattice backend, and a
`SharedVocabARTrie` spelling dictionary. Underneath, the pieces are generic and you can assemble
them yourself — a different semiring, a different backend, a different dictionary, extra layers:

```rust
use libgrammstein::integration::{EditConfig, GrammsteinLanguageModel, LevenshteinCorrectionLayer};
use lling_llang::backend::HashMapBackend;
use lling_llang::layers::rescoring::LanguageModelLayer;
use lling_llang::semiring::TropicalWeight;

// The spelling layer is generic over <semiring W, lattice backend B, dictionary D>.
// D is ANY libdictenstein `Dictionary`: the persistent vocabulary trie here, but a
// DynamicDawgChar or DoubleArrayTrie works just as well for a small, static word list.
let spelling: LevenshteinCorrectionLayer<TropicalWeight, HashMapBackend, _> =
    LevenshteinCorrectionLayer::new(vocabulary.clone(), EditConfig::default());

// The rescoring layer accepts any `lling_llang::layers::LanguageModel` — ours included.
let rescore = LanguageModelLayer::new(Box::new(GrammsteinLanguageModel::from_ngram(model)))
    .with_weight(0.75);   // the λ of (S3)
```

The automaton inside `LevenshteinCorrectionLayer` is built **once**, at layer construction, and
reused for every `apply` — it is never rebuilt per word. The layer never mutates its input either:
`apply` returns a fresh lattice that clones the backend and copies the original edges.

For the lattice construction, layer application and decoding that go around these two layers, read
`HierarchicalCorrector::correct` in
[`src/integration/corrector.rs`](../../src/integration/corrector.rs) — it is forty lines, and it
is the reference implementation of the pattern. The design rationale, the other dimensions
(phonetic, semantic) and the alternative WFST-composition route are laid out in
[Hierarchical Correction](../integration/lling-llang/hierarchical-correction.md) and
[Pipeline Assembly](../integration/lling-llang/pipeline-assembly.md).

## 10. Pitfalls

| Pitfall | Why | Fix |
|---|---|---|
| Nothing gets corrected | the token is **in** $`V`$ — the spelling layer only fires on OOV tokens | that is by design; real-word errors are handled by the prior, and need `keep_original: true` plus competing candidates |
| Real-word errors survive (*"to"* vs *"two"*) | a zero-edit token always has a zero-cost edge | raise `lm_weight`; ensure `keep_original` is `true` so the alternatives compete on fluency |
| 1–2 character typos ignored | `min_word_length` is `2` by default | lower it — and expect noise, since short words have dense neighbourhoods |
| Output words you never wanted | anything in $`V`$ is reachable | curate $`V`$: it is the output alphabet, not a suggestion list |
| Candidate explosion / slowness | `max_edit_distance: 3` on long words is combinatorial | keep $`k \leq 2`$; cap with `max_corrections_per_word` |
| Sorting results by "highest score" | scores are **costs** under $`(\min, +)`$ | results are already best-first; lower is better |
| `from_checkpoint` won't compile | it is gated on `serde-extras` | enable it, or build in memory with `from_ngram_model` / `from_parts` |
| The LM has no opinion | the model was trained in vocabulary-indexed mode | retrain in the default legacy mode (§7) |

## References

1. C. E. Shannon (1948). *A mathematical theory of communication.* Bell System Technical Journal
   27(3), 379–423.
   [doi:10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)
2. M. D. Kernighan, K. W. Church & W. A. Gale (1990). *A spelling correction program based on a
   noisy channel model.* COLING '90, 205–210.
   [doi:10.3115/997939.997975](https://doi.org/10.3115/997939.997975)
3. M. Mohri, F. Pereira & M. Riley (2002). *Weighted finite-state transducers in speech
   recognition.* Computer Speech & Language 16(1), 69–88.
   [doi:10.1006/csla.2001.0184](https://doi.org/10.1006/csla.2001.0184)
4. E. Brill & R. C. Moore (2000). *An improved error model for noisy channel spelling correction.*
   ACL '00, 286–293. [doi:10.3115/1075218.1075255](https://doi.org/10.3115/1075218.1075255)
5. F. J. Damerau (1964). *A technique for computer detection and correction of spelling errors.*
   Communications of the ACM 7(3), 171–176.
   [doi:10.1145/363958.363994](https://doi.org/10.1145/363958.363994)
6. K. U. Schulz & S. Mihov (2002). *Fast string correction with Levenshtein automata.* IJDAR 5(1),
   67–85. [doi:10.1007/s10032-002-0082-8](https://doi.org/10.1007/s10032-002-0082-8)

## See also

- [Hierarchical Correction](../integration/lling-llang/hierarchical-correction.md) — the full $`N`$-dimensional theory behind this page
- [Pipeline Assembly](../integration/lling-llang/pipeline-assembly.md) — the `HierarchicalCorrector` API in detail
- [Dictionary Backend](../integration/lling-llang/dictionary-backend.md) — how the automaton is fed from the persistent trie
- [Correction Dimensions](../integration/lling-llang/dimensions.md) — orthographic, phonetic, syntactic, contextual, semantic
- [Train and Evaluate](train-and-evaluate.md) — building the prior this page consumes
- [OOV Handling](../components/hybrid/oov-handling.md) — what the embedding adds when the n-gram has never seen a word
