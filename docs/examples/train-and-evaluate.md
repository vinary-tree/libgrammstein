# Example: Train and Evaluate a Language Model

This is the **canonical end-to-end walkthrough**: read a corpus, train a Modified Kneser-Ney
n-gram model, train a subword embedding, fuse the two into a hybrid, measure the result with
perplexity, query it, and persist it. Every snippet on this page compiles against the shipped
API — no pseudo-API, no invented constructors — and §11 assembles them into one runnable
program.

> **Scope.** Source of truth:
> [`src/ngram/trainer.rs`](../../src/ngram/trainer.rs),
> [`src/embedding/trainer.rs`](../../src/embedding/trainer.rs),
> [`src/hybrid/model.rs`](../../src/hybrid/model.rs),
> [`src/scoring/perplexity.rs`](../../src/scoring/perplexity.rs) and
> [`src/corpus/plaintext.rs`](../../src/corpus/plaintext.rs).
> For the *theory* behind each stage see [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md),
> [Subword Embeddings](../components/embedding/overview.md) and
> [Hybrid Interpolation](../components/hybrid/interpolation.md). For a deeper treatment of
> perplexity itself, see [Perplexity Scoring](perplexity-scoring.md).

## Notation

Every symbol is defined before it is used.

| Symbol | Meaning |
|---|---|
| $`w`$ | a word (token) |
| $`h`$ | the *history* (context) — the words preceding $`w`$ |
| $`n`$ | the n-gram **order**: the maximum length of $`h\,w`$ (so $`\lvert h \rvert \leq n - 1`$) |
| $`\mathbb{P}(w \mid h)`$ | the probability the model assigns to $`w`$ after $`h`$ |
| $`\mathbb{P}_n`$ | the **n-gram** expert's probability (Modified Kneser-Ney) |
| $`\mathbb{P}_e`$ | the **embedding** expert's (unnormalized) probability |
| $`\alpha`$ | the hybrid interpolation weight — the mass given to $`\mathbb{P}_n`$ |
| $`N`$ | the number of tokens in the evaluation corpus |
| $`\lvert V \rvert`$ | vocabulary size (distinct unigrams seen in training) |
| $`\mathrm{PP}(W)`$ | the perplexity of a token sequence $`W = w_1 \ldots w_N`$ |
| $`D`$ | the Rust type parameter naming the **dictionary backend** (trie) |

**Acronyms.** *MKN* — Modified Kneser-Ney; *OOV* — Out-Of-Vocabulary; *PP* — perplexity;
*DAWG* — Directed Acyclic Word Graph.

## 1. What you will build

Four artifacts, in dependency order:

| # | Artifact | Rust type | Built by |
|---|---|---|---|
| 1 | corpus stream | `PlaintextReader` | `PlaintextReader::from_file` |
| 2 | n-gram model | `NgramModel<D>` | `TrainerBuilder::new(dict).order(n).train(reader)` |
| 3 | subword embedding | `SubwordEmbedding` | `EmbeddingTrainerBuilder::new()…train(reader)` |
| 4 | hybrid model | `HybridLanguageModel<D>` | `HybridLanguageModel::new(ngram, embedding, config)` |

The n-gram model is sharp but brittle: it knows exactly what it saw and nothing else. The
embedding is blurry but robust: it can place a word it has never seen, because a word's vector is
composed from its character n-grams. The hybrid interpolates the two so each covers the other's
blind spot — see [Hybrid Interpolation](../components/hybrid/interpolation.md) for the four
strategies and the mathematics.

## 2. The workflow

![The train-fuse-evaluate-persist workflow. A corpus file on disk is opened by PlaintextReader::from_file, which normalizes (NFC, control characters, whitespace) and tokenizes (lowercasing, splitting sentences on terminal punctuation). One reader is moved by value into TrainerBuilder, which counts n-grams in parallel with Rayon over atomic entries, collects continuation counts, and derives the Modified Kneser-Ney discounts, producing an NgramModel. A second reader is moved into EmbeddingTrainerBuilder, which runs skip-gram training with negative sampling to produce a SubwordEmbedding. HybridLanguageModel::new fuses the two experts under an InterpolationStrategy with a lock-free score cache. The n-gram model feeds Perplexity for a baseline score, the hybrid is scored per sentence, and both can be persisted with the serde-extras feature.](../diagrams/example-train-eval.svg)

**Figure 1.** Train → fuse → evaluate → persist. Note the two arrows out of the reader: a
`CorpusReader` is **consumed** by `train`, so each training pass needs its own reader.
*(Rendered from `docs/diagrams/example-train-eval.puml`.)*

## 3. Setup

```toml
[dependencies]
# The dictionary backend lives in libdictenstein; libgrammstein is generic over it.
libgrammstein = { version = "0.2", features = ["serde-extras"] }
libdictenstein = { version = "0.2", features = ["persistent-artrie"] }

[dev-dependencies]
tempfile = "3.14"   # only for the self-contained example in §11
```

Within this workspace the three crates (`libgrammstein`, `libdictenstein`,
`liblevenshtein`) are **path** dependencies of one another; the versions above are what a
published consumer would pin.

| Feature | Enables |
|---|---|
| *(none)* | training, querying, perplexity — everything in §4–§9 |
| `serde-extras` | `save` / `load` and `save_portable` / `load_portable` (§10) |
| `lling-llang-integration` | `GrammsteinLanguageModel`, `HierarchicalCorrector` — see [Spell Correction](spell-correction.md) |
| `cli` | the `grammstein` binary — see the [CLI guide](../cli/README.md) |

**Choosing a dictionary backend.** `TrainerBuilder` is generic over any `D` implementing
`MutableMappedDictionary<Value = NgramEntry> + IterableDictionary + Send + Sync`:

| Backend (`libdictenstein`) | Use when | serde |
|---|---|---|
| `dynamic_dawg::char::DynamicDawgChar<NgramEntry>` | general purpose; you want `save` / `load` | yes (with `serde-extras`) |
| `pathmap::PathMapDictionary<NgramEntry>` | fast in-memory training, no serialization needed | no — use `save_portable` |
| persistent ARTrie | Google-Books-scale, memory-mapped, evictable | via checkpoints |

This page uses `DynamicDawgChar` because §10 serializes the model. See
[Backend Selection](../integration/liblevenshtein/backend-selection.md) for the full comparison.

## 4. Step 1 — Read the corpus

```rust
use libgrammstein::corpus::{CorpusReader, PlaintextReader};

let reader = PlaintextReader::from_file("train.txt")?;      // one file
// let reader = PlaintextReader::from_directory("corpus/")?; // …or every file in a directory

for sentence in reader.sentences() {
    println!("{sentence}");
}
# Ok::<(), std::io::Error>(())
```

`PlaintextReader::from_file` wires up two stages that you inherit *by default*, and whose exact
behavior determines what a "token" is for the rest of this page:

1. **`Normalizer::new()`** — Unicode NFC, control-character removal, whitespace collapsing, trim.
   It does **not** remove punctuation.
2. **`Tokenizer::new()`** — splits sentences on the regular expression `[.!?]+\s+`, discards
   sentences shorter than **10 characters**, and **lowercases**. Word tokenization (used during
   training) splits on `[\s\p{P}]+`, so punctuation is stripped from *words* but not from the
   *sentence* strings that `sentences()` yields.

That asymmetry matters when you evaluate; it is dissected in
[Perplexity Scoring §5](perplexity-scoring.md#5-the-tokenization-contract). Override either stage
with `with_normalizer` / `with_tokenizer` — for example, to keep case:

```rust
use libgrammstein::corpus::{PlaintextReader, Tokenizer};

let reader = PlaintextReader::from_file("train.txt")?
    .with_tokenizer(Tokenizer::new().with_lowercase(false));
# Ok::<(), std::io::Error>(())
```

Other readers implement the same `CorpusReader` trait and are drop-in substitutes:
`GutenbergReader` (boilerplate-stripping) and `WikipediaReader` (streaming XML dumps). See
[Corpus Formats](../components/corpus/formats.md).

## 5. Step 2 — Train the n-gram model

```rust
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};

let reader = PlaintextReader::from_file("train.txt")?;

let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
    .order(3)            // trigram: |h| ≤ 2
    .batch_size(10_000)  // sentences per parallel batch (default 10_000)
    .train(reader)?;     // ← the reader is MOVED, not borrowed

println!("{} words, {} n-grams", model.vocab_size(), model.ngram_count());
# Ok::<(), libgrammstein::Error>(())
```

`train` performs three phases (`NgramTrainer::train`):

1. **Count.** Sentences are prefetched into batches and counted in parallel with Rayon; each
   `NgramEntry` holds atomics, so workers never take a lock
   ([Threading Model](../architecture/threading.md)).
2. **Continuation counts.** A second pass computes $`N_{1+}(\bullet, w)`$ (how many distinct
   histories precede $`w`$) and $`N_{1+}(h, \bullet)`$ (how many distinct words follow $`h`$) —
   the statistics that make Kneser-Ney *Kneser-Ney*.
3. **Discounts.** The count-of-counts $`n_1, n_2, n_3, n_4`$ yield $`D_1, D_2, D_{3+}`$ by the
   Chen & Goodman formulae [[1]](#references); if the corpus is too small for all four counts to
   be non-zero, fixed defaults are used instead.

Smoothing is **always on** and always MKN — there is no "unsmoothed" mode, because an unsmoothed
model assigns $`\log \mathbb{P} = -\infty`$ to any unseen n-gram.

**Order.** $`n = 3`$ is a sound default for corpora of a few million tokens; $`n = 5`$ (the
`TrainingConfig` default) pays off only when the corpus is large enough to populate 5-grams.
§4 of [Perplexity Scoring](perplexity-scoring.md#4-choosing-the-order) shows how to *measure*
the right order rather than guess it.

## 6. Step 3 — Train the subword embedding

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;

let reader = PlaintextReader::from_file("train.txt")?;   // a FRESH reader — the n-gram pass consumed the first

let embedding = EmbeddingTrainerBuilder::new()
    .dim(64)          // vector width      (default 100)
    .window_size(3)   // context radius    (default 5)
    .min_count(1)     // vocabulary cutoff (default 5) — APPLIED: words below this are dropped
    .epochs(10)       // passes over the corpus (default 5)
    .train(reader)?;

println!("{} vectors of width {}", embedding.vocab_size(), embedding.dim());
# Ok::<(), libgrammstein::Error>(())
```

Training is skip-gram with negative sampling [[3]](#references), and every word vector is the sum
of its **hashed character n-grams** (lengths 3–6 by default) — the FastText construction
[[4]](#references). That is precisely why the embedding can score an OOV word: *misspellling*
shares most of its character n-grams with *misspelling*.

For corpora too large to buffer, use the streaming form, which re-opens the corpus once per
epoch instead of holding it in memory:

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;

let path = std::path::PathBuf::from("train.txt");
let embedding = EmbeddingTrainerBuilder::new()
    .dim(100)
    .epochs(5)
    .train_streaming(|| Ok(PlaintextReader::from_file(&path)?))?;
# Ok::<(), libgrammstein::Error>(())
```

## 7. Step 4 — Fuse into a hybrid model

```rust
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};

let config = HybridConfig {
    strategy: InterpolationStrategy::Linear { alpha: 0.7 },
    ..Default::default()   // cache_size 50_000 · embedding_smoothing 1e-8 · temperature 1.0
};

// `ngram` is cloned because we keep the pure n-gram model for the baseline in §8;
// `embedding` is MOVED — SubwordEmbedding is not Clone (see §12).
let hybrid = HybridLanguageModel::new(ngram.clone(), embedding, config);

// Reach the components back out of the hybrid whenever you need them:
let similar = hybrid.embedding_model().most_similar("language", 5);
let vocab   = hybrid.ngram_model().vocab_size();
```

> **`NgramModel::clone` is cheap.** The trie lives behind an `Arc`, so cloning a model is a
> refcount bump, not a copy of the n-gram store — the clone *shares* the trie. Models are
> read-only after training, so sharing is exactly what you want. `SubwordEmbedding`, by contrast,
> is not `Clone` at all; the hybrid takes ownership of it.

The `Linear` strategy computes

```math
\begin{array}{lr}
\displaystyle \mathbb{P}(w \mid h) = \alpha\,\mathbb{P}_n(w \mid h) + (1 - \alpha)\,\mathbb{P}_e(w \mid h) & \text{(T1)}
\end{array}
```

and `HybridLanguageModel::score` returns its **natural logarithm**. `HybridConfig::default()`
uses `Linear { alpha: 0.8 }`; the other three strategies (`LogLinear`,
`NgramWithEmbeddingFallback`, `Dynamic`) and the definition of $`\mathbb{P}_e`$ are covered in
[Hybrid Interpolation](../components/hybrid/interpolation.md).

> **$`\mathbb{P}_e`$ is a score, not a calibrated probability.** The embedding side turns cosine
> similarity into a pseudo-probability without the vocabulary-wide softmax normalizer. It is
> designed for *ranking* and *interpolation*; the n-gram side supplies calibration. Perplexities
> computed from hybrid scores are therefore comparable to *each other*, but a hybrid PP and a
> pure-n-gram PP are not on an identical footing — see §8.

## 8. Step 5 — Evaluate with perplexity

Perplexity is the exponentiated per-token cross-entropy: the model's average branching factor.
Lower is better.

```math
\begin{array}{lr}
\displaystyle \mathrm{PP}(W) \;=\; \exp\!\Bigl(-\tfrac{1}{N} \sum_{i=1}^{N} \log \mathbb{P}(w_i \mid h_i)\Bigr) & \text{(T2)}
\end{array}
```

`Perplexity` evaluates an **`NgramModel`** over a whole corpus in one streaming pass:

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::scoring::Perplexity;

let test_reader = PlaintextReader::from_file("test.txt")?;
let result = Perplexity::new(&model).corpus_perplexity(&test_reader)?;

println!("{result}");                                  // Display: PP | tokens | OOV% | sentences
println!("PP        = {:.2}", result.perplexity);
println!("OOV rate  = {:.1}%", result.oov_rate * 100.0);
println!("tokens    = {}",     result.total_tokens);
# Ok::<(), libgrammstein::Error>(())
```

The `HybridLanguageModel` has **its own** `perplexity` method, taking an already-tokenized
sentence, because `Perplexity` is typed against `NgramModel`:

```rust
let tokens: Vec<&str> = "the quick brown fox".split_whitespace().collect();

let ngram_pp  = (-model.sentence_log_prob(&tokens) / tokens.len() as f64).exp();
let hybrid_pp = hybrid.perplexity(&tokens);   // same formula (T2), hybrid scores

println!("n-gram {ngram_pp:.2} · hybrid {hybrid_pp:.2}");
```

Aggregating that over a corpus (and comparing the two fairly) is the subject of
[Perplexity Scoring §6](perplexity-scoring.md#6-n-gram-versus-hybrid).

## 9. Step 6 — Query the models

```rust
// Conditional log-probability, log P(fox | quick brown).
let log_p = model.log_prob("fox", &["quick", "brown"]);

// Whole-sentence log-probability (sums per-token log-probs with growing context).
let log_p_sentence = model.sentence_log_prob(&["the", "quick", "brown", "fox"]);

// Raw corpus count of an n-gram, and vocabulary membership.
let count = model.count(&["quick", "brown"]);
let known = model.in_vocabulary("fox");

// The hybrid's fused log-score, robust when "brown" is rare or unseen.
let log_p_hybrid = hybrid.score("brown", &["the", "quick"]);

// Rank a candidate set — returns the best (word, score) pair.
let best = hybrid.predict_next(&["the", "quick"], &["brown", "brwon", "purple"]);

// Nearest neighbours in embedding space.
for (word, cosine) in hybrid.embedding_model().most_similar("language", 5) {
    println!("{word}: {cosine:.4}");
}
```

All log-probabilities are **natural logs** and are always finite: the MKN recursion terminates in
the strictly positive unigram term $`1 / \lvert V \rvert`$, and every interpolation weight is
non-negative, so the fused probability is always $`> 0`$
([Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md)). Note that the *final* value
for an unseen word typically lands **below** $`\log(1/\lvert V \rvert)`$, because the backoff
weight $`\lambda(h) < 1`$ scales that uniform term down — which is exactly why the OOV test in
`Perplexity` is $`\leq`$ and not $`=`$
([Perplexity Scoring §3](perplexity-scoring.md#3-the-evaluation-loop)). Contexts longer than
$`n - 1`$ are *not* an error — the model simply uses the most recent $`n - 1`$ words.

## 10. Step 7 — Persist and reload

Persistence is gated on the **`serde-extras`** feature. Two families exist, and the difference is
the dictionary backend:

| Method | Requires | Notes |
|---|---|---|
| `save` / `load` | `D: Serialize + DeserializeOwned` | direct bincode; `DynamicDawgChar` qualifies |
| `save_portable` / `load_portable` | *any* backend | exports `(key, entry)` pairs; rebuilds the trie through a factory closure |

With `model` and `hybrid` as built in §5 and §7:

```rust
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libgrammstein::hybrid::HybridLanguageModel;
use libgrammstein::ngram::{NgramEntry, NgramModel};

// Direct — the backend is serde-able.
model.save("ngram.bin")?;
let reloaded: NgramModel<DynamicDawgChar<NgramEntry>> = NgramModel::load("ngram.bin")?;

// Portable — works for any backend; the closure rebuilds an empty trie to fill.
hybrid.save_portable("hybrid.bin")?;
let reloaded_hybrid: HybridLanguageModel<DynamicDawgChar<NgramEntry>> =
    HybridLanguageModel::load_portable("hybrid.bin", DynamicDawgChar::new)?;

// The embedding serializes on its own too.
hybrid.embedding_model().save("embedding.bin")?;
# Ok::<(), libgrammstein::Error>(())
```

A reloaded model is *numerically identical* to the original: the counts, the discounts and
$`N_{1+}(\bullet,\bullet)`$ all round-trip. The hybrid's score cache is **not** serialized; it is
reconstructed empty, so the first queries after a load are cache misses.

## 11. The complete program

Everything above, in one file. It writes its corpora into a temporary directory so it runs
anywhere with no fixtures.

```rust
use std::io::Write;

use libdictenstein::dynamic_dawg::char::DynamicDawgChar;
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::embedding::EmbeddingTrainerBuilder;
use libgrammstein::hybrid::{HybridConfig, HybridLanguageModel, InterpolationStrategy};
use libgrammstein::ngram::{NgramEntry, TrainerBuilder};
use libgrammstein::scoring::Perplexity;

/// Repetition raises the counts so the tiny corpus can support a trigram model.
const TRAIN: &str = "\
the quick brown fox jumps over the lazy dog.
natural language processing lets computers understand text.
machine learning is transforming how we process language.
the brown dog chased the quick fox.
language models predict the next word in a sequence.
deep learning has reshaped natural language processing.
";

const TEST: &str = "\
the quick fox jumped over the fence.
language processing helps computers understand people.
";

fn main() -> libgrammstein::Result<()> {
    // ---- 0. materialize the corpora -------------------------------------------------
    let dir = tempfile::TempDir::new()?;
    let train_path = dir.path().join("train.txt");
    let test_path = dir.path().join("test.txt");
    {
        let mut file = std::fs::File::create(&train_path)?;
        // Repeat the corpus so n-gram counts exceed the MKN discounts.
        for _ in 0..20 {
            file.write_all(TRAIN.as_bytes())?;
        }
        std::fs::write(&test_path, TEST)?;
    }

    // ---- 1. train the n-gram model (pass 1) -----------------------------------------
    let ngram = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(3)
        .train(PlaintextReader::from_file(&train_path)?)?;
    println!(
        "n-gram : {} words · {} n-grams · order {}",
        ngram.vocab_size(),
        ngram.ngram_count(),
        ngram.order()
    );

    // ---- 2. train the embedding (pass 2, a fresh reader) ----------------------------
    let embedding = EmbeddingTrainerBuilder::new()
        .dim(64)
        .window_size(3)
        .min_count(1)
        .epochs(10)
        .train(PlaintextReader::from_file(&train_path)?)?;
    println!(
        "embed  : {} vectors · width {}",
        embedding.vocab_size(),
        embedding.dim()
    );

    // ---- 3. fuse (the embedding is moved; the n-gram is cloned for the baseline) -----
    let hybrid = HybridLanguageModel::new(
        ngram.clone(),
        embedding,
        HybridConfig {
            strategy: InterpolationStrategy::Linear { alpha: 0.7 },
            ..Default::default()
        },
    );

    // ---- 4. evaluate -----------------------------------------------------------------
    let report = Perplexity::new(&ngram).corpus_perplexity(&PlaintextReader::from_file(&test_path)?)?;
    println!("n-gram PP on held-out corpus → {report}");

    // ---- 5. query --------------------------------------------------------------------
    for (word, context) in [
        ("fox", &["quick", "brown"][..]),
        ("dog", &["lazy"][..]),
        ("zyzzyva", &["the"][..]), // OOV: the n-gram floors out, the hybrid does not
    ] {
        println!(
            "  log P({word:>8} | {context:?}) → n-gram {:>8.4} · hybrid {:>8.4}",
            ngram.log_prob(word, context),
            hybrid.score(word, context)
        );
    }

    // ---- 6. persist and verify -------------------------------------------------------
    let model_path = dir.path().join("ngram.bin");
    ngram.save(&model_path)?;
    let reloaded: libgrammstein::ngram::NgramModel<DynamicDawgChar<NgramEntry>> =
        libgrammstein::ngram::NgramModel::load(&model_path)?;
    assert!(
        (ngram.log_prob("fox", &["brown"]) - reloaded.log_prob("fox", &["brown"])).abs() < 1e-12,
        "a reloaded model must score identically"
    );
    println!("round-trip verified");

    Ok(())
}
```

Step 6 needs `serde-extras`, so build with it enabled. Drop the program into your own crate's
`src/main.rs` and `cargo run --features serde-extras`; to run it *inside this repository*, save
it as `examples/train_and_evaluate.rs` and register the example in `Cargo.toml`:

```toml
[[example]]
name = "train_and_evaluate"
required-features = ["serde-extras"]
```

```sh
cargo run --example train_and_evaluate --features serde-extras
```

The one example that ships pre-registered is
[`examples/correct_sentence.rs`](../../examples/correct_sentence.rs) — the subject of
[Spell Correction](spell-correction.md).

A run of the program above prints something close to this — the exact figures are discussed below:

```
n-gram : 33 words · 105 n-grams · order 3
embed  : 34 vectors · width 64
n-gram PP on held-out corpus → Perplexity: 79.40 | Tokens: 13 | OOV: 46.15% | Sentences: 2
  log P(     fox | ["quick", "brown"]) → n-gram  -0.0251 · hybrid  -0.0704
  log P(     dog | ["lazy"]) → n-gram  -0.0459 · hybrid  -0.1504
  log P( zyzzyva | ["the"]) → n-gram  -7.0031 · hybrid  -2.1489
round-trip verified
```

**How to read it.**

- **The n-gram half is deterministic.** Counts, discounts and every `log_prob` are reproducible
  bit-for-bit across runs, which is why the round-trip assertion is safe to write as an equality
  within $`10^{-12}`$.
- **The embedding half is stochastic.** Random initialization and negative sampling, with no seed
  knob on the builder, so the hybrid columns and `most_similar` shift a little from run to run.
- **The OOV gap is the point.** `zyzzyva` is unseen, so the n-gram can only hand it the discounted
  uniform term — $`-7.00`$, well below $`\log(1/33) = -3.50`$, because the backoff weight scales
  the uniform mass down. The hybrid rescues it to $`-2.15`$ purely from the word's character
  n-grams. That five-nat gap *is* the value the embedding adds.
- **That 46% OOV rate is a tokenization artifact, not a vocabulary failure.** Six of the thirteen
  test tokens are OOV, and two of them are only OOV because they carry a trailing period
  (`fence.`, `people.`) — the training tokenizer strips punctuation, the evaluator's whitespace
  split does not. This is the single most common source of inflated perplexity, and it is dissected
  in [Perplexity Scoring §5](perplexity-scoring.md#5-the-tokenization-contract).
- **Absolute perplexity on a corpus this small is meaningless.** It becomes meaningful the moment
  you compare two models on the *same* held-out corpus with the *same* tokenization.

## 12. Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Reusing a reader | `use of moved value: reader` | `train` takes the reader **by value**; build a fresh `PlaintextReader` per pass |
| `SubwordEmbedding` is not `Clone` | `no method named clone` | the hybrid **owns** the embedding; read it back with `hybrid.embedding_model()`, or `save`/`load` it to get a second copy |
| Importing from the prelude | `unresolved import` | the prelude exports `NgramModel`, `NgramEntry`, `CorpusReader`, `Perplexity`, `Error`, `Result` — **not** `TrainerBuilder`, `EmbeddingTrainerBuilder` or `HybridLanguageModel`; import those from their modules |
| `save` / `load` without the feature | `no method named save` | enable `serde-extras`; for a non-serde backend use `save_portable` / `load_portable` |
| Short lines vanish | fewer sentences than lines | `Tokenizer::new()` drops sentences under **10 characters**; relax with `with_min_sentence_length(0)` |
| `min_word_freq` seems inert | rare words still present | it **is** inert: the n-gram trainer stores the knob but does not yet filter on it. The embedding trainer's `min_count` *is* applied |
| Comparing PP across models | nonsensical improvements | perplexity is only comparable when the vocabulary **and** the tokenization are identical — see [Perplexity Scoring §9](perplexity-scoring.md#9-pitfalls) |

## References

1. S. F. Chen & J. Goodman (1999). *An empirical study of smoothing techniques for language
   modeling.* Computer Speech & Language 13(4), 359–393.
   [doi:10.1006/csla.1999.0128](https://doi.org/10.1006/csla.1999.0128)
2. R. Kneser & H. Ney (1995). *Improved backing-off for M-gram language modeling.* ICASSP '95,
   181–184. [doi:10.1109/ICASSP.1995.479394](https://doi.org/10.1109/ICASSP.1995.479394)
3. T. Mikolov, I. Sutskever, K. Chen, G. Corrado & J. Dean (2013). *Distributed representations of
   words and phrases and their compositionality.* NeurIPS 26.
   [doi:10.48550/arXiv.1310.4546](https://doi.org/10.48550/arXiv.1310.4546)
4. P. Bojanowski, E. Grave, A. Joulin & T. Mikolov (2017). *Enriching word vectors with subword
   information* (FastText). TACL 5, 135–146.
   [doi:10.1162/tacl_a_00051](https://doi.org/10.1162/tacl_a_00051)
5. F. Jelinek, R. L. Mercer, L. R. Bahl & J. K. Baker (1977). *Perplexity — a measure of the
   difficulty of speech recognition tasks.* JASA 62(S1), S63.
   [doi:10.1121/1.2016299](https://doi.org/10.1121/1.2016299)

## See also

- [Perplexity Scoring](perplexity-scoring.md) — evaluation in depth: OOV accounting, order sweeps, quality filtering
- [Domain Adaptation](domain-adaptation.md) — mixing a general and a domain model
- [Spell Correction](spell-correction.md) — putting the model to work in a correction pipeline
- [Modified Kneser-Ney](../components/ngram/modified-kneser-ney.md) — the smoothing at the core of step 2
- [Hybrid Interpolation](../components/hybrid/interpolation.md) — the four fusion strategies of step 4
- [N-gram Training](../training/ngram.md) · [Embedding Training](../training/embedding.md) · [Hybrid Training](../training/hybrid.md)
- [Hyperparameters](../training/hyperparameters.md) — what to tune, and in what order
