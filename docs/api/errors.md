# Error Types Reference

libgrammstein has a single crate-level error type, `Error`, and the usual `Result<T>` alias.
`Error` is a `thiserror` enum whose variants divide cleanly into two groups: those the crate
**produces** (from I/O, from an empty corpus, from bincode) and those that exist only as
**`From` conversion hooks** for downstream code. This page enumerates the real variants, says
honestly which are live, and shows the handling patterns.

> **Scope.** Source of truth: the `error` module inside
> [`src/lib.rs`](../../src/lib.rs) — `Error` and `Result` are defined there, not in a separate
> `src/error.rs` — and [`src/cli/error.rs`](../../src/cli/error.rs) for the CLI's own type.

## `Error` and `Result`

```rust
pub use libgrammstein::{Error, Result};   // also at libgrammstein::error::{Error, Result}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// I/O error during corpus reading or model loading.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// XML parsing error (Wikipedia dump).
    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    /// Invalid n-gram order (must be >= 1).
    #[error("Invalid n-gram order: {0} (must be >= 1)")]
    InvalidOrder(usize),

    /// Empty corpus provided for training.
    #[error("Empty corpus: no sentences found")]
    EmptyCorpus,

    /// Model not trained.
    #[error("Model not trained: {0}")]
    NotTrained(String),

    /// Serialization error (bincode).
    #[cfg(feature = "serde-extras")]
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// Serialization error (general, e.g. JSON).
    #[error("Serialization error: {0}")]
    SerializationMessage(String),

    /// Neural model error.
    #[cfg(feature = "neural-rescore")]
    #[error("Neural error: {0}")]
    Neural(#[from] crate::neural::NeuralError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

Two variants are **feature-gated**: `Serialization` (`serde-extras`) and `Neural`
(`neural-rescore`). A `match` on `Error` in a crate that does not enable them will not see them,
so a wildcard arm is required for forward compatibility either way.

## Which variants are actually produced

| Variant | Live? | Produced by |
|---|---|---|
| `Io` | **yes** | Implicitly, via `?` on any `std::io` operation inside a function returning `Result<T>` — `NgramModel::save`/`load`, `SubwordEmbedding::save`/`load`, and `PlaintextReader::from_file(…)?`. |
| `EmptyCorpus` | **yes** | `NgramTrainer` (no batches arrived), `EmbeddingTrainer` / `BpeTrainer` (empty vocabulary), and `Perplexity::corpus_perplexity` (zero scorable tokens). |
| `Serialization` | **yes** (`serde-extras`) | Implicitly, via `?` on `bincode::serialize_into` / `deserialize_from` in every `save`/`load`/`*_portable` method. |
| `SerializationMessage` | **yes** (`cli`) | The language registry's JSON (de)serialization. |
| `Xml` | conversion hook | The `#[from] quick_xml::Error` impl exists, but no libgrammstein function currently returns it — `WikipediaReader` handles XML events internally and surfaces `std::io::Result`. |
| `Neural` | conversion hook | The `#[from] NeuralError` impl exists; the neural module works in `Result<_, NeuralError>` and does not currently lift into `Error`. |
| `InvalidOrder` | **never constructed** | Declared, but nothing in the crate produces it. See the caveat below. |
| `NotTrained` | **never constructed** | Declared, but nothing in the crate produces it. |

> **`InvalidOrder` is not a guard rail.** `TrainerBuilder::order(0)` does **not** return
> `Err(Error::InvalidOrder(0))` — the order is never validated. Training simply counts nothing
> (the counting loop `for n in 1..=order` is empty), and the resulting model has `order() == 0`,
> whose `sentence_log_prob` computes `order - 1` on a `usize` and therefore panics with an
> arithmetic overflow in debug builds (and wraps in release). **Validate the order at the call
> site**; a `1..=5` range is the practical envelope.

## Handling

### Propagate with `?`

Every fallible API returns `Result<T>`, and the `#[from]` impls make `?` transparent across
`std::io::Error` and (with `serde-extras`) `bincode::Error`:

```rust
use libgrammstein::corpus::PlaintextReader;
use libgrammstein::ngram::{NgramEntry, NgramModel, TrainerBuilder};
use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

fn train_and_save(corpus: &str, out: &str) -> libgrammstein::Result<()> {
    // io::Error -> Error::Io, automatically
    let reader = PlaintextReader::from_file(corpus)?;

    let model = TrainerBuilder::new(DynamicDawgChar::<NgramEntry>::new())
        .order(5)
        .train(reader)?;

    model.save(out)?;   // bincode::Error -> Error::Serialization, automatically
    Ok(())
}
```

### Match the variants you can act on

```rust
use libgrammstein::Error;

fn explain(err: &Error) -> String {
    match err {
        Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
            "Corpus or model file not found — check the path.".into()
        }
        Error::Io(e) => format!("I/O failure: {e}"),
        Error::EmptyCorpus => {
            "No scorable sentences. The file may be empty, or every line was filtered out \
             by the tokenizer's minimum-sentence-length rule.".into()
        }
        #[cfg(feature = "serde-extras")]
        Error::Serialization(e) => format!("Corrupt or version-mismatched model file: {e}"),
        // InvalidOrder / NotTrained / Xml / Neural are not produced today — but a
        // wildcard keeps this exhaustive across features and future versions.
        other => format!("{other}"),
    }
}
```

### `EmptyCorpus` is the one to expect

It is by far the most commonly hit variant, and it has three distinct causes worth
distinguishing:

| Where | Why |
|---|---|
| `TrainerBuilder::train` | The reader yielded no batches — an empty/missing file, or a directory with no matching extensions. |
| `EmbeddingTrainerBuilder::train` | The vocabulary came out empty — usually `min_count` (default `5`) filtered every word out of a small corpus. Lower it. |
| `Perplexity::corpus_perplexity` | Zero tokens were scorable — every sentence was blank. |

A subtle third-party cause: `Tokenizer` drops sentences shorter than `min_sentence_length`
(default **10 characters**), so a corpus of very short lines can train to `EmptyCorpus` even
though the file is non-empty.

## `CliError` (feature `cli`)

The CLI wraps the library error rather than replacing it:

```rust
use libgrammstein::cli::error::{CliError, CliResult};

#[derive(thiserror::Error, Debug)]
pub enum CliError {
    #[error("{0}")]                                   Library(#[from] crate::Error),
    #[error("I/O error: {0}")]                        Io(#[from] std::io::Error),
    #[error("File not found: {path}")]                FileNotFound { path: PathBuf },
    #[error("Invalid argument: {message}")]           InvalidArgument { message: String },
    #[error("Failed to load model from {path}: {reason}")]
                                                      ModelLoad { path: PathBuf, reason: String },
    #[error("Corpus error: {message}")]               Corpus { message: String },
    #[error("Checkpoint error: {message}")]           Checkpoint { message: String },
    #[error("Training interrupted")]                  Interrupted,
    #[error("Training error: {message}")]             Training { message: String },
    #[error("I/O error: {message}")]                  IoError { message: String },
    #[error("REPL error: {message}")]                 Repl { message: String },
    #[error("Serialization error: {0}")]              Serialization(String),
    #[error("Language detection error: {message}")]   LanguageDetection { message: String },
    #[error("Unsupported: {message}")]                Unsupported { message: String },
}

pub type CliResult<T> = Result<T, CliError>;
```

Constructors keep call sites terse — `CliError::file_not_found(path)`,
`invalid_argument(msg)`, `model_load(path, reason)`, `corpus(msg)`, `checkpoint(msg)`,
`repl(msg)`, `unsupported(msg)`, `training(msg)`, `io(msg)` — and the module also provides the
colorized reporters `CliError::print_error(&self)`, `print_warning`, `print_info`, and
`print_success`.

Because of `#[from] crate::Error`, any library `Result` propagates into a `CliResult` with `?`.

## Best practices

1. **Return `Result<T>` and propagate with `?`.** The `#[from]` impls mean you rarely convert by
   hand.
2. **Validate the n-gram order yourself** (`1..=5`). The library will not do it for you, and
   `order == 0` is a panic waiting to happen.
3. **Do not test for OOV by catching an error.** `log_prob` never fails and never returns
   $`-\infty`$; it backs off to the uniform floor. Use `in_vocabulary(word)`, or compare against
   `oov_log_prob()` — see [Scoring](scoring.md#perplexityresult).
4. **Keep a wildcard arm** when matching `Error`: two variants are feature-gated, and two more
   are currently unconstructed but remain part of the public enum.
5. **Add context at the boundary**, where you know the path or the operation:

   ```rust
   let model = NgramModel::<DynamicDawgChar<NgramEntry>>::load(path)
       .map_err(|e| {
           log::error!("failed to load model from {path}: {e}");
           e
       })?;
   ```

## See also

- [NgramModel API](ngram.md) — the methods that return `Result`
- [SubwordEmbedding API](embedding.md) — training and persistence errors
- [Scoring API](scoring.md) — `EmptyCorpus` from `corpus_perplexity`
- [Traits API](traits.md) — `CorpusReader`, whose implementations surface `std::io::Error`
- [CLI reference](../cli/README.md) — how `CliError` is rendered to the terminal
