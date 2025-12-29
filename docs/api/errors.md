# Error Types Reference

This document covers error handling in libgrammstein.

## Error Enum

The main error type is `libgrammstein::Error`:

```rust
use libgrammstein::Error;

#[derive(Error, Debug)]
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

    /// Serialization error (requires serde-extras feature).
    #[cfg(feature = "serde-extras")]
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),
}
```

## Result Type

The library provides a type alias for convenience:

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

## Error Variants

### Io

I/O errors from file operations.

```rust
// Example: File not found
let result = PlaintextReader::from_file("nonexistent.txt");
match result {
    Err(Error::Io(e)) => println!("I/O error: {}", e),
    _ => {}
}
```

**Common causes:**
- File not found
- Permission denied
- Disk full
- Network errors (HTTP corpus)

### Xml

XML parsing errors when reading Wikipedia dumps.

```rust
// Example: Malformed XML
let result = WikipediaReader::from_dump("malformed.xml");
match result {
    Err(Error::Xml(e)) => println!("XML error: {}", e),
    _ => {}
}
```

**Common causes:**
- Corrupted XML file
- Incomplete download
- Wrong file format

### InvalidOrder

N-gram order must be at least 1.

```rust
// Example: Invalid order
let result = TrainerBuilder::new(dictionary)
    .order(0)  // Invalid!
    .train(&reader);

match result {
    Err(Error::InvalidOrder(order)) => {
        println!("Invalid order: {}", order);
    }
    _ => {}
}
```

### EmptyCorpus

Training requires at least one sentence.

```rust
// Example: Empty input
let reader = PlaintextReader::from_string("");
let result = TrainerBuilder::new(dictionary).train(&reader);

match result {
    Err(Error::EmptyCorpus) => println!("No sentences found!"),
    _ => {}
}
```

**Common causes:**
- Empty file
- All text filtered out by quality filters
- Tokenizer producing no words

### NotTrained

Attempted to use a model that hasn't been trained.

```rust
// Example: Using untrained model
match result {
    Err(Error::NotTrained(reason)) => {
        println!("Model not trained: {}", reason);
    }
    _ => {}
}
```

### Serialization

Errors during save/load operations (requires `serde-extras` feature).

```rust
// Example: Corrupted model file
let result = NgramModel::<D>::load("corrupted.bin");

match result {
    Err(Error::Serialization(e)) => {
        println!("Serialization error: {}", e);
    }
    _ => {}
}
```

**Common causes:**
- Corrupted file
- Version mismatch
- Incomplete save
- Wrong model type

## Error Handling Patterns

### Basic Pattern

```rust
use libgrammstein::Result;

fn train_model() -> Result<NgramModel<D>> {
    let reader = PlaintextReader::from_file("corpus.txt")?;
    let model = TrainerBuilder::new(dictionary).train(&reader)?;
    Ok(model)
}
```

### With Context

```rust
fn train_model(path: &str) -> Result<NgramModel<D>> {
    let reader = PlaintextReader::from_file(path)
        .map_err(|e| {
            eprintln!("Failed to open corpus: {}", path);
            e
        })?;

    TrainerBuilder::new(dictionary).train(&reader)
}
```

### Matching Specific Errors

```rust
use libgrammstein::Error;

fn handle_training_error(err: Error) {
    match err {
        Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Corpus file not found. Please provide a valid path.");
        }
        Error::EmptyCorpus => {
            eprintln!("Corpus contains no valid sentences. Check your input.");
        }
        Error::InvalidOrder(n) => {
            eprintln!("N-gram order {} is invalid. Use 1 or higher.", n);
        }
        _ => {
            eprintln!("Training failed: {}", err);
        }
    }
}
```

### Converting Errors

```rust
use std::io::ErrorKind;

// From std::io::Error
let io_err = std::io::Error::new(ErrorKind::NotFound, "file not found");
let err: Error = io_err.into();

// From quick_xml::Error
// (automatically converted via #[from])
```

### Propagating with ?

```rust
fn load_and_query(path: &str) -> Result<f64> {
    let model: NgramModel<D> = NgramModel::load(path)?;
    Ok(model.log_prob("test", &["a"]))
}
```

## CLI Error Handling

The CLI provides user-friendly error messages:

```rust
use libgrammstein::cli::error::CliError;

#[derive(Error, Debug)]
pub enum CliError {
    #[error("Corpus not found: {0}")]
    CorpusNotFound(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error(transparent)]
    Library(#[from] libgrammstein::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

## Best Practices

1. **Use `?` for propagation**
   ```rust
   let model = NgramModel::load(path)?;
   ```

2. **Provide context in errors**
   ```rust
   let reader = PlaintextReader::from_file(path)
       .map_err(|e| format!("Failed to read {}: {}", path, e))?;
   ```

3. **Handle recoverable errors gracefully**
   ```rust
   match model.log_prob(word, context) {
       prob if prob.is_finite() => prob,
       _ => model.oov_log_prob(),  // Fallback for OOV
   }
   ```

4. **Log errors for debugging**
   ```rust
   if let Err(e) = model.save(path) {
       log::error!("Failed to save model: {}", e);
       return Err(e);
   }
   ```

## See Also

- [NgramModel API](ngram.md) - N-gram methods that return Result
- [SubwordEmbedding API](embedding.md) - Embedding methods
- [CLI Reference](../cli/README.md) - CLI error handling
