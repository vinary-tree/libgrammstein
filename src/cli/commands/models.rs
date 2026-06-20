//! Model management command implementations.

use std::path::Path;

use console::style;

use crate::cli::args::{ModelsCommands, ModelsInfoArgs, ModelsListArgs, OutputFormat};
use crate::cli::error::{CliError, CliResult};
use crate::cli::output;

/// Run the models command.
pub fn run(cmd: ModelsCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        ModelsCommands::List(args) => models_list(args, verbose),
        ModelsCommands::Info(args) => models_info(args, verbose),
    }
}

/// Detected model type with metadata.
enum DetectedModel {
    Hybrid { ngram_vocab_size: usize },
    Ngram { vocab_size: usize },
    Embedding { vocab_size: usize },
    Unknown,
}

impl DetectedModel {
    fn type_name(&self) -> &'static str {
        match self {
            DetectedModel::Hybrid { .. } => "hybrid",
            DetectedModel::Ngram { .. } => "ngram",
            DetectedModel::Embedding { .. } => "embedding",
            DetectedModel::Unknown => "unknown",
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            DetectedModel::Hybrid {
                ngram_vocab_size, ..
            } => *ngram_vocab_size,
            DetectedModel::Ngram { vocab_size, .. } => *vocab_size,
            DetectedModel::Embedding { vocab_size, .. } => *vocab_size,
            DetectedModel::Unknown => 0,
        }
    }
}

/// Try to detect model type and extract metadata.
fn detect_model(path: &Path) -> DetectedModel {
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::{NgramEntry, NgramModel};
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

    // Try hybrid model first (most complex)
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new)
    {
        return DetectedModel::Hybrid {
            ngram_vocab_size: model.ngram_model().vocab_size(),
        };
    }

    // Try n-gram model
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new) {
        return DetectedModel::Ngram {
            vocab_size: model.vocab_size(),
        };
    }

    // Try embedding model
    if let Ok(model) = SubwordEmbedding::load(path) {
        return DetectedModel::Embedding {
            vocab_size: model.vocab_size(),
        };
    }

    DetectedModel::Unknown
}

/// Extract language info from path structure.
/// Expected: models_dir/<lang>/<dialect>/model.bin or models_dir/<lang>/model.bin
fn extract_language_from_path(path: &Path, models_dir: &Path) -> (String, Option<String>) {
    if let Ok(rel_path) = path.strip_prefix(models_dir) {
        let components: Vec<_> = rel_path.components().collect();

        if components.len() >= 2 {
            let lang = components[0].as_os_str().to_string_lossy().to_string();

            // Check if second component is a dialect (e.g., "en-US") or file
            if components.len() >= 3 {
                let dialect = components[1].as_os_str().to_string_lossy().to_string();
                // Only treat as dialect if it contains the language prefix
                if dialect.starts_with(&lang) || dialect.contains('-') {
                    return (lang, Some(dialect));
                }
            }
            return (lang, None);
        }
    }

    // Fallback: try to infer from filename
    let filename = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    // Check for patterns like "en_ngram", "de-DE_hybrid"
    if let Some(underscore_pos) = filename.find('_') {
        let lang_part = &filename[..underscore_pos];
        if lang_part.contains('-') {
            // It's a dialect like "en-US"
            let base = lang_part.split('-').next().unwrap_or(lang_part);
            return (base.to_string(), Some(lang_part.to_string()));
        }
        return (lang_part.to_string(), None);
    }

    ("unknown".to_string(), None)
}

/// Recursively scan directory for model files.
fn scan_models_dir(dir: &Path, models_dir: &Path, verbose: bool) -> Vec<ModelEntry> {
    let mut models = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            if verbose {
                eprintln!("  Warning: Cannot read directory {}: {}", dir.display(), e);
            }
            return models;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectories
            models.extend(scan_models_dir(&path, models_dir, verbose));
        } else if let Some(ext) = path.extension() {
            // Check for model file extensions
            if ext == "bin" || ext == "model" {
                if verbose {
                    eprintln!("  Scanning: {}", path.display());
                }

                let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

                let detected = detect_model(&path);
                let (language, dialect) = extract_language_from_path(&path, models_dir);

                models.push(ModelEntry {
                    language,
                    dialect,
                    model_type: detected.type_name().to_string(),
                    vocab_size: detected.vocab_size(),
                    size_bytes,
                    path,
                });
            }
        }
    }

    models
}

/// List installed models.
fn models_list(args: ModelsListArgs, verbose: bool) -> CliResult<()> {
    if !args.models_dir.exists() {
        eprintln!(
            "{}: Models directory does not exist: {}",
            style("warning").yellow(),
            args.models_dir.display()
        );
        return Ok(());
    }

    if verbose {
        eprintln!("Scanning models directory: {}", args.models_dir.display());
    }

    // Scan models directory
    let models = scan_models_dir(&args.models_dir, &args.models_dir, verbose);

    // Filter by language if specified
    let filtered: Vec<_> = if let Some(ref lang) = args.language {
        models
            .into_iter()
            .filter(|m| m.language == *lang || m.dialect.as_ref() == Some(lang))
            .collect()
    } else {
        models
    };

    if filtered.is_empty() {
        if args.language.is_some() {
            eprintln!(
                "No models found for language: {}",
                args.language.as_ref().unwrap()
            );
        } else {
            eprintln!("No models found in: {}", args.models_dir.display());
        }
        return Ok(());
    }

    // Sort by language, then dialect, then type
    let mut sorted = filtered;
    sorted.sort_by(|a, b| {
        a.language
            .cmp(&b.language)
            .then_with(|| a.dialect.cmp(&b.dialect))
            .then_with(|| a.model_type.cmp(&b.model_type))
    });

    match args.format {
        OutputFormat::Table => {
            let headers = &["Language", "Dialect", "Type", "Vocab Size", "Size", "Path"];
            let rows: Vec<Vec<String>> = sorted
                .iter()
                .map(|m| {
                    vec![
                        m.language.clone(),
                        m.dialect.clone().unwrap_or_else(|| "-".to_string()),
                        m.model_type.clone(),
                        format_number(m.vocab_size),
                        humansize::format_size(m.size_bytes, humansize::BINARY),
                        m.path
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| m.path.display().to_string()),
                    ]
                })
                .collect();
            output::print_table(headers, rows);
        }
        OutputFormat::Json => {
            let json: Vec<_> = sorted
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "language": m.language,
                        "dialect": m.dialect,
                        "type": m.model_type,
                        "vocab_size": m.vocab_size,
                        "size_bytes": m.size_bytes,
                        "path": m.path.display().to_string()
                    })
                })
                .collect();
            output::print_json(&json)?;
        }
    }

    println!();
    println!("{} {} model(s) found", style("info:").cyan(), sorted.len());

    Ok(())
}

/// Display model information.
fn models_info(args: ModelsInfoArgs, verbose: bool) -> CliResult<()> {
    if !args.model.exists() {
        return Err(CliError::file_not_found(&args.model));
    }

    if verbose {
        eprintln!("Loading model: {}", args.model.display());
    }

    let file_size_bytes = std::fs::metadata(&args.model).map(|m| m.len()).unwrap_or(0);

    // Try to load and get detailed info
    let info = load_model_info(&args.model, file_size_bytes)?;

    if args.json {
        output::print_json(&info.to_json())?;
    } else {
        print_model_info(&info);
    }

    Ok(())
}

/// Load model and extract detailed information.
fn load_model_info(path: &Path, file_size_bytes: u64) -> CliResult<ModelInfo> {
    use crate::embedding::SubwordEmbedding;
    use crate::hybrid::HybridLanguageModel;
    use crate::ngram::{NgramEntry, NgramModel};
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

    // Try hybrid model first
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new)
    {
        return Ok(ModelInfo {
            path: path.display().to_string(),
            model_type: "HybridLanguageModel<DynamicDawgChar>".to_string(),
            ngram_order: Some(model.ngram_model().order()),
            vocab_size: model.ngram_model().vocab_size(),
            ngram_count: Some(model.ngram_model().ngram_count() as u64),
            total_count: Some(model.ngram_model().total_count()),
            embedding_dim: Some(model.embedding_model().dim()),
            embedding_vocab_size: Some(model.embedding_model().vocab_size()),
            smoothing: Some("Modified Kneser-Ney".to_string()),
            strategy: Some(format!("{:?}", model.config().strategy)),
            cache_size: Some(model.config().cache_size),
            file_size_bytes,
        });
    }

    // Try n-gram model
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::<NgramEntry>::new) {
        return Ok(ModelInfo {
            path: path.display().to_string(),
            model_type: "NgramModel<DynamicDawgChar>".to_string(),
            ngram_order: Some(model.order()),
            vocab_size: model.vocab_size(),
            ngram_count: Some(model.ngram_count() as u64),
            total_count: Some(model.total_count()),
            embedding_dim: None,
            embedding_vocab_size: None,
            smoothing: Some("Modified Kneser-Ney".to_string()),
            strategy: None,
            cache_size: None,
            file_size_bytes,
        });
    }

    // Try embedding model
    if let Ok(model) = SubwordEmbedding::load(path) {
        return Ok(ModelInfo {
            path: path.display().to_string(),
            model_type: "SubwordEmbedding".to_string(),
            ngram_order: None,
            vocab_size: model.vocab_size(),
            ngram_count: None,
            total_count: None,
            embedding_dim: Some(model.dim()),
            embedding_vocab_size: Some(model.vocab_size()),
            smoothing: None,
            strategy: None,
            cache_size: None,
            file_size_bytes,
        });
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (unknown format or corrupted file)".to_string(),
    ))
}

/// Print model info in human-readable format.
fn print_model_info(info: &ModelInfo) {
    println!("{}", style("Model Information").bold().underlined());
    println!();
    println!("Path: {}", style(&info.path).cyan());
    println!("Type: {}", info.model_type);
    println!(
        "Size: {}",
        humansize::format_size(info.file_size_bytes, humansize::BINARY)
    );
    println!();

    if let Some(order) = info.ngram_order {
        println!("{}", style("N-gram component:").bold());
        println!("  Order:       {}", order);
        println!("  Vocab size:  {}", format_number(info.vocab_size));
        if let Some(count) = info.ngram_count {
            println!("  N-grams:     {}", format_number(count as usize));
        }
        if let Some(total) = info.total_count {
            println!("  Total count: {}", format_number(total as usize));
        }
        if let Some(ref smoothing) = info.smoothing {
            println!("  Smoothing:   {}", smoothing);
        }
        println!();
    }

    if let Some(dim) = info.embedding_dim {
        println!("{}", style("Embedding component:").bold());
        println!("  Dimension:   {}", dim);
        if let Some(vocab) = info.embedding_vocab_size {
            println!("  Vocab size:  {}", format_number(vocab));
        }
        println!();
    }

    if info.strategy.is_some() || info.cache_size.is_some() {
        println!("{}", style("Hybrid config:").bold());
        if let Some(ref strategy) = info.strategy {
            println!("  Strategy:    {}", strategy);
        }
        if let Some(cache) = info.cache_size {
            println!("  Cache size:  {}", format_number(cache));
        }
        println!();
    }
}

/// Format number with thousands separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}

/// Model entry for listing.
struct ModelEntry {
    language: String,
    dialect: Option<String>,
    model_type: String,
    vocab_size: usize,
    size_bytes: u64,
    path: std::path::PathBuf,
}

/// Model information for display.
struct ModelInfo {
    path: String,
    model_type: String,
    ngram_order: Option<usize>,
    vocab_size: usize,
    ngram_count: Option<u64>,
    total_count: Option<u64>,
    embedding_dim: Option<usize>,
    embedding_vocab_size: Option<usize>,
    smoothing: Option<String>,
    strategy: Option<String>,
    cache_size: Option<usize>,
    file_size_bytes: u64,
}

impl ModelInfo {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "path": self.path,
            "type": self.model_type,
            "file_size_bytes": self.file_size_bytes,
            "ngram": if self.ngram_order.is_some() {
                Some(serde_json::json!({
                    "order": self.ngram_order,
                    "vocab_size": self.vocab_size,
                    "ngram_count": self.ngram_count,
                    "total_count": self.total_count,
                    "smoothing": self.smoothing
                }))
            } else {
                None
            },
            "embedding": if self.embedding_dim.is_some() {
                Some(serde_json::json!({
                    "dimension": self.embedding_dim,
                    "vocab_size": self.embedding_vocab_size
                }))
            } else {
                None
            },
            "hybrid_config": if self.strategy.is_some() {
                Some(serde_json::json!({
                    "strategy": self.strategy,
                    "cache_size": self.cache_size
                }))
            } else {
                None
            }
        })
    }
}
