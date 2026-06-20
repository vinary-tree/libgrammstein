//! Interactive REPL implementation.

use std::path::{Path, PathBuf};

use console::style;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::cli::args::ReplArgs;
use crate::cli::error::{CliError, CliResult};
use crate::embedding::SubwordEmbedding;
use crate::hybrid::HybridLanguageModel;
use crate::ngram::{NgramEntry, NgramModel};

type DynamicDict = libdictenstein::dynamic_dawg::char::DynamicDawgChar<NgramEntry>;

/// Loaded model type.
enum LoadedModel {
    Ngram(NgramModel<DynamicDict>),
    Embedding(SubwordEmbedding),
    Hybrid(HybridLanguageModel<DynamicDict>),
}

impl LoadedModel {
    /// Get model description.
    fn description(&self) -> String {
        match self {
            LoadedModel::Ngram(m) => {
                format!("N-gram (order={}, vocab={})", m.order(), m.vocab_size())
            }
            LoadedModel::Embedding(m) => {
                format!("Embedding (dim={}, vocab={})", m.dim(), m.vocab_size())
            }
            LoadedModel::Hybrid(m) => {
                format!(
                    "Hybrid (order={}, ngram_vocab={}, emb_vocab={})",
                    m.ngram_model().order(),
                    m.ngram_model().vocab_size(),
                    m.embedding_model().vocab_size()
                )
            }
        }
    }

    /// Score a sentence.
    fn sentence_log_prob(&self, tokens: &[&str]) -> Option<f64> {
        match self {
            LoadedModel::Ngram(m) => Some(m.sentence_log_prob(tokens)),
            LoadedModel::Hybrid(m) => Some(m.sentence_log_prob(tokens)),
            LoadedModel::Embedding(_) => None, // Embedding can't score sentences
        }
    }

    /// Get log probability of word given context.
    fn log_prob(&self, word: &str, context: &[&str]) -> Option<f64> {
        match self {
            LoadedModel::Ngram(m) => Some(m.log_prob(word, context)),
            LoadedModel::Hybrid(m) => Some(m.score(word, context)),
            LoadedModel::Embedding(_) => None,
        }
    }

    /// Find similar words.
    fn most_similar(&self, word: &str, k: usize) -> Option<Vec<(String, f32)>> {
        match self {
            LoadedModel::Embedding(m) => Some(m.most_similar(word, k)),
            LoadedModel::Hybrid(m) => Some(m.embedding_model().most_similar(word, k)),
            LoadedModel::Ngram(_) => None, // N-gram can't find similar words
        }
    }

    /// Check if word is in vocabulary.
    fn contains(&self, word: &str) -> bool {
        match self {
            LoadedModel::Ngram(m) => m.in_vocabulary(word),
            LoadedModel::Embedding(m) => m.contains(word),
            LoadedModel::Hybrid(m) => {
                m.ngram_model().in_vocabulary(word) || m.embedding_model().contains(word)
            }
        }
    }

    /// Get vocabulary words for completions.
    fn iter_vocabulary(&self) -> Box<dyn Iterator<Item = String> + '_> {
        match self {
            LoadedModel::Embedding(m) => Box::new(
                (0..m.vocab_size()).filter_map(move |i| m.index_to_word(i).map(|s| s.to_string())),
            ),
            LoadedModel::Hybrid(m) => {
                let emb = m.embedding_model();
                Box::new(
                    (0..emb.vocab_size())
                        .filter_map(move |i| emb.index_to_word(i).map(|s| s.to_string())),
                )
            }
            LoadedModel::Ngram(m) => {
                // Extract unigrams (no separator)
                Box::new(m.trie().iter_entries().filter_map(|(key, _)| {
                    if !key.contains('|') {
                        Some(key)
                    } else {
                        None
                    }
                }))
            }
        }
    }
}

/// Load a model from path.
fn load_model(path: &Path) -> CliResult<LoadedModel> {
    use libdictenstein::dynamic_dawg::char::DynamicDawgChar;

    // Try hybrid first
    if let Ok(model) = HybridLanguageModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(LoadedModel::Hybrid(model));
    }

    // Try N-gram
    if let Ok(model) = NgramModel::load_portable(path, DynamicDawgChar::new) {
        return Ok(LoadedModel::Ngram(model));
    }

    // Try embedding
    if let Ok(model) = SubwordEmbedding::load(path) {
        return Ok(LoadedModel::Embedding(model));
    }

    Err(CliError::model_load(
        path.to_path_buf(),
        "Failed to load model (unknown format or corrupted file)".to_string(),
    ))
}

/// REPL session state.
struct ReplSession {
    /// Currently loaded model path.
    model_path: Option<PathBuf>,
    /// Loaded model.
    model: Option<LoadedModel>,
}

impl ReplSession {
    fn new() -> Self {
        Self {
            model_path: None,
            model: None,
        }
    }

    fn load_model(&mut self, path: &str) -> CliResult<()> {
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(CliError::file_not_found(&path));
        }

        print!("Loading model... ");
        let model = load_model(&path)?;
        println!("{}", style("done").green());

        println!(
            "Loaded: {} ({})",
            style(path.display()).cyan(),
            model.description()
        );

        self.model_path = Some(path);
        self.model = Some(model);

        Ok(())
    }

    fn show_info(&self) {
        if let (Some(ref path), Some(ref model)) = (&self.model_path, &self.model) {
            println!("Loaded model: {}", path.display());
            println!("  Type: {}", model.description());

            // Additional info based on type
            match model {
                LoadedModel::Ngram(m) => {
                    println!("  N-grams: {}", m.ngram_count());
                    println!("  Total tokens: {}", m.total_count());
                }
                LoadedModel::Embedding(m) => {
                    println!("  Buckets: {}", m.bucket_count());
                }
                LoadedModel::Hybrid(m) => {
                    println!("  N-grams: {}", m.ngram_model().ngram_count());
                    println!("  Total tokens: {}", m.ngram_model().total_count());
                    println!(
                        "  Embedding buckets: {}",
                        m.embedding_model().bucket_count()
                    );
                }
            }
        } else {
            println!("No model loaded. Use 'load <path>' to load a model.");
        }
    }

    fn score(&self, tokens: &[&str]) {
        let model = match &self.model {
            Some(m) => m,
            None => {
                println!("No model loaded. Use 'load <path>' first.");
                return;
            }
        };

        match model.sentence_log_prob(tokens) {
            Some(log_prob) => {
                let perplexity = if tokens.is_empty() {
                    0.0
                } else {
                    (-log_prob / tokens.len() as f64).exp()
                };

                println!("Tokens: {}", style(tokens.join(" ")).cyan());
                println!(
                    "Log probability: {}",
                    style(format!("{:.4}", log_prob)).green()
                );
                println!(
                    "Perplexity:      {}",
                    style(format!("{:.2}", perplexity)).green()
                );
            }
            None => {
                println!("This model type doesn't support sentence scoring.");
                println!("Scoring is available for N-gram and Hybrid models.");
            }
        }
    }

    fn prob(&self, context: &[&str], word: &str) {
        let model = match &self.model {
            Some(m) => m,
            None => {
                println!("No model loaded. Use 'load <path>' first.");
                return;
            }
        };

        match model.log_prob(word, context) {
            Some(log_prob) => {
                let prob = log_prob.exp();

                println!(
                    "P({} | {}) = {} (log: {})",
                    style(word).cyan(),
                    style(context.join(" ")).cyan(),
                    style(format!("{:.6}", prob)).green(),
                    style(format!("{:.4}", log_prob)).green()
                );
            }
            None => {
                println!("This model type doesn't support probability queries.");
                println!("Probabilities are available for N-gram and Hybrid models.");
            }
        }
    }

    fn similar(&self, word: &str, n: usize) {
        let model = match &self.model {
            Some(m) => m,
            None => {
                println!("No model loaded. Use 'load <path>' first.");
                return;
            }
        };

        match model.most_similar(word, n) {
            Some(similar) => {
                let in_vocab = model.contains(word);
                if !in_vocab {
                    println!(
                        "{} Word '{}' not in vocabulary (using subword representation)",
                        style("note:").yellow(),
                        word
                    );
                }

                println!("Similar to \"{}\":", style(word).cyan());
                for (i, (w, score)) in similar.iter().enumerate() {
                    println!(
                        "  {}. {:<20} {}",
                        i + 1,
                        w,
                        style(format!("{:.4}", score)).green()
                    );
                }
            }
            None => {
                println!("This model type doesn't support similarity queries.");
                println!("Similarity is available for Embedding and Hybrid models.");
            }
        }
    }

    fn complete(&self, context: &[&str], n: usize) {
        let model = match &self.model {
            Some(m) => m,
            None => {
                println!("No model loaded. Use 'load <path>' first.");
                return;
            }
        };

        // Check if model supports scoring
        if model.log_prob("test", &[]).is_none() {
            println!("This model type doesn't support completions.");
            println!("Completions are available for N-gram and Hybrid models.");
            return;
        }

        println!("Computing completions...");

        // Collect vocabulary and score
        let mut scored: Vec<(String, f64)> = model
            .iter_vocabulary()
            .filter_map(|word| model.log_prob(&word, context).map(|lp| (word, lp)))
            .collect();

        // Sort by log probability (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top n
        scored.truncate(n);

        println!(
            "Top completions for \"{}\":",
            style(context.join(" ")).cyan()
        );
        for (i, (w, lp)) in scored.iter().enumerate() {
            println!(
                "  {}. {:<20} {} (P={})",
                i + 1,
                w,
                style(format!("{:.3}", lp)).green(),
                style(format!("{:.4}", lp.exp())).dim()
            );
        }
    }

    fn perplexity(&self, file_path: &str) {
        let model = match &self.model {
            Some(m) => m,
            None => {
                println!("No model loaded. Use 'load <path>' first.");
                return;
            }
        };

        // Check if model supports scoring
        if model.sentence_log_prob(&["test"]).is_none() {
            println!("This model type doesn't support perplexity evaluation.");
            println!("Perplexity is available for N-gram and Hybrid models.");
            return;
        }

        let path = Path::new(file_path);
        if !path.exists() {
            println!("File not found: {}", file_path);
            return;
        }

        // Load file and compute perplexity
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                println!("Failed to read file: {}", e);
                return;
            }
        };

        let tokenizer = crate::corpus::Tokenizer::new();
        let mut total_log_prob = 0.0f64;
        let mut total_tokens = 0u64;
        let mut total_sentences = 0u64;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let tokens_owned: Vec<String> = tokenizer.words(line).collect();
            if tokens_owned.is_empty() {
                continue;
            }

            let tokens: Vec<&str> = tokens_owned.iter().map(|s| s.as_str()).collect();

            if let Some(log_prob) = model.sentence_log_prob(&tokens) {
                total_log_prob += log_prob;
                total_tokens += tokens.len() as u64;
                total_sentences += 1;
            }
        }

        if total_tokens == 0 {
            println!("No tokens found in file.");
            return;
        }

        let perplexity = (-total_log_prob / total_tokens as f64).exp();

        println!("File: {}", style(file_path).cyan());
        println!("Sentences: {}", total_sentences);
        println!("Tokens: {}", total_tokens);
        println!(
            "Perplexity: {}",
            style(format!("{:.2}", perplexity)).green().bold()
        );
        println!("Log probability: {:.2}", total_log_prob);
    }
}

/// Print REPL help.
fn print_help() {
    println!("{}", style("Available commands:").bold());
    println!("  load <path>                  Load a model");
    println!("  info                         Show loaded model info");
    println!("  score <tokens...>            Score a sentence");
    println!("  prob <context...> | <word>   P(word | context)");
    println!("  similar <word> [n]           Find similar words");
    println!("  complete <context...> [n]    Get completions");
    println!("  perplexity <file>            Evaluate perplexity on file");
    println!("  help                         Show this help");
    println!("  quit                         Exit REPL");
}

/// Run the REPL.
pub fn run(args: ReplArgs) -> CliResult<()> {
    println!(
        "{}",
        style("grammstein REPL - Language Model Explorer").bold()
    );
    println!("Type 'help' for available commands, 'quit' to exit.\n");

    let mut session = ReplSession::new();

    // Load model if specified
    if let Some(ref model) = args.model {
        if let Err(e) = session.load_model(&model.display().to_string()) {
            eprintln!("Warning: Failed to load model: {}", e);
        }
    }

    // Initialize readline
    let mut rl = DefaultEditor::new().map_err(|e| CliError::repl(e.to_string()))?;

    // Load history
    let history_path = shellexpand::tilde(&args.history.display().to_string()).to_string();
    let _ = rl.load_history(&history_path);

    loop {
        let readline = rl.readline(&format!("{} ", style("grammstein>").cyan()));

        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(line);

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "quit" | "exit" | "q" => {
                        println!("Goodbye!");
                        break;
                    }
                    "help" | "h" | "?" => {
                        print_help();
                    }
                    "load" => {
                        if parts.len() < 2 {
                            println!("Usage: load <path>");
                        } else {
                            if let Err(e) = session.load_model(parts[1]) {
                                println!("Error: {}", e);
                            }
                        }
                    }
                    "info" => {
                        session.show_info();
                    }
                    "score" => {
                        if parts.len() < 2 {
                            println!("Usage: score <tokens...>");
                        } else {
                            session.score(&parts[1..]);
                        }
                    }
                    "prob" => {
                        // Parse "prob context... | word"
                        if let Some(pipe_pos) = parts.iter().position(|&p| p == "|") {
                            if pipe_pos < 2 || pipe_pos >= parts.len() - 1 {
                                println!("Usage: prob <context...> | <word>");
                            } else {
                                let context = &parts[1..pipe_pos];
                                let word = parts[pipe_pos + 1];
                                session.prob(context, word);
                            }
                        } else {
                            println!("Usage: prob <context...> | <word>");
                        }
                    }
                    "similar" => {
                        if parts.len() < 2 {
                            println!("Usage: similar <word> [n]");
                        } else {
                            let n = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
                            session.similar(parts[1], n);
                        }
                    }
                    "complete" => {
                        if parts.len() < 2 {
                            println!("Usage: complete <context...> [n]");
                        } else {
                            // Check if last arg is a number
                            let (context, n) = if let Some(last) = parts.last() {
                                if let Ok(n) = last.parse::<usize>() {
                                    (&parts[1..parts.len() - 1], n)
                                } else {
                                    (&parts[1..], 10)
                                }
                            } else {
                                (&parts[1..], 10)
                            };
                            session.complete(context, n);
                        }
                    }
                    "perplexity" | "ppl" => {
                        if parts.len() < 2 {
                            println!("Usage: perplexity <file>");
                        } else {
                            session.perplexity(parts[1]);
                        }
                    }
                    _ => {
                        println!(
                            "Unknown command: {}. Type 'help' for available commands.",
                            parts[0]
                        );
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_path);

    Ok(())
}
