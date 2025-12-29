//! Formatted output utilities for CLI commands.

use std::io::{self, Write};

use comfy_table::{Cell, Color, ContentArrangement, Table};
use console::style;

/// Print a formatted table.
pub fn print_table(headers: &[&str], rows: Vec<Vec<String>>) {
    let mut table = Table::new();
    table.set_content_arrangement(ContentArrangement::Dynamic);

    // Add header row with styling
    table.set_header(headers.iter().map(|h| Cell::new(h).fg(Color::Cyan)));

    // Add data rows
    for row in rows {
        table.add_row(row);
    }

    println!("{table}");
}

/// Print key-value pairs in a formatted way.
pub fn print_info_block(title: &str, items: &[(&str, String)]) {
    println!("{}", style(title).bold().underlined());
    for (key, value) in items {
        println!("  {}: {}", style(key).dim(), value);
    }
    println!();
}

/// Print a section header.
pub fn print_section(title: &str) {
    println!("\n{}", style(title).bold().cyan());
    println!("{}", style("─".repeat(title.len())).dim());
}

/// Print model summary.
pub fn print_model_summary(
    model_type: &str,
    vocab_size: usize,
    ngram_count: Option<u64>,
    embedding_dim: Option<usize>,
) {
    println!("{}", style("Model Summary").bold().underlined());
    println!("  Type:       {}", model_type);
    println!("  Vocab size: {}", vocab_size);

    if let Some(count) = ngram_count {
        println!("  N-grams:    {}", count);
    }

    if let Some(dim) = embedding_dim {
        println!("  Embed dim:  {}", dim);
    }

    println!();
}

/// Print perplexity results.
pub fn print_perplexity_results(
    model_path: &str,
    corpus_path: &str,
    sentences: u64,
    tokens: u64,
    perplexity: f64,
    log_prob: f64,
    oov_rate: f64,
) {
    println!("{}", style("Perplexity Evaluation").bold().underlined());
    println!();
    println!("  Model:         {}", model_path);
    println!(
        "  Test corpus:   {} ({} sentences, {} tokens)",
        corpus_path, sentences, tokens
    );
    println!();
    println!(
        "  Perplexity:     {}",
        style(format!("{:.2}", perplexity)).bold()
    );
    println!("  Log probability: {:.2}", log_prob);
    println!("  OOV rate:       {:.2}%", oov_rate * 100.0);
    println!(
        "  Avg tokens/sent: {:.2}",
        tokens as f64 / sentences as f64
    );
    println!();
}

/// Print comparison results as a table.
pub fn print_comparison_table(results: Vec<ModelComparisonResult>) {
    let headers = &["Model", "Perplexity", "OOV Rate", "Time (s)"];

    let rows: Vec<Vec<String>> = results
        .iter()
        .map(|r| {
            vec![
                r.model_name.clone(),
                format!("{:.2}", r.perplexity),
                format!("{:.2}%", r.oov_rate * 100.0),
                format!("{:.2}", r.eval_time_secs),
            ]
        })
        .collect();

    print_table(headers, rows);
}

/// Model comparison result.
pub struct ModelComparisonResult {
    /// Model name or path.
    pub model_name: String,
    /// Perplexity on test set.
    pub perplexity: f64,
    /// Out-of-vocabulary rate.
    pub oov_rate: f64,
    /// Evaluation time in seconds.
    pub eval_time_secs: f64,
}

/// Print similar words results.
pub fn print_similar_words(word: &str, similar: &[(String, f64)]) {
    println!("Similar to \"{}\":", style(word).bold());
    for (i, (w, score)) in similar.iter().enumerate() {
        println!("  {}. {:<15} {:.4}", i + 1, w, score);
    }
}

/// Print completions.
pub fn print_completions(context: &[String], completions: &[(String, f64)]) {
    println!(
        "Top completions for \"{}\":",
        style(context.join(" ")).bold()
    );
    for (i, (word, log_prob)) in completions.iter().enumerate() {
        let prob = log_prob.exp();
        println!(
            "  {}. {:<15} {:.3}  (P={:.4})",
            i + 1,
            word,
            log_prob,
            prob
        );
    }
}

/// Print score result.
pub fn print_score(tokens: &[String], log_prob: f64, is_sentence: bool) {
    let label = if is_sentence {
        "Sentence"
    } else {
        "Continuation"
    };

    println!("{} \"{}\":", label, tokens.join(" "));
    println!("  Log probability: {:.4}", log_prob);
    println!("  Perplexity:      {:.2}", (-log_prob / tokens.len() as f64).exp());
}

/// Print JSON output.
pub fn print_json<T: serde::Serialize>(value: &T) -> io::Result<()> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    println!("{}", json);
    Ok(())
}

/// Print corpus statistics.
pub fn print_corpus_stats(
    path: &str,
    format: &str,
    documents: u64,
    sentences: u64,
    tokens: u64,
    unique_words: u64,
    top_words: &[(String, u64)],
) {
    println!("{}", style("Corpus Statistics").bold().underlined());
    println!();
    println!("  Path:         {}", path);
    println!("  Format:       {}", format);
    println!();
    println!("  Documents:    {}", documents);
    println!("  Sentences:    {}", sentences);
    println!("  Tokens:       {}", tokens);
    println!("  Unique words: {}", unique_words);
    println!();

    if !top_words.is_empty() {
        println!("  Top {} words:", top_words.len());
        for (i, (word, count)) in top_words.iter().enumerate() {
            let pct = *count as f64 / tokens as f64 * 100.0;
            println!("    {}. {:<15} {:>8} ({:.2}%)", i + 1, word, count, pct);
        }
    }
}

/// Flush stdout.
pub fn flush_stdout() -> io::Result<()> {
    io::stdout().flush()
}
