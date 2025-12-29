//! Corpus utility command implementations.

use std::collections::HashMap;
use std::path::Path;

use console::style;
use rand::prelude::{Rng, SeedableRng, StdRng};

use crate::cli::args::{
    CorpusCommands, CorpusDetectArgs, CorpusDownloadArgs, CorpusFormat,
    CorpusSampleArgs, CorpusSource, CorpusStatsArgs,
};
use crate::cli::error::{CliError, CliResult};
use crate::cli::output;
use crate::corpus::{CorpusReader, PlaintextReader, WikipediaReader, GutenbergReader, Tokenizer};

/// Run the corpus command.
pub fn run(cmd: CorpusCommands, verbose: bool) -> CliResult<()> {
    match cmd {
        CorpusCommands::Stats(args) => corpus_stats(args, verbose),
        CorpusCommands::Sample(args) => corpus_sample(args, verbose),
        CorpusCommands::Download(args) => corpus_download(args, verbose),
        CorpusCommands::Detect(args) => corpus_detect(args, verbose),
    }
}

/// Create a corpus reader based on path and format.
fn create_corpus_reader(path: &str, format: CorpusFormat) -> CliResult<Box<dyn CorpusReader>> {
    let path_obj = Path::new(path);

    match format {
        CorpusFormat::Plaintext => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    PlaintextReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    PlaintextReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Wikipedia => {
            // Check if it's an HTTP URL
            #[cfg(feature = "http-corpus")]
            if path.starts_with("http://") || path.starts_with("https://") {
                return Ok(Box::new(
                    WikipediaReader::from_url(path, crate::corpus::WikipediaConfig::default())
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ));
            }

            // Local file
            if path_obj.exists() {
                Ok(Box::new(
                    WikipediaReader::new(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Gutenberg => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    GutenbergReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    GutenbergReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
    }
}

/// Show corpus statistics.
fn corpus_stats(args: CorpusStatsArgs, verbose: bool) -> CliResult<()> {
    let path = Path::new(&args.corpus);
    let format_str = format!("{:?}", args.format);

    // Determine if directory or file
    let corpus_type = if path.is_dir() {
        format!("{} (directory)", format_str)
    } else if path.exists() {
        format!("{} (file)", format_str)
    } else {
        return Err(CliError::file_not_found(path));
    };

    if verbose {
        eprintln!("Analyzing corpus: {}", args.corpus);
        eprintln!("  Format: {}", corpus_type);
    }

    eprintln!("Loading corpus...");

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Get document count
    let doc_count = reader.document_count();

    eprintln!("Analyzing sentences and tokens...");

    // Count statistics
    let tokenizer = Tokenizer::new();
    let mut total_sentences = 0u64;
    let mut total_tokens = 0u64;
    let mut word_counts: HashMap<String, u64> = HashMap::new();
    let mut tokens_per_doc: Vec<usize> = Vec::new();

    for doc in reader.documents() {
        let mut doc_tokens = 0usize;

        for sentence in tokenizer.sentences(&doc.content) {
            total_sentences += 1;
            for word in tokenizer.words(&sentence) {
                total_tokens += 1;
                doc_tokens += 1;
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        tokens_per_doc.push(doc_tokens);
    }

    let unique_words = word_counts.len();

    // Get top 10 words
    let mut word_vec: Vec<(String, u64)> = word_counts.into_iter().collect();
    word_vec.sort_by(|a, b| b.1.cmp(&a.1));
    let top_words: Vec<(String, u64)> = word_vec.into_iter().take(10).collect();

    // Calculate document statistics
    let num_docs = doc_count.unwrap_or(tokens_per_doc.len());
    let (min_tokens, max_tokens, avg_tokens) = if !tokens_per_doc.is_empty() {
        let min = *tokens_per_doc.iter().min().unwrap_or(&0);
        let max = *tokens_per_doc.iter().max().unwrap_or(&0);
        let sum: usize = tokens_per_doc.iter().sum();
        let avg = sum as f64 / tokens_per_doc.len() as f64;
        (min, max, avg)
    } else {
        (0, 0, 0.0)
    };

    // Print statistics
    output::print_corpus_stats(
        &args.corpus,
        &corpus_type,
        num_docs as u64,
        total_sentences,
        total_tokens,
        unique_words as u64,
        &top_words,
    );

    // Print additional statistics
    println!();
    println!("{}", style("Token distribution:").bold());
    println!("  Min tokens/doc:  {}", min_tokens);
    println!("  Max tokens/doc:  {}", max_tokens);
    println!("  Avg tokens/doc:  {:.1}", avg_tokens);

    Ok(())
}

/// Sample sentences from corpus.
fn corpus_sample(args: CorpusSampleArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Sampling from corpus: {}", args.corpus);
        eprintln!("  Count: {}", args.count);
        if let Some(seed) = args.seed {
            eprintln!("  Seed: {}", seed);
        }
    }

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    // Initialize RNG - use StdRng for seeding capability
    let mut rng = if let Some(seed) = args.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    };

    // Collect all sentences (for small corpora) or use reservoir sampling
    eprintln!("Reading corpus...");

    // Use reservoir sampling to efficiently sample from the stream
    let mut reservoir: Vec<String> = Vec::with_capacity(args.count);
    let mut total_seen = 0u64;

    for sentence in reader.sentences() {
        total_seen += 1;

        if reservoir.len() < args.count {
            // Fill reservoir
            reservoir.push(sentence);
        } else {
            // Reservoir sampling: replace with probability count/total_seen
            let j = rng.gen_range(0..total_seen);
            if (j as usize) < args.count {
                reservoir[j as usize] = sentence;
            }
        }
    }

    if reservoir.is_empty() {
        eprintln!("{}: Corpus contains no sentences", style("warning").yellow());
        return Ok(());
    }

    // Print samples
    println!("{}", style("Sample sentences:").bold());
    for (i, sample) in reservoir.iter().enumerate() {
        println!("  {}. {}", i + 1, sample);
    }

    println!();
    println!(
        "{} {} sentences sampled from {} total",
        style("info:").cyan(),
        reservoir.len(),
        total_seen
    );

    Ok(())
}

/// Download corpus for language.
fn corpus_download(args: CorpusDownloadArgs, verbose: bool) -> CliResult<()> {
    if verbose {
        eprintln!("Downloading corpus for language: {}", args.language);
        eprintln!("  Source: {:?}", args.source);
        if let Some(ref output) = args.output {
            eprintln!("  Output: {}", output.display());
        }
    }

    // Get the download URL based on source and language
    let url = match args.source {
        CorpusSource::Wikipedia => wikipedia_dump_url(&args.language),
        CorpusSource::Gutenberg => {
            return Err(CliError::unsupported(
                "Gutenberg download not yet implemented. Visit https://www.gutenberg.org/",
            ));
        }
        CorpusSource::Oscar => {
            return Err(CliError::unsupported(
                "OSCAR download not yet implemented. Visit https://oscar-project.github.io/documentation/",
            ));
        }
    };

    // For now, provide instructions for manual download
    println!("{}", style("Corpus Download").bold().underlined());
    println!();
    println!("Language: {}", style(&args.language).cyan());
    println!("Source:   {:?}", args.source);
    println!();
    println!("{}", style("Download URL:").bold());
    println!("  {}", style(&url).green());
    println!();
    println!("{}", style("Manual download instructions:").bold());
    println!("  1. Download the file using wget or curl:");
    println!("     wget -c \"{}\"", url);
    println!();
    println!("  2. The file is bz2-compressed XML. You can use it directly with:");
    println!("     grammstein train ngram {} model.bin --format wikipedia",
             url.split('/').last().unwrap_or("dump.xml.bz2"));
    println!();

    if args.sample {
        println!(
            "{}: Sample download (--sample) is not yet implemented.",
            style("note").yellow()
        );
    }

    if args.resume {
        println!(
            "{}: Resume download (--resume) is not yet implemented.",
            style("note").yellow()
        );
    }

    // Return Ok since we provided useful information
    Ok(())
}

/// Get Wikipedia dump URL for a language.
fn wikipedia_dump_url(lang: &str) -> String {
    format!(
        "https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2",
        lang, lang
    )
}

/// Detect corpus language.
fn corpus_detect(args: CorpusDetectArgs, verbose: bool) -> CliResult<()> {
    use whatlang::{detect, Lang};

    if verbose {
        eprintln!("Detecting language of corpus: {}", args.corpus);
    }

    // Create corpus reader
    let reader = create_corpus_reader(&args.corpus, args.format)?;

    eprintln!("Sampling text for language detection...");

    // Sample sentences to build detection corpus
    let mut sample_text = String::new();
    let sample_limit = 10000; // Characters to sample
    let mut sentence_count = 0u64;

    for sentence in reader.sentences() {
        sample_text.push_str(&sentence);
        sample_text.push(' ');
        sentence_count += 1;

        if sample_text.len() >= sample_limit {
            break;
        }
    }

    if sample_text.is_empty() {
        return Err(CliError::corpus("Corpus contains no text for language detection".to_string()));
    }

    // Detect language
    let detection = detect(&sample_text);

    match detection {
        Some(info) => {
            let lang_code = lang_to_code(info.lang());
            let confidence = info.confidence() * 100.0;
            let reliable = info.is_reliable();

            println!("{}", style("Language Detection Results").bold().underlined());
            println!();
            println!("Detected language: {} ({})", style(lang_code).cyan().bold(), lang_name(info.lang()));
            println!("Confidence:        {:.1}%", confidence);
            println!("Reliable:          {}", if reliable { style("yes").green() } else { style("no").yellow() });
            println!();
            println!("Sample size:       {} sentences ({} characters)", sentence_count, sample_text.len());

            if !reliable {
                println!();
                println!(
                    "{}: Detection confidence is low. Consider providing more text.",
                    style("note").yellow()
                );
            }
        }
        None => {
            println!(
                "{}: Could not detect language. Text may be too short or contain mixed languages.",
                style("warning").yellow()
            );
        }
    }

    Ok(())
}

/// Convert whatlang Lang to ISO 639-1 code.
fn lang_to_code(lang: whatlang::Lang) -> &'static str {
    use whatlang::Lang::*;
    match lang {
        Eng => "en",
        Spa => "es",
        Deu => "de",
        Fra => "fr",
        Por => "pt",
        Ita => "it",
        Nld => "nl",
        Rus => "ru",
        Zho => "zh",
        Jpn => "ja",
        Kor => "ko",
        Ara => "ar",
        Hin => "hi",
        Pol => "pl",
        Tur => "tr",
        Vie => "vi",
        Ind => "id",
        Tha => "th",
        Swe => "sv",
        Ces => "cs",
        Dan => "da",
        Fin => "fi",
        Ell => "el",
        Heb => "he",
        Hun => "hu",
        Nob => "nb",
        Ron => "ro",
        Slk => "sk",
        Ukr => "uk",
        Bul => "bg",
        Cat => "ca",
        Hrv => "hr",
        Est => "et",
        Lav => "lv",
        Lit => "lt",
        Slv => "sl",
        Epo => "eo",
        Lat => "la",
        _ => "unknown",
    }
}

/// Get human-readable language name.
fn lang_name(lang: whatlang::Lang) -> &'static str {
    use whatlang::Lang::*;
    match lang {
        Eng => "English",
        Spa => "Spanish",
        Deu => "German",
        Fra => "French",
        Por => "Portuguese",
        Ita => "Italian",
        Nld => "Dutch",
        Rus => "Russian",
        Zho => "Chinese",
        Jpn => "Japanese",
        Kor => "Korean",
        Ara => "Arabic",
        Hin => "Hindi",
        Pol => "Polish",
        Tur => "Turkish",
        Vie => "Vietnamese",
        Ind => "Indonesian",
        Tha => "Thai",
        Swe => "Swedish",
        Ces => "Czech",
        Dan => "Danish",
        Fin => "Finnish",
        Ell => "Greek",
        Heb => "Hebrew",
        Hun => "Hungarian",
        Nob => "Norwegian Bokmål",
        Ron => "Romanian",
        Slk => "Slovak",
        Ukr => "Ukrainian",
        Bul => "Bulgarian",
        Cat => "Catalan",
        Hrv => "Croatian",
        Est => "Estonian",
        Lav => "Latvian",
        Lit => "Lithuanian",
        Slv => "Slovenian",
        Epo => "Esperanto",
        Lat => "Latin",
        _ => "Unknown",
    }
}
