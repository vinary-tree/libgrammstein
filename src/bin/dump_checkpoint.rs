//! Diagnostic tool to inspect checkpoint state from artrie and JSON files.
//!
//! This tool helps debug resume issues by showing the exact checkpoint state,
//! including prefix completion status, n-gram counts, and WAL file sizes.
//!
//! # Usage
//!
//! ```bash
//! # Inspect a specific directory
//! cargo run --release --bin dump_checkpoint --features cli,google-books -- \
//!     --dir bak-sharded-interrupted/
//!
//! # Compare multiple directories
//! cargo run --release --bin dump_checkpoint --features cli,google-books -- \
//!     --dir bak-sharded-interrupted/ --dir bak-sharded-completed/ --dir .
//! ```

#[cfg(feature = "mimalloc-alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use libdictenstein::persistent_artrie_char::PersistentARTrieChar;
use std::collections::HashMap;
use std::path::PathBuf;

/// Checkpoint state inspector for debugging resume issues.
#[derive(Parser)]
#[command(name = "dump_checkpoint")]
#[command(about = "Inspect checkpoint state from artrie files for debugging")]
struct Args {
    /// Directory or directories to inspect (can specify multiple with --dir)
    #[arg(long = "dir", value_name = "PATH")]
    directories: Vec<PathBuf>,

    /// Prefix for the model files (e.g., "english" for english.checkpoint.artrie)
    #[arg(long, default_value = "english")]
    prefix: String,

    /// Show all prefixes, not just completed ones
    #[arg(long, short = 'a')]
    all_prefixes: bool,

    /// Show raw checkpoint keys from trie
    #[arg(long)]
    raw_keys: bool,

    /// Verbose output
    #[arg(long, short = 'v')]
    verbose: bool,
}

/// Checkpoint key constants (duplicated from checkpoint.rs for standalone binary)
const CHECKPOINT_KEY_PREFIX: &str = "\x00__ckpt__";
const CHECKPOINT_VERSION_KEY: &str = "\x00__ckpt__:version";
const CHECKPOINT_MKN_PHASE_KEY: &str = "\x00__ckpt__:mkn_phase";
const CHECKPOINT_BYTE_OFFSET_KEY: &str = "\x00__ckpt__:byte_offset";
const CHECKPOINT_TIMESTAMP_KEY: &str = "\x00__ckpt__:timestamp";
const CHECKPOINT_NGRAMS_PROCESSED_KEY: &str = "\x00__ckpt__:ngrams_processed";
const CHECKPOINT_UNIQUE_NGRAMS_KEY: &str = "\x00__ckpt__:unique_ngrams";
const CHECKPOINT_FILES_PROCESSED_KEY: &str = "\x00__ckpt__:files_processed";
const CHECKPOINT_BYTES_DOWNLOADED_KEY: &str = "\x00__ckpt__:bytes_downloaded";
const CHECKPOINT_ELAPSED_KEY: &str = "\x00__ckpt__:elapsed_seconds";
const CHECKPOINT_NGRAMS_BY_ORDER_PREFIX: &str = "\x00__ckpt__:ngrams_by_order:";
const CHECKPOINT_PREFIX_KEY_PREFIX: &str = "\x00__ckpt__:prefix:";
const CHECKPOINT_ORDER_COMPLETE_PREFIX: &str = "\x00__ckpt__:order_complete:";
const CHECKPOINT_BITMAP_PREFIX: &str = "\x00__ckpt__:bitmap:";
const CHECKPOINT_ORDER_NGRAMS_PREFIX: &str = "\x00__ckpt__:order_ngrams:";

/// Prefix status codes
const STATUS_COMPLETED: u64 = 1;
const STATUS_IN_PROGRESS: u64 = 2;
const STATUS_FAILED: u64 = 3;

/// Bitmap state encoding
const BITMAP_STATE_IN_PROGRESS: u8 = 0b01;
const BITMAP_STATE_COMPLETED: u8 = 0b10;
const BITMAP_STATE_FAILED: u8 = 0b11;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    let args = Args::parse();

    if args.directories.is_empty() {
        eprintln!("Error: No directories specified. Use --dir <path> to specify directories.");
        std::process::exit(1);
    }

    for dir in &args.directories {
        println!("\n{}", "=".repeat(80));
        println!("Directory: {}", dir.display());
        println!("{}", "=".repeat(80));

        if let Err(e) = inspect_directory(dir, &args) {
            eprintln!("Error inspecting {}: {}", dir.display(), e);
        }
    }

    Ok(())
}

fn inspect_directory(dir: &PathBuf, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Check for WAL files first - this is crucial
    println!("\n--- WAL Files ---");
    check_wal_files(dir, &args.prefix)?;

    // Check for JSON checkpoint
    let json_path = dir.join(format!("{}.checkpoint.json", args.prefix));
    if json_path.exists() {
        println!("\n--- JSON Checkpoint ---");
        inspect_json_checkpoint(&json_path)?;
    } else {
        println!("\n--- JSON Checkpoint ---");
        println!("  Not found: {}", json_path.display());
    }

    // Check for trie-based checkpoint
    let trie_path = dir.join(format!("{}.checkpoint.artrie", args.prefix));
    if trie_path.exists() {
        println!("\n--- Trie Checkpoint ---");
        inspect_trie_checkpoint(&trie_path, args)?;
    } else {
        println!("\n--- Trie Checkpoint ---");
        println!("  Not found: {}", trie_path.display());
    }

    // Check vocabulary file
    let vocab_path = dir.join(format!("{}.vocab.artrie", args.prefix));
    if vocab_path.exists() {
        println!("\n--- Vocabulary ---");
        inspect_vocabulary(&vocab_path)?;
    }

    // Check sharding checkpoint if it exists
    let shard_checkpoint = dir.join(format!("{}_shards", args.prefix)).join("checkpoint.json");
    if shard_checkpoint.exists() {
        println!("\n--- Sharding Checkpoint ---");
        inspect_sharding_checkpoint(&shard_checkpoint)?;
    }

    Ok(())
}

fn check_wal_files(dir: &PathBuf, prefix: &str) -> Result<(), Box<dyn std::error::Error>> {
    let wal_patterns = [
        format!("{}.wal", prefix),
        format!("{}.vocab.wal", prefix),
        format!("{}.checkpoint.wal", prefix),
        format!("{}.mkn.wal", prefix),
    ];

    for pattern in &wal_patterns {
        let wal_path = dir.join(pattern);
        if wal_path.exists() {
            let metadata = std::fs::metadata(&wal_path)?;
            let size = metadata.len();
            let size_str = format_size(size);

            // WAL files with just headers are 64 bytes
            let status = if size <= 64 {
                "empty (checkpointed)"
            } else if size > 1_000_000 {
                "LARGE - NOT CHECKPOINTED!"
            } else {
                "has pending data"
            };

            println!("  {} - {} ({})", pattern, size_str, status);
        }
    }

    // Check for wal_archive
    let archive_dir = dir.join("wal_archive");
    if archive_dir.exists() && archive_dir.is_dir() {
        let count = std::fs::read_dir(&archive_dir)?.count();
        println!("  wal_archive/ - {} archived WAL files", count);
    }

    Ok(())
}

fn inspect_json_checkpoint(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let value: serde_json::Value = serde_json::from_reader(file)?;

    if let Some(version) = value.get("version").and_then(|v| v.as_u64()) {
        println!("  Version: {}", version);
    }

    if let Some(timestamp) = value.get("timestamp").and_then(|v| v.as_str()) {
        println!("  Timestamp: {}", timestamp);
    }

    if let Some(stats) = value.get("stats").and_then(|v| v.as_object()) {
        println!("  Stats:");
        if let Some(ngrams) = stats.get("ngrams_processed").and_then(|v| v.as_u64()) {
            println!("    N-grams processed: {}", ngrams);
        }
        if let Some(unique) = stats.get("unique_ngrams").and_then(|v| v.as_u64()) {
            println!("    Unique n-grams: {}", unique);
        }
        if let Some(files) = stats.get("files_processed").and_then(|v| v.as_u64()) {
            println!("    Files processed: {}", files);
        }
    }

    if let Some(order_progress) = value.get("order_progress").and_then(|v| v.as_object()) {
        println!("  Order Progress:");
        for (order, progress) in order_progress {
            if let Some(progress_obj) = progress.as_object() {
                let is_complete = progress_obj
                    .get("is_complete")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let completed_count = progress_obj
                    .get("completed_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let in_progress_count = progress_obj
                    .get("in_progress_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let failed_count = progress_obj
                    .get("failed_prefixes")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);

                let ngrams = progress_obj
                    .get("ngrams_processed")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                println!(
                    "    Order {}: completed={}, in_progress={}, failed={}, ngrams={}, is_complete={}",
                    order, completed_count, in_progress_count, failed_count, ngrams, is_complete
                );

                // Show completed prefixes
                if let Some(completed) = progress_obj.get("completed_prefixes").and_then(|v| v.as_array()) {
                    let prefixes: Vec<&str> = completed
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect();
                    if !prefixes.is_empty() {
                        let prefix_str = if prefixes.len() > 10 {
                            format!("{} (and {} more)", prefixes[..10].join(", "), prefixes.len() - 10)
                        } else {
                            prefixes.join(", ")
                        };
                        println!("      Completed: {}", prefix_str);
                    }
                }

                // Show in-progress prefixes (important for debugging)
                if let Some(in_progress) = progress_obj.get("in_progress_prefixes").and_then(|v| v.as_array()) {
                    let prefixes: Vec<&str> = in_progress
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect();
                    if !prefixes.is_empty() {
                        println!("      In Progress: {}", prefixes.join(", "));
                    }
                }
            }
        }
    }

    Ok(())
}

fn inspect_trie_checkpoint(path: &PathBuf, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let trie = PersistentARTrieChar::<u64>::open(path)?;

    println!("  Trie size: {} entries", trie.len());

    // Load basic metadata
    if let Some(version) = get_checkpoint_value(&trie, CHECKPOINT_VERSION_KEY)? {
        println!("  Version: {}", version);
    }

    if let Some(timestamp) = get_checkpoint_value(&trie, CHECKPOINT_TIMESTAMP_KEY)? {
        let dt = chrono::DateTime::from_timestamp(timestamp as i64, 0);
        if let Some(dt) = dt {
            println!("  Timestamp: {}", dt.format("%Y-%m-%d %H:%M:%S UTC"));
        }
    }

    if let Some(ngrams) = get_checkpoint_value(&trie, CHECKPOINT_NGRAMS_PROCESSED_KEY)? {
        println!("  N-grams processed: {}", ngrams);
    }

    if let Some(unique) = get_checkpoint_value(&trie, CHECKPOINT_UNIQUE_NGRAMS_KEY)? {
        println!("  Unique n-grams: {}", unique);
    }

    if let Some(files) = get_checkpoint_value(&trie, CHECKPOINT_FILES_PROCESSED_KEY)? {
        println!("  Files processed: {}", files);
    }

    if let Some(mkn_phase) = get_checkpoint_value(&trie, CHECKPOINT_MKN_PHASE_KEY)? {
        let phase_str = match mkn_phase {
            0 => "NotStarted",
            100 => "Pass1Complete",
            200 => "Complete",
            n if n >= 1 && n < 100 => "Pass1InProgress",
            n if n >= 101 && n < 200 => "Pass2InProgress",
            _ => "Unknown",
        };
        println!("  MKN Phase: {} ({})", mkn_phase, phase_str);
    }

    // Show ngrams by order
    println!("  N-grams by order:");
    for order in 1..=5u8 {
        let key = format!("{}{}", CHECKPOINT_NGRAMS_BY_ORDER_PREFIX, order);
        if let Some(count) = get_checkpoint_value(&trie, &key)? {
            if count > 0 {
                println!("    Order {}: {}", order, count);
            }
        }
    }

    // Load prefix states using v3 bitmap format or v2 key-per-prefix format
    println!("  Prefix states by order:");

    for order in 1..=5u8 {
        // Check if order is complete
        let complete_key = format!("{}{}", CHECKPOINT_ORDER_COMPLETE_PREFIX, order);
        let is_complete = get_checkpoint_value(&trie, &complete_key)?
            .map(|v| v == 1)
            .unwrap_or(false);

        // Try v3 bitmap format first
        let prefix_len = if order == 1 { 1u8 } else { 2u8 };
        let mut states = load_bitmap_states(&trie, order, prefix_len)?;

        // If no bitmap states, try v2 key-per-prefix format
        if states.is_empty() {
            states = load_v2_prefix_states(&trie, order)?;
        }

        if is_complete || !states.is_empty() {
            let completed: Vec<_> = states.iter().filter(|(_, s)| *s == "Completed").map(|(p, _)| p.as_str()).collect();
            let in_progress: Vec<_> = states.iter().filter(|(_, s)| *s == "InProgress").map(|(p, _)| p.as_str()).collect();
            let failed: Vec<_> = states.iter().filter(|(_, s)| *s == "Failed").map(|(p, _)| p.as_str()).collect();

            println!(
                "    Order {}: completed={}, in_progress={}, failed={}, is_complete={}",
                order,
                completed.len(),
                in_progress.len(),
                failed.len(),
                is_complete
            );

            if args.all_prefixes || args.verbose {
                if !completed.is_empty() {
                    let prefix_str = if completed.len() > 20 {
                        format!("{} (and {} more)", completed[..20].join(", "), completed.len() - 20)
                    } else {
                        completed.join(", ")
                    };
                    println!("      Completed: {}", prefix_str);
                }
            }

            if !in_progress.is_empty() {
                println!("      In Progress: {}", in_progress.join(", "));
            }

            if !failed.is_empty() {
                println!("      Failed: {}", failed.join(", "));
            }
        }
    }

    // Show raw keys if requested
    if args.raw_keys {
        println!("\n  Raw checkpoint keys:");
        if let Some(entries) = trie.iter_prefix_with_values(CHECKPOINT_KEY_PREFIX)? {
            for (key, value) in entries {
                // Clean up key for display
                let display_key = key.replace('\x00', "\\0");
                println!("    {} = {}", display_key, value);
            }
        }
    }

    Ok(())
}

fn get_checkpoint_value(
    trie: &PersistentARTrieChar<u64>,
    key: &str,
) -> Result<Option<u64>, Box<dyn std::error::Error>> {
    // The trie's get method returns Option<&u64>, dereference with .copied()
    Ok(trie.get(key).copied())
}

fn load_bitmap_states(
    trie: &PersistentARTrieChar<u64>,
    order: u8,
    prefix_len: u8,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let max_index: u16 = if prefix_len == 1 { 26 } else { 676 };
    let prefixes_per_chunk = 32usize;
    let num_chunks = (max_index as usize + prefixes_per_chunk - 1) / prefixes_per_chunk;

    // Load chunks
    let mut chunks = vec![0u64; num_chunks];
    let mut has_any = false;

    for chunk_idx in 0..num_chunks {
        let key = format!("{}{}:{}", CHECKPOINT_BITMAP_PREFIX, order, chunk_idx);
        if let Some(value) = get_checkpoint_value(trie, &key)? {
            chunks[chunk_idx] = value;
            if value != 0 {
                has_any = true;
            }
        }
    }

    if !has_any {
        return Ok(HashMap::new());
    }

    // Unpack states
    let mut states = HashMap::new();
    for index in 0..max_index {
        let chunk_idx = (index as usize) / prefixes_per_chunk;
        let bit_pos = ((index as usize) % prefixes_per_chunk) * 2;

        let state_bits = ((chunks[chunk_idx] >> bit_pos) & 0b11) as u8;
        let state = match state_bits {
            0b00 => continue, // NotStarted
            BITMAP_STATE_IN_PROGRESS => "InProgress",
            BITMAP_STATE_COMPLETED => "Completed",
            BITMAP_STATE_FAILED => "Failed",
            _ => continue,
        };

        let prefix = index_to_prefix(index, prefix_len);
        states.insert(prefix, state.to_string());
    }

    Ok(states)
}

fn load_v2_prefix_states(
    trie: &PersistentARTrieChar<u64>,
    order: u8,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let prefix_key_prefix = format!("{}{}:", CHECKPOINT_PREFIX_KEY_PREFIX, order);
    let mut states = HashMap::new();

    if let Some(entries) = trie.iter_prefix_with_values(&prefix_key_prefix)? {
        for (key, status_code) in entries {
            if let Some(prefix) = key.strip_prefix(&prefix_key_prefix) {
                let state = match status_code {
                    STATUS_COMPLETED => "Completed",
                    STATUS_IN_PROGRESS => "InProgress",
                    STATUS_FAILED => "Failed",
                    _ => continue,
                };
                states.insert(prefix.to_string(), state.to_string());
            }
        }
    }

    Ok(states)
}

fn index_to_prefix(index: u16, prefix_len: u8) -> String {
    match prefix_len {
        1 => {
            let c = (b'a' + index as u8) as char;
            c.to_string()
        }
        2 => {
            let c1 = (b'a' + (index / 26) as u8) as char;
            let c2 = (b'a' + (index % 26) as u8) as char;
            format!("{}{}", c1, c2)
        }
        _ => format!("?{}", index),
    }
}

fn inspect_vocabulary(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let trie = PersistentARTrieChar::<u64>::open(path)?;
    println!("  Vocabulary entries: {}", trie.len());

    // Check for metadata entries
    if let Some(entries) = trie.iter_prefix_with_values("\x00")? {
        let count = entries.len();
        if count > 0 {
            println!("  Metadata entries: {}", count);
        }
    }

    Ok(())
}

fn inspect_sharding_checkpoint(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let value: serde_json::Value = serde_json::from_reader(file)?;

    if let Some(state) = value.get("import_state") {
        println!("  Import state: {:?}", state);
    }

    if let Some(shards) = value.get("shards").and_then(|v| v.as_object()) {
        println!("  Shards: {}", shards.len());

        // Count in-progress shards
        let in_progress: Vec<_> = shards
            .iter()
            .filter(|(_, v)| {
                v.get("current_prefix")
                    .and_then(|p| p.as_str())
                    .is_some()
            })
            .collect();

        if !in_progress.is_empty() {
            println!("  In-progress shards: {}", in_progress.len());
            for (key, _) in in_progress {
                println!("    {}", key);
            }
        }
    }

    Ok(())
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
