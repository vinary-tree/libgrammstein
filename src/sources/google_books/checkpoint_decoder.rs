//! Human-readable checkpoint decoder for debugging and inspection.
//!
//! This module provides utilities to decode checkpoint data into
//! human-readable format for debugging, logging, and inspection.

use super::checkpoint::{ImportCheckpoint, PrefixState};

/// Decode checkpoint to a human-readable string format.
///
/// This produces a detailed text representation of the checkpoint
/// suitable for debugging and inspection.
///
/// # Example Output
///
/// ```text
/// Checkpoint v3
/// Timestamp: 2024-01-15 10:30:00 UTC
/// MKN Phase: Pass1InProgress { current_order: 2 }
///
/// === Order 1 ===
/// Complete: true
/// N-grams: 1234567
/// Completed (26): ["a", "b", "c", ...]
/// In Progress (0): []
/// Failed (0): []
///
/// === Order 2 ===
/// Complete: false
/// N-grams: 9876543
/// Completed (500): ["aa", "ab", ...]
/// In Progress (3): ["th", "wh", "qu"]
/// Failed (1): ["zz"]
/// ```
pub fn decode_checkpoint(checkpoint: &ImportCheckpoint) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!("Checkpoint v{}\n", checkpoint.version));
    output.push_str(&format!(
        "Timestamp: {}\n",
        checkpoint.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
    ));
    output.push_str(&format!("MKN Phase: {:?}\n", checkpoint.mkn_phase));
    output.push_str(&format!("Byte Offset: {}\n", checkpoint.byte_offset));
    if let Some(ref prefix) = checkpoint.current_prefix {
        output.push_str(&format!("Current Prefix: {}\n", prefix));
    }
    output.push('\n');

    // Global stats
    output.push_str("=== Global Stats ===\n");
    output.push_str(&format!(
        "N-grams Processed: {}\n",
        checkpoint.stats.ngrams_processed
    ));
    output.push_str(&format!(
        "Unique N-grams: {}\n",
        checkpoint.stats.unique_ngrams
    ));
    output.push_str(&format!(
        "Files Processed: {}\n",
        checkpoint.stats.files_processed
    ));
    output.push_str(&format!(
        "Bytes Downloaded: {}\n",
        checkpoint.stats.bytes_downloaded
    ));
    output.push_str(&format!(
        "Elapsed Seconds: {}\n",
        checkpoint.stats.elapsed_seconds
    ));
    output.push_str(&format!(
        "N-grams by Order: {:?}\n",
        checkpoint.stats.ngrams_by_order
    ));
    output.push('\n');

    // Per-order progress
    let mut orders: Vec<_> = checkpoint.order_progress.keys().collect();
    orders.sort();

    for order in orders {
        let progress = &checkpoint.order_progress[order];

        output.push_str(&format!("=== Order {} ===\n", order));
        output.push_str(&format!("Complete: {}\n", progress.is_complete));
        output.push_str(&format!("N-grams: {}\n", progress.ngrams_processed));

        // Collect and sort prefixes by state
        let mut completed: Vec<_> = progress.completed_prefixes().collect();
        let mut in_progress: Vec<_> = progress.in_progress_prefixes().collect();
        let mut failed: Vec<_> = progress.failed_prefixes().collect();

        completed.sort();
        in_progress.sort();
        failed.sort();

        // Format prefix lists (truncate if too long)
        output.push_str(&format_prefix_list("Completed", &completed));
        output.push_str(&format_prefix_list("In Progress", &in_progress));
        output.push_str(&format_prefix_list("Failed", &failed));
        output.push('\n');
    }

    output
}

/// Decode checkpoint to a compact summary string.
///
/// This produces a single-line summary suitable for logging.
///
/// # Example Output
///
/// ```text
/// v3 | MKN:Pass1InProgress(2) | Orders: 1(✓), 2(500/676, 3 in-prog, 1 fail)
/// ```
pub fn decode_checkpoint_summary(checkpoint: &ImportCheckpoint) -> String {
    let mut parts = Vec::new();

    parts.push(format!("v{}", checkpoint.version));

    // MKN phase (compact)
    let mkn_str = match &checkpoint.mkn_phase {
        super::checkpoint::MknPhase::NotStarted => "NotStarted".to_string(),
        super::checkpoint::MknPhase::Pass1InProgress { current_order } => {
            format!("Pass1({})", current_order)
        }
        super::checkpoint::MknPhase::Pass1Complete => "Pass1Done".to_string(),
        super::checkpoint::MknPhase::Pass2InProgress { current_order } => {
            format!("Pass2({})", current_order)
        }
        super::checkpoint::MknPhase::Complete => "Complete".to_string(),
    };
    parts.push(format!("MKN:{}", mkn_str));

    // Per-order status
    let mut order_parts = Vec::new();
    let mut orders: Vec<_> = checkpoint.order_progress.keys().collect();
    orders.sort();

    for order in orders {
        let progress = &checkpoint.order_progress[order];
        let completed_count = progress.count_state(PrefixState::Completed);
        let in_progress_count = progress.count_state(PrefixState::InProgress);
        let failed_count = progress.count_state(PrefixState::Failed);

        let total_prefixes = if *order == 1 { 26 } else { 676 };

        if progress.is_complete {
            order_parts.push(format!("{}(✓)", order));
        } else {
            let mut status = format!("{}({}/{}", order, completed_count, total_prefixes);
            if in_progress_count > 0 {
                status.push_str(&format!(", {} in-prog", in_progress_count));
            }
            if failed_count > 0 {
                status.push_str(&format!(", {} fail", failed_count));
            }
            status.push(')');
            order_parts.push(status);
        }
    }

    if !order_parts.is_empty() {
        parts.push(format!("Orders: {}", order_parts.join(", ")));
    }

    parts.join(" | ")
}

/// Format a list of prefixes with count and optional truncation.
fn format_prefix_list(label: &str, prefixes: &[&String]) -> String {
    const MAX_DISPLAY: usize = 20;

    if prefixes.is_empty() {
        return format!("{} (0): []\n", label);
    }

    if prefixes.len() <= MAX_DISPLAY {
        let list: Vec<_> = prefixes.iter().map(|s| format!("\"{}\"", s)).collect();
        format!("{} ({}): [{}]\n", label, prefixes.len(), list.join(", "))
    } else {
        // Truncate with ellipsis
        let shown: Vec<_> = prefixes[..MAX_DISPLAY]
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect();
        format!(
            "{} ({}): [{}, ... and {} more]\n",
            label,
            prefixes.len(),
            shown.join(", "),
            prefixes.len() - MAX_DISPLAY
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_checkpoint_empty() {
        let checkpoint = ImportCheckpoint::new();
        let output = decode_checkpoint(&checkpoint);

        assert!(output.contains("Checkpoint v3"));
        assert!(output.contains("MKN Phase: NotStarted"));
        assert!(output.contains("N-grams Processed: 0"));
    }

    #[test]
    fn test_decode_checkpoint_with_progress() {
        let mut checkpoint = ImportCheckpoint::new();
        checkpoint.complete_prefix(1, "a");
        checkpoint.complete_prefix(1, "b");
        checkpoint.start_prefix(1, "c");
        checkpoint.fail_prefix(1, "d");

        let output = decode_checkpoint(&checkpoint);

        assert!(output.contains("=== Order 1 ==="));
        assert!(output.contains("Completed (2)"));
        assert!(output.contains("In Progress (1)"));
        assert!(output.contains("Failed (1)"));
    }

    #[test]
    fn test_decode_checkpoint_summary_empty() {
        let checkpoint = ImportCheckpoint::new();
        let summary = decode_checkpoint_summary(&checkpoint);

        assert!(summary.contains("v3"));
        assert!(summary.contains("MKN:NotStarted"));
    }

    #[test]
    fn test_decode_checkpoint_summary_with_progress() {
        let mut checkpoint = ImportCheckpoint::new();

        // Complete order 1
        for c in 'a'..='z' {
            checkpoint.complete_prefix(1, &c.to_string());
        }
        checkpoint.complete_order(1).expect("should complete");

        // Partial order 2
        checkpoint.complete_prefix(2, "aa");
        checkpoint.complete_prefix(2, "ab");
        checkpoint.start_prefix(2, "ac");
        checkpoint.fail_prefix(2, "zz");

        let summary = decode_checkpoint_summary(&checkpoint);

        assert!(summary.contains("1(✓)"));
        assert!(summary.contains("2(2/676"));
        assert!(summary.contains("1 in-prog"));
        assert!(summary.contains("1 fail"));
    }

    #[test]
    fn test_format_prefix_list_empty() {
        let prefixes: Vec<&String> = vec![];
        let output = format_prefix_list("Test", &prefixes);
        assert_eq!(output, "Test (0): []\n");
    }

    #[test]
    fn test_format_prefix_list_short() {
        let items = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let prefixes: Vec<_> = items.iter().collect();
        let output = format_prefix_list("Test", &prefixes);
        assert!(output.contains("Test (3)"));
        assert!(output.contains("\"a\""));
        assert!(output.contains("\"b\""));
        assert!(output.contains("\"c\""));
    }

    #[test]
    fn test_format_prefix_list_truncated() {
        // Create 30 prefixes
        let items: Vec<String> = (0..30).map(|i| format!("prefix_{}", i)).collect();
        let prefixes: Vec<_> = items.iter().collect();
        let output = format_prefix_list("Test", &prefixes);

        assert!(output.contains("Test (30)"));
        assert!(output.contains("and 10 more"));
    }
}
