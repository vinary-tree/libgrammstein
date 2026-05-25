//! Custom widgets and helper functions for the TUI.

use std::collections::VecDeque;

use ratatui::prelude::*;

/// Create a sparkline widget showing throughput history.
///
/// Note: Due to ratatui's Sparkline requiring owned data, this function
/// consumes the data for rendering. The caller should pass a clone if
/// the original data needs to be preserved.
pub fn sparkline_data(data: &VecDeque<f64>) -> Vec<u64> {
    // Convert f64 to u64 for sparkline (scale to reasonable range)
    let max_val = data.iter().cloned().fold(f64::MIN, f64::max);
    let scale = if max_val > 0.0 { 100.0 / max_val } else { 1.0 };

    data.iter().map(|v| (v * scale) as u64).collect()
}

/// Create a mini progress bar using Unicode block characters.
///
/// Returns a string like "████████░░" representing the progress.
pub fn mini_progress_bar(progress: f32, width: usize) -> String {
    let filled = (progress * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);

    let filled_char = '\u{2588}'; // Full block
    let empty_char = '\u{2591}'; // Light shade

    format!(
        "{}{}",
        filled_char.to_string().repeat(filled),
        empty_char.to_string().repeat(empty)
    )
}

/// Format a number with SI suffixes (K, M, G).
pub fn format_si(n: u64) -> String {
    const K: u64 = 1_000;
    const M: u64 = 1_000_000;
    const G: u64 = 1_000_000_000;

    if n >= G {
        format!("{:.1}G", n as f64 / G as f64)
    } else if n >= M {
        format!("{:.1}M", n as f64 / M as f64)
    } else if n >= K {
        format!("{:.1}K", n as f64 / K as f64)
    } else {
        format!("{}", n)
    }
}

/// Creates a progress bar with colored segments for different outcomes.
///
/// Returns a `Vec<Span>` that can be used in a `Line`:
/// - Green segment: successfully completed files
/// - Yellow segment: skipped/failed files (will retry next session)
/// - Empty segment: remaining files
pub fn segmented_progress_bar(
    succeeded: u64,
    skipped: u64,
    total: u64,
    width: usize,
) -> Vec<Span<'static>> {
    if total == 0 {
        return vec![Span::raw("░".repeat(width))];
    }

    let succeeded_width = ((succeeded as f64 / total as f64) * width as f64).round() as usize;
    let skipped_width = ((skipped as f64 / total as f64) * width as f64).round() as usize;
    let empty_width = width.saturating_sub(succeeded_width + skipped_width);

    vec![
        Span::styled(
            "█".repeat(succeeded_width),
            Style::default().fg(Color::Green),
        ),
        Span::styled(
            "█".repeat(skipped_width),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw("░".repeat(empty_width)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_progress_bar() {
        // Use chars().count() since Unicode block characters are multi-byte
        assert_eq!(mini_progress_bar(0.0, 10).chars().count(), 10);
        assert_eq!(mini_progress_bar(1.0, 10).chars().count(), 10);
        assert_eq!(mini_progress_bar(0.5, 10).chars().count(), 10);
    }

    #[test]
    fn test_format_si() {
        assert_eq!(format_si(0), "0");
        assert_eq!(format_si(999), "999");
        assert_eq!(format_si(1000), "1.0K");
        assert_eq!(format_si(1500), "1.5K");
        assert_eq!(format_si(1_000_000), "1.0M");
        assert_eq!(format_si(1_500_000), "1.5M");
        assert_eq!(format_si(1_000_000_000), "1.0G");
    }
}
