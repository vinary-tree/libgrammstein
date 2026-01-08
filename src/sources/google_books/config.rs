//! Configuration for Google Books N-gram import.

use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;
use std::path::PathBuf;

/// Configuration for Google Books N-gram import.
///
/// This struct controls all aspects of the import process, including:
/// - Which n-gram orders to import (1-5)
/// - Minimum frequency thresholds
/// - Optional year filtering
/// - Parallelism settings
/// - Output paths
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoogleBooksConfig {
    /// Target language (BCP-47 tag: "en", "de", "fr", "es", etc.).
    pub language: String,

    /// N-gram orders to import.
    ///
    /// Valid range: 1..=5
    ///
    /// Unigrams (1-grams) provide:
    /// - Base probabilities in backoff/interpolation smoothing
    /// - Unknown word handling and OOV estimation
    /// - Vocabulary for dictionary extraction
    pub orders: RangeInclusive<u8>,

    /// Minimum frequency threshold.
    ///
    /// Google's default threshold is 40. Higher values filter out rare
    /// (potentially misspelled) n-grams, reducing storage size.
    pub min_count: u64,

    /// Optional year range filter (inclusive).
    ///
    /// If set, only n-gram occurrences from these years are counted.
    /// Useful for building models from specific time periods.
    ///
    /// Example: `Some((1990, 2020))` for modern English.
    pub year_range: Option<(u16, u16)>,

    /// Output path for PersistentARTrie file.
    ///
    /// This is the training-phase storage. For production, translate
    /// to PathMap using `PathMapTranslator`.
    pub output_path: PathBuf,

    /// Buffer pool size for PersistentARTrie.
    ///
    /// Default: 256 pages = 64MB.
    /// Increase for faster writes on systems with more RAM.
    pub buffer_pool_size: usize,

    /// Number of parallel download streams (for HTTP mode).
    ///
    /// Default: 4. Increase for faster networks, decrease for rate limiting.
    pub parallel_downloads: usize,

    /// Progress callback interval (every N n-grams).
    ///
    /// Default: 100_000. Lower values give more frequent updates
    /// but add slight overhead.
    pub progress_interval: usize,

    /// Whether to skip n-grams containing POS tags.
    ///
    /// Google Books n-grams include syntactic annotations like "_NOUN_".
    /// Set to `true` to filter these out for cleaner language models.
    ///
    /// Default: true.
    pub skip_pos_tags: bool,
}

impl Default for GoogleBooksConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            orders: 1..=5,
            min_count: 40,
            year_range: None,
            output_path: PathBuf::from("ngrams.artrie"),
            buffer_pool_size: 256,
            parallel_downloads: 4,
            progress_interval: 100_000,
            skip_pos_tags: true,
        }
    }
}

impl GoogleBooksConfig {
    /// Create a new configuration builder.
    pub fn builder() -> GoogleBooksConfigBuilder {
        GoogleBooksConfigBuilder::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate orders
        if *self.orders.start() < 1 || *self.orders.end() > 5 {
            return Err(ConfigError::InvalidOrders {
                start: *self.orders.start(),
                end: *self.orders.end(),
            });
        }

        // Validate year range
        if let Some((start, end)) = self.year_range {
            if start > end {
                return Err(ConfigError::InvalidYearRange { start, end });
            }
        }

        // Validate parallel downloads
        if self.parallel_downloads == 0 {
            return Err(ConfigError::ZeroParallelDownloads);
        }

        Ok(())
    }

    /// Get the checkpoint file path.
    pub fn checkpoint_path(&self) -> PathBuf {
        self.output_path.with_extension("checkpoint.json")
    }
}

/// Builder for GoogleBooksConfig.
#[derive(Default)]
pub struct GoogleBooksConfigBuilder {
    config: GoogleBooksConfig,
}

impl GoogleBooksConfigBuilder {
    /// Set the target language.
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.config.language = lang.into();
        self
    }

    /// Set the n-gram orders to import.
    pub fn orders(mut self, orders: RangeInclusive<u8>) -> Self {
        self.config.orders = orders;
        self
    }

    /// Set the minimum frequency threshold.
    pub fn min_count(mut self, count: u64) -> Self {
        self.config.min_count = count;
        self
    }

    /// Set an optional year range filter.
    pub fn year_range(mut self, start: u16, end: u16) -> Self {
        self.config.year_range = Some((start, end));
        self
    }

    /// Set the output path.
    pub fn output_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.output_path = path.into();
        self
    }

    /// Set the buffer pool size.
    pub fn buffer_pool_size(mut self, size: usize) -> Self {
        self.config.buffer_pool_size = size;
        self
    }

    /// Set the number of parallel downloads.
    pub fn parallel_downloads(mut self, count: usize) -> Self {
        self.config.parallel_downloads = count;
        self
    }

    /// Set the progress callback interval.
    pub fn progress_interval(mut self, interval: usize) -> Self {
        self.config.progress_interval = interval;
        self
    }

    /// Set whether to skip POS-tagged n-grams.
    pub fn skip_pos_tags(mut self, skip: bool) -> Self {
        self.config.skip_pos_tags = skip;
        self
    }

    /// Build and validate the configuration.
    pub fn build(self) -> Result<GoogleBooksConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Invalid n-gram order range.
    #[error("Invalid n-gram orders: {start}..={end} (must be 1..=5)")]
    InvalidOrders { start: u8, end: u8 },

    /// Invalid year range.
    #[error("Invalid year range: {start} > {end}")]
    InvalidYearRange { start: u16, end: u16 },

    /// Zero parallel downloads.
    #[error("Parallel downloads must be at least 1")]
    ZeroParallelDownloads,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GoogleBooksConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.orders, 1..=5);
        assert_eq!(config.min_count, 40);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = GoogleBooksConfig::builder()
            .language("de")
            .orders(2..=4)
            .min_count(100)
            .year_range(2000, 2020)
            .build()
            .unwrap();

        assert_eq!(config.language, "de");
        assert_eq!(config.orders, 2..=4);
        assert_eq!(config.min_count, 100);
        assert_eq!(config.year_range, Some((2000, 2020)));
    }

    #[test]
    fn test_invalid_orders() {
        let result = GoogleBooksConfig::builder()
            .orders(0..=5)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidOrders { .. })));

        let result = GoogleBooksConfig::builder()
            .orders(1..=6)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidOrders { .. })));
    }

    #[test]
    fn test_invalid_year_range() {
        let result = GoogleBooksConfig::builder()
            .year_range(2020, 2000)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidYearRange { .. })));
    }
}
