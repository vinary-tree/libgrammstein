//! Configuration for PDF extraction pipeline.

use super::backend::Backend;
use super::postprocess::PostProcessorConfig;
use super::router::RouterConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for PDF extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfConfig {
    /// Router configuration for backend selection.
    pub router: RouterConfig,

    /// Postprocessor configuration.
    pub postprocess: PostProcessorConfig,

    /// Path to Python executable (defaults to "python3").
    pub python_path: PathBuf,

    /// Path to Marker installation (optional, uses PATH if not set).
    pub marker_path: Option<PathBuf>,

    /// Path to Nougat installation (optional, uses PATH if not set).
    pub nougat_path: Option<PathBuf>,

    /// Temporary directory for intermediate files.
    pub temp_dir: Option<PathBuf>,

    /// Maximum time per page extraction.
    pub page_timeout: Duration,

    /// Maximum time for entire document.
    pub document_timeout: Duration,

    /// Number of parallel workers for batch processing.
    pub batch_workers: usize,

    /// Whether to keep intermediate files for debugging.
    pub keep_temp_files: bool,

    /// Maximum memory usage per worker (in bytes).
    pub max_memory_per_worker: Option<usize>,

    /// Device to use for neural backends (e.g., "cuda:0", "cpu").
    pub device: String,
}

impl Default for PdfConfig {
    fn default() -> Self {
        Self {
            router: RouterConfig::default(),
            postprocess: PostProcessorConfig::default(),
            python_path: PathBuf::from("python3"),
            marker_path: None,
            nougat_path: None,
            temp_dir: None,
            page_timeout: Duration::from_secs(60),
            document_timeout: Duration::from_secs(3600),
            batch_workers: num_cpus(),
            keep_temp_files: false,
            max_memory_per_worker: None,
            device: "cpu".to_string(),
        }
    }
}

impl PdfConfig {
    /// Create a new configuration builder.
    pub fn builder() -> PdfConfigBuilder {
        PdfConfigBuilder::new()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.page_timeout.is_zero() {
            return Err("page_timeout must be positive".into());
        }
        if self.document_timeout.is_zero() {
            return Err("document_timeout must be positive".into());
        }
        if self.batch_workers == 0 {
            return Err("batch_workers must be at least 1".into());
        }
        if self.router.math_density_threshold < 0.0 || self.router.math_density_threshold > 1.0 {
            return Err("math_density_threshold must be between 0.0 and 1.0".into());
        }
        Ok(())
    }

    /// Get the effective temporary directory.
    pub fn effective_temp_dir(&self) -> PathBuf {
        self.temp_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("libgrammstein-pdf"))
    }
}

/// Builder for PDF configuration.
#[derive(Debug, Clone, Default)]
pub struct PdfConfigBuilder {
    config: PdfConfig,
}

impl PdfConfigBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            config: PdfConfig::default(),
        }
    }

    /// Set the default backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.config.router.default_backend = backend;
        self
    }

    /// Set the math density threshold for router.
    pub fn math_density_threshold(mut self, threshold: f32) -> Self {
        self.config.router.math_density_threshold = threshold;
        self
    }

    /// Enable/disable parallel page processing.
    pub fn parallel_pages(mut self, enabled: bool) -> Self {
        self.config.router.parallel_pages = enabled;
        self
    }

    /// Set the Python executable path.
    pub fn python_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.python_path = path.into();
        self
    }

    /// Set the Marker installation path.
    pub fn marker_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.marker_path = Some(path.into());
        self
    }

    /// Set the Nougat installation path.
    pub fn nougat_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.nougat_path = Some(path.into());
        self
    }

    /// Set the temporary directory.
    pub fn temp_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.temp_dir = Some(path.into());
        self
    }

    /// Set the page timeout.
    pub fn page_timeout(mut self, timeout: Duration) -> Self {
        self.config.page_timeout = timeout;
        self
    }

    /// Set the document timeout.
    pub fn document_timeout(mut self, timeout: Duration) -> Self {
        self.config.document_timeout = timeout;
        self
    }

    /// Set the number of batch workers.
    pub fn batch_workers(mut self, workers: usize) -> Self {
        self.config.batch_workers = workers;
        self
    }

    /// Enable/disable keeping temporary files.
    pub fn keep_temp_files(mut self, keep: bool) -> Self {
        self.config.keep_temp_files = keep;
        self
    }

    /// Set maximum memory per worker.
    pub fn max_memory_per_worker(mut self, bytes: usize) -> Self {
        self.config.max_memory_per_worker = Some(bytes);
        self
    }

    /// Set the compute device.
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.config.device = device.into();
        self
    }

    /// Enable LaTeX validation in postprocessing.
    pub fn validate_latex(mut self, enabled: bool) -> Self {
        self.config.postprocess.validate_latex = enabled;
        self
    }

    /// Enable normalization in postprocessing.
    pub fn normalize_output(mut self, enabled: bool) -> Self {
        self.config.postprocess.normalize = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> PdfConfig {
        self.config
    }

    /// Build and validate the configuration.
    pub fn build_validated(self) -> Result<PdfConfig, String> {
        let config = self.build();
        config.validate()?;
        Ok(config)
    }
}

/// Get the number of available CPUs, with a reasonable default.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PdfConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = PdfConfigBuilder::new()
            .backend(Backend::Nougat)
            .math_density_threshold(0.5)
            .page_timeout(Duration::from_secs(120))
            .batch_workers(8)
            .build();

        assert_eq!(config.router.default_backend, Backend::Nougat);
        assert_eq!(config.router.math_density_threshold, 0.5);
        assert_eq!(config.page_timeout, Duration::from_secs(120));
        assert_eq!(config.batch_workers, 8);
    }

    #[test]
    fn test_validation() {
        let config = PdfConfigBuilder::new()
            .math_density_threshold(1.5) // Invalid
            .build();

        assert!(config.validate().is_err());
    }
}
