//! PDF to LaTeX extraction pipeline for libgrammstein.
//!
//! This module provides integrations with PDF-to-LaTeX conversion tools,
//! enabling training language models from academic PDF documents (arXiv, papers).
//!
//! ## Available Backends
//!
//! - **Marker**: Fast layout-aware PDF parser with good general performance.
//!   Uses LayoutLMv3 + OCR for text extraction. Best for speed.
//!
//! - **Nougat**: Neural OCR for academic documents with excellent math support.
//!   Uses Swin Transformer encoder + mBART decoder. Best for math-heavy PDFs.
//!
//! ## Design Philosophy
//!
//! The PDF extraction pipeline follows a multi-stage approach:
//!
//! 1. **Analysis**: Detect document characteristics (math density, layout complexity)
//! 2. **Routing**: Select optimal backend based on analysis
//! 3. **Extraction**: Convert PDF to LaTeX/Markdown
//! 4. **Postprocessing**: Normalize output, fix common issues
//! 5. **Validation**: Verify LaTeX syntax using latex-parser
//!
//! ## Backend Selection
//!
//! The router automatically selects backends based on:
//! - Math density (high → Nougat, low → Marker)
//! - Document length (long → Marker for speed)
//! - Layout complexity (tables/figures → Marker)
//!
//! Manual backend selection is also supported via configuration.
//!
//! ## Concurrency
//!
//! - Parallel page processing within documents
//! - Concurrent document processing with configurable workers
//! - Progress callbacks for long-running conversions
//!
//! ## Example
//!
//! ```ignore
//! use libgrammstein::sources::pdf::{PdfExtractor, PdfConfig, Backend};
//!
//! let config = PdfConfig {
//!     backend: Backend::Auto,  // Let router decide
//!     math_density_threshold: 0.3,
//!     parallel_pages: true,
//!     ..Default::default()
//! };
//!
//! let extractor = PdfExtractor::new(config)?;
//! let result = extractor.extract("paper.pdf", |progress| {
//!     println!("Progress: {}%", progress.percent);
//! })?;
//!
//! println!("Extracted {} pages, {} equations", result.pages, result.equations);
//! println!("LaTeX output:\n{}", result.latex);
//! ```

mod backend;
mod config;
mod error;
mod postprocess;
mod router;

pub use backend::{
    Backend, BackendCapabilities, BackendInfo, ExtractedDocument, ExtractedPage, MarkerBackend,
    NougatBackend, PdfBackend,
};
pub use config::{PdfConfig, PdfConfigBuilder};
pub use error::{PdfError, PdfResult};
pub use postprocess::{PostProcessor, PostProcessorConfig};
pub use router::{PdfRouter, RouterConfig, RouterDecision};

use std::path::Path;

/// High-level PDF extractor combining routing, extraction, and postprocessing.
pub struct PdfExtractor {
    router: PdfRouter,
    postprocessor: PostProcessor,
}

impl PdfExtractor {
    /// Create a new PDF extractor with the given configuration.
    pub fn new(config: PdfConfig) -> PdfResult<Self> {
        let router = PdfRouter::new(config.router.clone())?;
        let postprocessor = PostProcessor::new(config.postprocess.clone());

        Ok(Self {
            router,
            postprocessor,
        })
    }

    /// Extract LaTeX from a PDF file.
    pub fn extract<P, F>(&self, path: P, progress: F) -> PdfResult<ExtractedDocument>
    where
        P: AsRef<Path>,
        F: Fn(ExtractionProgress) + Send + Sync,
    {
        let path = path.as_ref();

        // Route to appropriate backend
        let decision = self.router.route(path)?;

        // Extract using selected backend
        let mut doc = match decision.backend {
            Backend::Marker => {
                let backend = MarkerBackend::new()?;
                backend.extract(path, &progress)?
            }
            Backend::Nougat => {
                let backend = NougatBackend::new()?;
                backend.extract(path, &progress)?
            }
            Backend::Auto => {
                // Should not reach here after routing
                return Err(PdfError::Configuration(
                    "Auto backend should have been resolved by router".to_string(),
                ));
            }
        };

        // Apply postprocessing
        doc = self.postprocessor.process(doc)?;

        Ok(doc)
    }

    /// Extract from multiple PDFs in parallel.
    pub fn extract_batch<P, F>(&self, paths: &[P], progress: F) -> Vec<PdfResult<ExtractedDocument>>
    where
        P: AsRef<Path> + Sync,
        F: Fn(BatchProgress) + Send + Sync,
    {
        use rayon::prelude::*;

        let total = paths.len();

        paths
            .par_iter()
            .enumerate()
            .map(|(idx, path)| {
                let result = self.extract(path, |_| {});
                progress(BatchProgress {
                    current: idx + 1,
                    total,
                    path: path.as_ref().to_string_lossy().to_string(),
                    success: result.is_ok(),
                });
                result
            })
            .collect()
    }
}

/// Progress information for single document extraction.
#[derive(Debug, Clone)]
pub struct ExtractionProgress {
    /// Current page being processed.
    pub current_page: usize,
    /// Total number of pages.
    pub total_pages: usize,
    /// Percentage complete (0-100).
    pub percent: f32,
    /// Current stage of processing.
    pub stage: ExtractionStage,
}

/// Stage of the extraction process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionStage {
    /// Analyzing document structure.
    Analyzing,
    /// Extracting text and layout.
    Extracting,
    /// Recognizing mathematical formulas.
    MathOcr,
    /// Postprocessing output.
    Postprocessing,
    /// Validating LaTeX syntax.
    Validating,
}

/// Progress information for batch extraction.
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Number of documents processed so far.
    pub current: usize,
    /// Total number of documents.
    pub total: usize,
    /// Path of current document.
    pub path: String,
    /// Whether extraction succeeded.
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = PdfConfigBuilder::new()
            .backend(Backend::Auto)
            .math_density_threshold(0.5)
            .parallel_pages(true)
            .build();

        assert_eq!(config.router.default_backend, Backend::Auto);
        assert_eq!(config.router.math_density_threshold, 0.5);
    }
}
