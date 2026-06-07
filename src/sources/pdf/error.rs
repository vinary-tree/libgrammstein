//! Error types for PDF extraction pipeline.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Result type for PDF operations.
pub type PdfResult<T> = Result<T, PdfError>;

/// Errors that can occur during PDF extraction.
#[derive(Error, Debug)]
pub enum PdfError {
    /// IO error reading/writing files.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// PDF file not found.
    #[error("PDF file not found: {0}")]
    FileNotFound(PathBuf),

    /// Invalid or corrupted PDF.
    #[error("Invalid PDF file: {0}")]
    InvalidPdf(String),

    /// Backend not available (not installed or not in PATH).
    #[error("Backend '{backend}' not available: {reason}")]
    BackendNotAvailable {
        /// Backend identifier.
        backend: String,
        /// Availability failure reason.
        reason: String,
    },

    /// Backend execution failed.
    #[error("Backend '{backend}' failed: {message}")]
    BackendFailed {
        /// Backend identifier.
        backend: String,
        /// Backend failure message.
        message: String,
    },

    /// Backend returned invalid output.
    #[error("Backend '{backend}' returned invalid output: {reason}")]
    InvalidOutput {
        /// Backend identifier.
        backend: String,
        /// Invalid-output reason.
        reason: String,
    },

    /// Timeout during extraction.
    #[error("Extraction timed out after {seconds}s")]
    Timeout {
        /// Timeout duration in seconds.
        seconds: u64,
    },

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Postprocessing error.
    #[error("Postprocessing failed: {0}")]
    Postprocess(String),

    /// LaTeX validation error.
    #[error("LaTeX validation failed: {0}")]
    Validation(String),

    /// Python environment error.
    #[error("Python environment error: {0}")]
    PythonEnvironment(String),

    /// Resource exhaustion (memory, disk space).
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),

    /// Page extraction failed for specific pages.
    #[error("Failed to extract pages: {pages:?}")]
    PageExtractionFailed {
        /// Pages that failed extraction.
        pages: Vec<usize>,
    },

    /// Unsupported PDF feature.
    #[error("Unsupported PDF feature: {0}")]
    UnsupportedFeature(String),
}

impl PdfError {
    /// Check if this error is recoverable (extraction can continue).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            PdfError::PageExtractionFailed { .. } | PdfError::Timeout { .. }
        )
    }

    /// Check if this error indicates a missing backend.
    pub fn is_backend_missing(&self) -> bool {
        matches!(self, PdfError::BackendNotAvailable { .. })
    }

    /// Get the backend name if this is a backend-related error.
    pub fn backend_name(&self) -> Option<&str> {
        match self {
            PdfError::BackendNotAvailable { backend, .. } => Some(backend),
            PdfError::BackendFailed { backend, .. } => Some(backend),
            PdfError::InvalidOutput { backend, .. } => Some(backend),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recoverable() {
        let err = PdfError::PageExtractionFailed { pages: vec![1, 2] };
        assert!(err.is_recoverable());

        let err = PdfError::InvalidPdf("corrupt".into());
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_backend_name() {
        let err = PdfError::BackendNotAvailable {
            backend: "marker".into(),
            reason: "not found".into(),
        };
        assert_eq!(err.backend_name(), Some("marker"));

        let err = PdfError::Io(io::Error::new(io::ErrorKind::NotFound, "test"));
        assert_eq!(err.backend_name(), None);
    }
}
