//! Smart routing between PDF extraction backends.
//!
//! The router analyzes PDF documents and selects the optimal backend
//! based on document characteristics.

use super::backend::Backend;
use super::error::{PdfError, PdfResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for the PDF router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Default backend when Auto is not specified.
    pub default_backend: Backend,

    /// Math density threshold for switching to Nougat (0.0 to 1.0).
    /// Documents with math density above this threshold use Nougat.
    pub math_density_threshold: f32,

    /// Page count threshold for preferring Marker (speed).
    /// Documents with more pages than this prefer Marker.
    pub page_count_threshold: usize,

    /// Whether to enable parallel page processing.
    pub parallel_pages: bool,

    /// Whether to fallback to alternate backend on failure.
    pub fallback_on_error: bool,

    /// Timeout in seconds before switching backends.
    pub timeout_switch_seconds: u64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_backend: Backend::Auto,
            math_density_threshold: 0.3,
            page_count_threshold: 50,
            parallel_pages: true,
            fallback_on_error: true,
            timeout_switch_seconds: 120,
        }
    }
}

impl RouterConfig {
    /// Create a config that always uses Marker.
    pub fn marker_only() -> Self {
        Self {
            default_backend: Backend::Marker,
            fallback_on_error: false,
            ..Default::default()
        }
    }

    /// Create a config that always uses Nougat.
    pub fn nougat_only() -> Self {
        Self {
            default_backend: Backend::Nougat,
            fallback_on_error: false,
            ..Default::default()
        }
    }

    /// Create a config optimized for math-heavy documents.
    pub fn math_optimized() -> Self {
        Self {
            default_backend: Backend::Auto,
            math_density_threshold: 0.1, // Lower threshold = more Nougat usage
            fallback_on_error: true,
            ..Default::default()
        }
    }

    /// Create a config optimized for speed.
    pub fn speed_optimized() -> Self {
        Self {
            default_backend: Backend::Auto,
            math_density_threshold: 0.5, // Higher threshold = more Marker usage
            page_count_threshold: 20,
            parallel_pages: true,
            ..Default::default()
        }
    }
}

/// Decision made by the router.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    /// Selected backend.
    pub backend: Backend,
    /// Reason for selection.
    pub reason: String,
    /// Document analysis results.
    pub analysis: DocumentAnalysis,
    /// Fallback backend if primary fails.
    pub fallback: Option<Backend>,
}

/// Analysis of a PDF document.
#[derive(Debug, Clone, Default)]
pub struct DocumentAnalysis {
    /// Estimated number of pages.
    pub page_count: usize,
    /// Estimated math density (0.0 to 1.0).
    pub math_density: f32,
    /// Whether document contains tables.
    pub has_tables: bool,
    /// Whether document contains figures.
    pub has_figures: bool,
    /// Whether document appears to be an academic paper.
    pub is_academic: bool,
    /// File size in bytes.
    pub file_size: u64,
    /// PDF version if detectable.
    pub pdf_version: Option<String>,
}

/// Router for selecting PDF extraction backends.
pub struct PdfRouter {
    config: RouterConfig,
}

impl PdfRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: RouterConfig) -> PdfResult<Self> {
        // Validate that at least one backend is available
        if !Backend::Marker.is_available() && !Backend::Nougat.is_available() {
            return Err(PdfError::BackendNotAvailable {
                backend: "any".into(),
                reason: "No PDF extraction backends available. Install marker-pdf or nougat-ocr."
                    .into(),
            });
        }

        Ok(Self { config })
    }

    /// Create a router with default configuration.
    pub fn default_router() -> PdfResult<Self> {
        Self::new(RouterConfig::default())
    }

    /// Route a PDF to the appropriate backend.
    pub fn route<P: AsRef<Path>>(&self, path: P) -> PdfResult<RouterDecision> {
        let path = path.as_ref();

        // Check file exists
        if !path.exists() {
            return Err(PdfError::FileNotFound(path.to_path_buf()));
        }

        // Analyze document
        let analysis = self.analyze_document(path)?;

        // If explicit backend configured, use it
        if self.config.default_backend != Backend::Auto {
            let backend = self.config.default_backend;
            let fallback = if self.config.fallback_on_error {
                self.get_fallback(backend)
            } else {
                None
            };

            return Ok(RouterDecision {
                backend,
                reason: format!("Explicitly configured to use {}", backend.name()),
                analysis,
                fallback,
            });
        }

        // Auto-select based on analysis
        self.auto_select(analysis)
    }

    /// Analyze a PDF document.
    fn analyze_document<P: AsRef<Path>>(&self, path: P) -> PdfResult<DocumentAnalysis> {
        let path = path.as_ref();

        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len();

        // Try to get page count and detect math from PDF
        let (page_count, math_density, has_tables, has_figures) = self
            .quick_pdf_analysis(path)
            .unwrap_or((0, 0.0, false, false));

        // Heuristic: academic papers typically have certain patterns
        let is_academic = self.detect_academic_paper(path, &metadata);

        Ok(DocumentAnalysis {
            page_count,
            math_density,
            has_tables,
            has_figures,
            is_academic,
            file_size,
            pdf_version: None,
        })
    }

    /// Quick analysis of PDF content.
    fn quick_pdf_analysis<P: AsRef<Path>>(&self, path: P) -> PdfResult<(usize, f32, bool, bool)> {
        let path = path.as_ref();

        // Try to read first few KB of PDF to detect patterns
        let content = std::fs::read(path).map_err(|e| PdfError::Io(e))?;

        // Count pages (look for /Type /Page in PDF)
        let content_str = String::from_utf8_lossy(&content);
        let page_count = content_str.matches("/Type /Page").count().max(1);

        // Estimate math density by looking for math-related patterns
        let math_indicators = [
            "equation",
            "\\frac",
            "\\sum",
            "\\int",
            "\\alpha",
            "\\beta",
            "\\gamma",
            "\\delta",
            "\\theta",
            "\\pi",
            "\\sigma",
            "\\partial",
            "\\nabla",
            "\\infty",
            "\\sqrt",
            "\\lim",
            "\\rightarrow",
            "\\Rightarrow",
            "\\leq",
            "\\geq",
            "\\neq",
            "\\approx",
            "\\times",
            "\\cdot",
            "\\mathbb",
            "\\mathcal",
        ];

        let math_count: usize = math_indicators
            .iter()
            .map(|&pat| content_str.matches(pat).count())
            .sum();

        // Normalize math density (heuristic)
        let math_density = (math_count as f32 / (page_count as f32 * 100.0)).min(1.0);

        // Check for table indicators
        let has_tables = content_str.contains("/Table") || content_str.contains("tabular");

        // Check for figure indicators
        let has_figures = content_str.contains("/Figure")
            || content_str.contains("/Image")
            || content_str.contains("/XObject");

        Ok((page_count, math_density, has_tables, has_figures))
    }

    /// Detect if document appears to be an academic paper.
    fn detect_academic_paper<P: AsRef<Path>>(
        &self,
        path: P,
        _metadata: &std::fs::Metadata,
    ) -> bool {
        let path = path.as_ref();

        // Check filename patterns common in academic papers
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        let academic_patterns = [
            "arxiv",
            "paper",
            "manuscript",
            "preprint",
            "journal",
            "conference",
            "proceedings",
        ];

        let filename_lower = filename.to_lowercase();
        academic_patterns
            .iter()
            .any(|&pat| filename_lower.contains(pat))
    }

    /// Auto-select backend based on analysis.
    fn auto_select(&self, analysis: DocumentAnalysis) -> PdfResult<RouterDecision> {
        let marker_available = Backend::Marker.is_available();
        let nougat_available = Backend::Nougat.is_available();

        // If only one backend available, use it
        if marker_available && !nougat_available {
            return Ok(RouterDecision {
                backend: Backend::Marker,
                reason: "Only Marker is available".to_string(),
                analysis,
                fallback: None,
            });
        }

        if nougat_available && !marker_available {
            return Ok(RouterDecision {
                backend: Backend::Nougat,
                reason: "Only Nougat is available".to_string(),
                analysis,
                fallback: None,
            });
        }

        // Both available - select based on analysis
        let (backend, reason) = if analysis.math_density > self.config.math_density_threshold {
            (
                Backend::Nougat,
                format!(
                    "High math density ({:.1}% > {:.1}% threshold)",
                    analysis.math_density * 100.0,
                    self.config.math_density_threshold * 100.0
                ),
            )
        } else if analysis.page_count > self.config.page_count_threshold {
            (
                Backend::Marker,
                format!(
                    "Large document ({} pages > {} threshold), preferring speed",
                    analysis.page_count, self.config.page_count_threshold
                ),
            )
        } else if analysis.is_academic && analysis.math_density > 0.1 {
            (
                Backend::Nougat,
                "Academic paper with math content".to_string(),
            )
        } else {
            (
                Backend::Marker,
                "Default to Marker for general documents".to_string(),
            )
        };

        let fallback = if self.config.fallback_on_error {
            self.get_fallback(backend)
        } else {
            None
        };

        Ok(RouterDecision {
            backend,
            reason,
            analysis,
            fallback,
        })
    }

    /// Get fallback backend.
    fn get_fallback(&self, primary: Backend) -> Option<Backend> {
        match primary {
            Backend::Marker if Backend::Nougat.is_available() => Some(Backend::Nougat),
            Backend::Nougat if Backend::Marker.is_available() => Some(Backend::Marker),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_defaults() {
        let config = RouterConfig::default();
        assert_eq!(config.default_backend, Backend::Auto);
        assert_eq!(config.math_density_threshold, 0.3);
        assert!(config.fallback_on_error);
    }

    #[test]
    fn test_router_config_presets() {
        let marker = RouterConfig::marker_only();
        assert_eq!(marker.default_backend, Backend::Marker);
        assert!(!marker.fallback_on_error);

        let nougat = RouterConfig::nougat_only();
        assert_eq!(nougat.default_backend, Backend::Nougat);

        let math = RouterConfig::math_optimized();
        assert_eq!(math.math_density_threshold, 0.1);

        let speed = RouterConfig::speed_optimized();
        assert_eq!(speed.math_density_threshold, 0.5);
    }

    #[test]
    fn test_document_analysis_default() {
        let analysis = DocumentAnalysis::default();
        assert_eq!(analysis.page_count, 0);
        assert_eq!(analysis.math_density, 0.0);
        assert!(!analysis.has_tables);
        assert!(!analysis.is_academic);
    }
}
