//! Backend trait and implementations for PDF extraction.
//!
//! This module provides the interface for PDF-to-LaTeX backends and
//! implementations for Marker and Nougat.

use super::error::{PdfError, PdfResult};
use super::ExtractionProgress;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Stdio};

/// Available PDF extraction backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    /// Automatically select based on document analysis.
    #[default]
    Auto,
    /// Marker: Fast layout-aware parser.
    Marker,
    /// Nougat: Neural OCR optimized for math.
    Nougat,
}

impl Backend {
    /// Get the human-readable name of this backend.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Marker => "marker",
            Backend::Nougat => "nougat",
        }
    }

    /// Check if this backend is available on the system.
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Auto => Backend::Marker.is_available() || Backend::Nougat.is_available(),
            Backend::Marker => check_marker_available(),
            Backend::Nougat => check_nougat_available(),
        }
    }
}

/// Capabilities of a backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Maximum concurrent pages.
    pub max_concurrent_pages: usize,
    /// Whether math OCR is supported.
    pub supports_math_ocr: bool,
    /// Whether table extraction is supported.
    pub supports_tables: bool,
    /// Whether figure extraction is supported.
    pub supports_figures: bool,
    /// Approximate speed (pages per second on CPU).
    pub pages_per_second_cpu: f32,
    /// Approximate speed (pages per second on GPU).
    pub pages_per_second_gpu: f32,
    /// Whether GPU acceleration is available.
    pub gpu_available: bool,
}

/// Information about a backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend type.
    pub backend: Backend,
    /// Version string.
    pub version: String,
    /// Path to executable.
    pub path: String,
    /// Capabilities.
    pub capabilities: BackendCapabilities,
}

/// Extracted page from a PDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedPage {
    /// Page number (1-indexed).
    pub page_number: usize,
    /// Extracted LaTeX content.
    pub latex: String,
    /// Raw markdown (if available).
    pub markdown: Option<String>,
    /// Number of detected equations.
    pub equation_count: usize,
    /// Number of detected figures.
    pub figure_count: usize,
    /// Number of detected tables.
    pub table_count: usize,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
    /// Warnings during extraction.
    pub warnings: Vec<String>,
}

/// Extracted document from a PDF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDocument {
    /// Source file path.
    pub source_path: String,
    /// Backend used for extraction.
    pub backend: Backend,
    /// Extracted pages.
    pub pages: Vec<ExtractedPage>,
    /// Combined LaTeX content.
    pub latex: String,
    /// Document metadata.
    pub metadata: DocumentMetadata,
    /// Extraction statistics.
    pub stats: ExtractionStats,
}

impl ExtractedDocument {
    /// Get the total number of pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Get the total number of equations.
    pub fn equation_count(&self) -> usize {
        self.pages.iter().map(|p| p.equation_count).sum()
    }

    /// Get average confidence score.
    pub fn average_confidence(&self) -> f32 {
        if self.pages.is_empty() {
            return 0.0;
        }
        self.pages.iter().map(|p| p.confidence).sum::<f32>() / self.pages.len() as f32
    }
}

/// Document metadata extracted from PDF.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title.
    pub title: Option<String>,
    /// Authors.
    pub authors: Vec<String>,
    /// Abstract.
    pub abstract_text: Option<String>,
    /// Keywords.
    pub keywords: Vec<String>,
    /// Creation date.
    pub created: Option<String>,
    /// PDF version.
    pub pdf_version: Option<String>,
}

/// Statistics about the extraction process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionStats {
    /// Total processing time in milliseconds.
    pub processing_time_ms: u64,
    /// Number of successful pages.
    pub successful_pages: usize,
    /// Number of failed pages.
    pub failed_pages: usize,
    /// Total characters extracted.
    pub total_characters: usize,
    /// Total equations detected.
    pub total_equations: usize,
    /// Total figures detected.
    pub total_figures: usize,
    /// Total tables detected.
    pub total_tables: usize,
}

/// Trait for PDF extraction backends.
pub trait PdfBackend {
    /// Get information about this backend.
    fn info(&self) -> PdfResult<BackendInfo>;

    /// Extract content from a PDF file.
    fn extract<P, F>(&self, path: P, progress: &F) -> PdfResult<ExtractedDocument>
    where
        P: AsRef<Path>,
        F: Fn(ExtractionProgress) + Send + Sync;

    /// Check if this backend is available.
    fn is_available(&self) -> bool;

    /// Get the backend type.
    fn backend_type(&self) -> Backend;
}

/// Marker backend implementation.
pub struct MarkerBackend {
    python_path: String,
    marker_path: Option<String>,
    device: String,
}

impl MarkerBackend {
    /// Create a new Marker backend.
    pub fn new() -> PdfResult<Self> {
        if !check_marker_available() {
            return Err(PdfError::BackendNotAvailable {
                backend: "marker".into(),
                reason: "Marker is not installed. Install with: pip install marker-pdf".into(),
            });
        }

        Ok(Self {
            python_path: "python3".to_string(),
            marker_path: None,
            device: "cpu".to_string(),
        })
    }

    /// Create with custom Python path.
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// Set the Marker module path.
    pub fn with_marker_path(mut self, path: impl Into<String>) -> Self {
        self.marker_path = Some(path.into());
        self
    }

    /// Set the compute device.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }
}

impl PdfBackend for MarkerBackend {
    fn info(&self) -> PdfResult<BackendInfo> {
        let version = get_marker_version(&self.python_path)?;
        Ok(BackendInfo {
            backend: Backend::Marker,
            version,
            path: self.python_path.clone(),
            capabilities: BackendCapabilities {
                max_concurrent_pages: 4,
                supports_math_ocr: true,
                supports_tables: true,
                supports_figures: true,
                pages_per_second_cpu: 1.0,
                pages_per_second_gpu: 5.0,
                gpu_available: self.device.starts_with("cuda"),
            },
        })
    }

    fn extract<P, F>(&self, path: P, progress: &F) -> PdfResult<ExtractedDocument>
    where
        P: AsRef<Path>,
        F: Fn(ExtractionProgress) + Send + Sync,
    {
        let path = path.as_ref();

        if !path.exists() {
            return Err(PdfError::FileNotFound(path.to_path_buf()));
        }

        let start_time = std::time::Instant::now();

        // Notify start of extraction
        progress(ExtractionProgress {
            current_page: 0,
            total_pages: 0,
            percent: 0.0,
            stage: super::ExtractionStage::Analyzing,
        });

        // Build Python script to run Marker
        let script = format!(
            r#"
import sys
import json
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Initialize models
model_dict = create_model_dict()

# Convert PDF
converter = PdfConverter(artifact_dict=model_dict)
rendered = converter("{}")

# Output as JSON
result = {{
    "markdown": rendered.markdown,
    "metadata": rendered.metadata if hasattr(rendered, 'metadata') else {{}},
}}
print(json.dumps(result))
"#,
            path.display()
        );

        // Execute Marker via Python subprocess
        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .env("TORCH_DEVICE", &self.device)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| PdfError::BackendFailed {
                backend: "marker".into(),
                message: format!("Failed to execute Python: {}", e),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PdfError::BackendFailed {
                backend: "marker".into(),
                message: format!("Marker failed: {}", stderr),
            });
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(&stdout).map_err(|e| {
            PdfError::InvalidOutput {
                backend: "marker".into(),
                reason: format!("Failed to parse JSON output: {}", e),
            }
        })?;

        let markdown = result["markdown"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Convert markdown to LaTeX (basic conversion)
        let latex = markdown_to_latex(&markdown);

        // Count equations (simple heuristic)
        let equation_count = latex.matches("\\begin{equation}").count()
            + latex.matches("\\[").count()
            + latex.matches("$$").count();

        let processing_time = start_time.elapsed();

        progress(ExtractionProgress {
            current_page: 1,
            total_pages: 1,
            percent: 100.0,
            stage: super::ExtractionStage::Postprocessing,
        });

        Ok(ExtractedDocument {
            source_path: path.to_string_lossy().to_string(),
            backend: Backend::Marker,
            pages: vec![ExtractedPage {
                page_number: 1,
                latex: latex.clone(),
                markdown: Some(markdown),
                equation_count,
                figure_count: 0,
                table_count: 0,
                confidence: 0.9,
                warnings: Vec::new(),
            }],
            latex,
            metadata: DocumentMetadata::default(),
            stats: ExtractionStats {
                processing_time_ms: processing_time.as_millis() as u64,
                successful_pages: 1,
                failed_pages: 0,
                total_characters: 0,
                total_equations: equation_count,
                total_figures: 0,
                total_tables: 0,
            },
        })
    }

    fn is_available(&self) -> bool {
        check_marker_available()
    }

    fn backend_type(&self) -> Backend {
        Backend::Marker
    }
}

/// Nougat backend implementation.
pub struct NougatBackend {
    python_path: String,
    nougat_path: Option<String>,
    device: String,
    model_tag: String,
}

impl NougatBackend {
    /// Create a new Nougat backend.
    pub fn new() -> PdfResult<Self> {
        if !check_nougat_available() {
            return Err(PdfError::BackendNotAvailable {
                backend: "nougat".into(),
                reason: "Nougat is not installed. Install with: pip install nougat-ocr".into(),
            });
        }

        Ok(Self {
            python_path: "python3".to_string(),
            nougat_path: None,
            device: "cpu".to_string(),
            model_tag: "0.1.0-base".to_string(),
        })
    }

    /// Create with custom Python path.
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// Set the Nougat module path.
    pub fn with_nougat_path(mut self, path: impl Into<String>) -> Self {
        self.nougat_path = Some(path.into());
        self
    }

    /// Set the compute device.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Set the model tag.
    pub fn with_model_tag(mut self, tag: impl Into<String>) -> Self {
        self.model_tag = tag.into();
        self
    }
}

impl PdfBackend for NougatBackend {
    fn info(&self) -> PdfResult<BackendInfo> {
        let version = get_nougat_version(&self.python_path)?;
        Ok(BackendInfo {
            backend: Backend::Nougat,
            version,
            path: self.python_path.clone(),
            capabilities: BackendCapabilities {
                max_concurrent_pages: 1, // Nougat is slower, less concurrent
                supports_math_ocr: true,
                supports_tables: true,
                supports_figures: false, // Nougat focuses on text/math
                pages_per_second_cpu: 0.1,
                pages_per_second_gpu: 1.0,
                gpu_available: self.device.starts_with("cuda"),
            },
        })
    }

    fn extract<P, F>(&self, path: P, progress: &F) -> PdfResult<ExtractedDocument>
    where
        P: AsRef<Path>,
        F: Fn(ExtractionProgress) + Send + Sync,
    {
        let path = path.as_ref();

        if !path.exists() {
            return Err(PdfError::FileNotFound(path.to_path_buf()));
        }

        let start_time = std::time::Instant::now();

        progress(ExtractionProgress {
            current_page: 0,
            total_pages: 0,
            percent: 0.0,
            stage: super::ExtractionStage::Analyzing,
        });

        // Use nougat CLI
        let output = Command::new("nougat")
            .arg(path)
            .arg("--out")
            .arg("-") // Output to stdout
            .arg("--model")
            .arg(&self.model_tag)
            .arg("--no-skipping")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| PdfError::BackendFailed {
                backend: "nougat".into(),
                message: format!("Failed to execute nougat: {}", e),
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PdfError::BackendFailed {
                backend: "nougat".into(),
                message: format!("Nougat failed: {}", stderr),
            });
        }

        let mmd_content = String::from_utf8_lossy(&output.stdout).to_string();

        // Nougat outputs Mathpix Markdown, convert to LaTeX
        let latex = mathpix_markdown_to_latex(&mmd_content);

        let equation_count = latex.matches("\\begin{equation}").count()
            + latex.matches("\\[").count()
            + latex.matches("$$").count()
            + latex.matches("\\begin{align}").count();

        let processing_time = start_time.elapsed();

        progress(ExtractionProgress {
            current_page: 1,
            total_pages: 1,
            percent: 100.0,
            stage: super::ExtractionStage::Postprocessing,
        });

        Ok(ExtractedDocument {
            source_path: path.to_string_lossy().to_string(),
            backend: Backend::Nougat,
            pages: vec![ExtractedPage {
                page_number: 1,
                latex: latex.clone(),
                markdown: Some(mmd_content),
                equation_count,
                figure_count: 0,
                table_count: 0,
                confidence: 0.95, // Nougat is generally high quality for math
                warnings: Vec::new(),
            }],
            latex,
            metadata: DocumentMetadata::default(),
            stats: ExtractionStats {
                processing_time_ms: processing_time.as_millis() as u64,
                successful_pages: 1,
                failed_pages: 0,
                total_characters: 0,
                total_equations: equation_count,
                total_figures: 0,
                total_tables: 0,
            },
        })
    }

    fn is_available(&self) -> bool {
        check_nougat_available()
    }

    fn backend_type(&self) -> Backend {
        Backend::Nougat
    }
}

// === Helper functions ===

/// Check if Marker is available.
fn check_marker_available() -> bool {
    Command::new("python3")
        .args(["-c", "import marker"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Check if Nougat is available.
fn check_nougat_available() -> bool {
    Command::new("nougat")
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Get Marker version.
fn get_marker_version(python_path: &str) -> PdfResult<String> {
    let output = Command::new(python_path)
        .args(["-c", "import marker; print(marker.__version__)"])
        .output()
        .map_err(|e| PdfError::BackendFailed {
            backend: "marker".into(),
            message: format!("Failed to get version: {}", e),
        })?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Ok("unknown".to_string())
    }
}

/// Get Nougat version.
fn get_nougat_version(python_path: &str) -> PdfResult<String> {
    let output = Command::new(python_path)
        .args(["-c", "import nougat; print(nougat.__version__)"])
        .output()
        .map_err(|e| PdfError::BackendFailed {
            backend: "nougat".into(),
            message: format!("Failed to get version: {}", e),
        })?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Ok("unknown".to_string())
    }
}

/// Basic markdown to LaTeX conversion.
fn markdown_to_latex(markdown: &str) -> String {
    let mut latex = markdown.to_string();

    // Convert headers
    latex = latex.replace("# ", "\\section{");
    latex = latex.replace("## ", "\\subsection{");
    latex = latex.replace("### ", "\\subsubsection{");

    // Close headers (simple heuristic: end at newline)
    // This is a basic conversion - real implementation would be more sophisticated

    // Convert inline code
    latex = regex::Regex::new(r"`([^`]+)`")
        .unwrap()
        .replace_all(&latex, r"\texttt{$1}")
        .to_string();

    // Convert bold
    latex = regex::Regex::new(r"\*\*([^*]+)\*\*")
        .unwrap()
        .replace_all(&latex, r"\textbf{$1}")
        .to_string();

    // Convert italic
    latex = regex::Regex::new(r"\*([^*]+)\*")
        .unwrap()
        .replace_all(&latex, r"\textit{$1}")
        .to_string();

    latex
}

/// Convert Mathpix Markdown to LaTeX.
fn mathpix_markdown_to_latex(mmd: &str) -> String {
    let mut latex = mmd.to_string();

    // Mathpix markdown uses \[ \] for display math which is already LaTeX
    // and $ $ for inline math which is also LaTeX

    // Convert markdown-style headers to LaTeX
    latex = regex::Regex::new(r"^# (.+)$")
        .unwrap()
        .replace_all(&latex, r"\section{$1}")
        .to_string();

    latex = regex::Regex::new(r"^## (.+)$")
        .unwrap()
        .replace_all(&latex, r"\subsection{$1}")
        .to_string();

    latex = regex::Regex::new(r"^### (.+)$")
        .unwrap()
        .replace_all(&latex, r"\subsubsection{$1}")
        .to_string();

    // Convert markdown bold/italic
    latex = regex::Regex::new(r"\*\*(.+?)\*\*")
        .unwrap()
        .replace_all(&latex, r"\textbf{$1}")
        .to_string();

    latex = regex::Regex::new(r"\*(.+?)\*")
        .unwrap()
        .replace_all(&latex, r"\textit{$1}")
        .to_string();

    latex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_names() {
        assert_eq!(Backend::Auto.name(), "auto");
        assert_eq!(Backend::Marker.name(), "marker");
        assert_eq!(Backend::Nougat.name(), "nougat");
    }

    #[test]
    fn test_markdown_to_latex() {
        let md = "# Title\n**bold** and *italic*";
        let latex = markdown_to_latex(md);
        assert!(latex.contains("\\section{"));
        assert!(latex.contains("\\textbf{bold}"));
        assert!(latex.contains("\\textit{italic}"));
    }
}
