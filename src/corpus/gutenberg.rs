//! Project Gutenberg corpus reader with boilerplate stripping.
//!
//! This module provides a reader for Project Gutenberg plain text files,
//! automatically stripping the standard header and footer boilerplate.

use std::fs;
use std::path::{Path, PathBuf};

use super::{CorpusReader, Document, Normalizer, Tokenizer};

/// Markers indicating the start of actual content in Gutenberg files.
const START_MARKERS: &[&str] = &[
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "***START OF THE PROJECT GUTENBERG",
    "*END*THE SMALL PRINT",
];

/// Markers indicating the end of actual content in Gutenberg files.
const END_MARKERS: &[&str] = &[
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
];

/// Reader for Project Gutenberg plain text files.
///
/// Automatically strips the standard Gutenberg header and footer boilerplate,
/// extracting only the actual book content.
///
/// # Example
///
/// ```ignore
/// use libgrammstein::corpus::{CorpusReader, GutenbergReader};
///
/// let reader = GutenbergReader::from_directory("./gutenberg/")?;
/// for sentence in reader.sentences() {
///     println!("{}", sentence);
/// }
/// ```
pub struct GutenbergReader {
    paths: Vec<PathBuf>,
    normalizer: Normalizer,
    tokenizer: Tokenizer,
}

impl GutenbergReader {
    /// Create a reader from a directory of Gutenberg text files.
    ///
    /// Recursively collects all `.txt` files from the directory.
    pub fn from_directory(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory not found: {}", dir.display()),
            ));
        }
        if !dir.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Not a directory: {}", dir.display()),
            ));
        }

        let paths = collect_txt_files(dir)?;
        log::info!("Found {} Gutenberg text files in {}", paths.len(), dir.display());

        Ok(Self {
            paths,
            normalizer: Normalizer::default(),
            tokenizer: Tokenizer::new(),
        })
    }

    /// Create a reader from a single Gutenberg text file.
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        Ok(Self {
            paths: vec![path.to_path_buf()],
            normalizer: Normalizer::default(),
            tokenizer: Tokenizer::new(),
        })
    }

    /// Create a reader from multiple file paths.
    pub fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            normalizer: Normalizer::default(),
            tokenizer: Tokenizer::new(),
        }
    }
}

impl CorpusReader for GutenbergReader {
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_> {
        let normalizer = self.normalizer.clone();
        let paths = self.paths.clone();

        Box::new(paths.into_iter().filter_map(move |path| {
            match read_gutenberg_file(&path) {
                Ok(content) => {
                    let stripped = strip_gutenberg_boilerplate(&content);
                    let normalized = normalizer.normalize(&stripped);

                    if normalized.is_empty() {
                        log::warn!("Empty content after stripping boilerplate: {}", path.display());
                        return None;
                    }

                    // Extract book ID from filename (e.g., "pg12345.txt" -> "12345")
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .and_then(|s| s.strip_prefix("pg"))
                        .map(|s| s.to_string())
                        .or_else(|| {
                            path.file_stem()
                                .and_then(|s| s.to_str())
                                .map(|s| s.to_string())
                        });

                    // Use filename as title
                    let title = path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string());

                    Some(Document {
                        id,
                        title,
                        content: normalized,
                        source: Some(path),
                    })
                }
                Err(e) => {
                    log::warn!("Failed to read Gutenberg file {}: {}", path.display(), e);
                    None
                }
            }
        }))
    }

    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        let tokenizer = self.tokenizer.clone();
        Box::new(self.documents().flat_map(move |doc| {
            tokenizer.sentences(&doc.content).collect::<Vec<_>>()
        }))
    }

    fn document_count(&self) -> Option<usize> {
        Some(self.paths.len())
    }
}

/// Recursively collect all .txt files from a directory.
fn collect_txt_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_txt_files_recursive(dir, &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_txt_files_recursive(dir: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_txt_files_recursive(&path, paths)?;
        } else if path.extension().map_or(false, |ext| ext == "txt") {
            paths.push(path);
        }
    }
    Ok(())
}

/// Read a Gutenberg text file.
fn read_gutenberg_file(path: &Path) -> std::io::Result<String> {
    fs::read_to_string(path)
}

/// Strip Project Gutenberg header and footer boilerplate.
///
/// Gutenberg files typically have:
/// - Header ending with "*** START OF" or similar
/// - Footer starting with "*** END OF" or similar
fn strip_gutenberg_boilerplate(text: &str) -> String {
    let mut start_idx = 0;
    let mut end_idx = text.len();

    // Find start of actual content
    for marker in START_MARKERS {
        if let Some(pos) = text.find(marker) {
            // Skip to the next line after the marker
            if let Some(newline_pos) = text[pos..].find('\n') {
                let candidate = pos + newline_pos + 1;
                // Skip any additional blank lines
                let remaining = &text[candidate..];
                let skip_blanks = remaining
                    .chars()
                    .take_while(|c| c.is_whitespace())
                    .count();
                start_idx = candidate + skip_blanks;
                break;
            }
        }
    }

    // Find end of actual content
    for marker in END_MARKERS {
        if let Some(pos) = text[start_idx..].find(marker) {
            // Trim back to skip any trailing whitespace before the marker
            let candidate = start_idx + pos;
            end_idx = text[..candidate].trim_end().len();
            break;
        }
    }

    // Ensure we have valid indices
    if start_idx >= end_idx || start_idx >= text.len() {
        // If we couldn't find markers, return the whole text
        return text.to_string();
    }

    text[start_idx..end_idx].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_gutenberg_boilerplate() {
        let text = r#"
The Project Gutenberg EBook of Test Book

*** START OF THE PROJECT GUTENBERG EBOOK ***

This is the actual content of the book.
It spans multiple lines.

*** END OF THE PROJECT GUTENBERG EBOOK ***

End of the Project Gutenberg EBook
"#;
        let stripped = strip_gutenberg_boilerplate(text);
        assert!(stripped.contains("This is the actual content"));
        assert!(stripped.contains("multiple lines"));
        assert!(!stripped.contains("START OF"));
        assert!(!stripped.contains("END OF"));
    }

    #[test]
    fn test_strip_no_markers() {
        let text = "Just some plain text without Gutenberg markers.";
        let stripped = strip_gutenberg_boilerplate(text);
        assert_eq!(stripped, text);
    }

    #[test]
    fn test_extract_book_id() {
        // Test that book ID extraction works
        let path = PathBuf::from("/some/path/pg12345.txt");
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("pg"))
            .map(|s| s.to_string());
        assert_eq!(id, Some("12345".to_string()));
    }
}
