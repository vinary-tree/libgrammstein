//! Plaintext corpus reader.
//!
//! Reads plain text files or directories of text files.

use super::{CorpusReader, Document, Normalizer, Tokenizer};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Reader for plain text files or directories.
///
/// Supports:
/// - Single text files
/// - Directories of text files (recursive)
/// - Streaming line-by-line reading
///
/// # Example
///
/// ```ignore
/// use libgrammstein::corpus::{PlaintextReader, CorpusReader};
///
/// // Read a single file
/// let reader = PlaintextReader::from_file("corpus.txt")?;
///
/// // Read a directory of files
/// let reader = PlaintextReader::from_directory("corpus/")?;
///
/// for sentence in reader.sentences() {
///     println!("{}", sentence);
/// }
/// ```
pub struct PlaintextReader {
    /// Paths to read from.
    paths: Vec<PathBuf>,

    /// Text normalizer.
    normalizer: Normalizer,

    /// Sentence tokenizer.
    tokenizer: Tokenizer,

    /// File extensions to include (empty = all).
    extensions: Vec<String>,
}

impl PlaintextReader {
    /// Create a reader for a single file.
    pub fn from_file(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            ));
        }

        Ok(Self {
            paths: vec![path],
            normalizer: Normalizer::new(),
            tokenizer: Tokenizer::new(),
            extensions: vec![],
        })
    }

    /// Create a reader for a directory of files.
    pub fn from_directory(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Directory not found: {}", path.display()),
            ));
        }

        let paths = Self::collect_files(path, &["txt", "text"])?;

        Ok(Self {
            paths,
            normalizer: Normalizer::new(),
            tokenizer: Tokenizer::new(),
            extensions: vec!["txt".to_string(), "text".to_string()],
        })
    }

    /// Create a reader for multiple paths.
    pub fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            normalizer: Normalizer::new(),
            tokenizer: Tokenizer::new(),
            extensions: vec![],
        }
    }

    /// Set custom normalizer.
    pub fn with_normalizer(mut self, normalizer: Normalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set custom tokenizer.
    pub fn with_tokenizer(mut self, tokenizer: Tokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set file extensions to include.
    pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
        self.extensions = extensions;
        self
    }

    /// Recursively collect files from a directory.
    fn collect_files(dir: &Path, extensions: &[&str]) -> std::io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                files.extend(Self::collect_files(&path, extensions)?);
            } else if path.is_file() {
                if extensions.is_empty() {
                    files.push(path);
                } else if let Some(ext) = path.extension() {
                    if extensions.iter().any(|e| ext == *e) {
                        files.push(path);
                    }
                }
            }
        }

        Ok(files)
    }

    /// Read a single file and return its content.
    fn read_file(path: &Path) -> std::io::Result<String> {
        fs::read_to_string(path)
    }
}

impl CorpusReader for PlaintextReader {
    fn documents(&self) -> Box<dyn Iterator<Item = Document> + Send + '_> {
        let normalizer = self.normalizer.clone();
        let paths = self.paths.clone();

        Box::new(
            paths
                .into_iter()
                .filter_map(move |path| match Self::read_file(&path) {
                    Ok(content) => {
                        let normalized = normalizer.normalize(&content);
                        Some(Document {
                            id: None,
                            title: path.file_stem().map(|s| s.to_string_lossy().to_string()),
                            content: normalized,
                            source: Some(path),
                        })
                    }
                    Err(e) => {
                        log::warn!("Failed to read file {}: {}", path.display(), e);
                        None
                    }
                }),
        )
    }

    fn sentences(&self) -> Box<dyn Iterator<Item = String> + Send + '_> {
        let tokenizer = self.tokenizer.clone();
        let documents = self.documents();

        Box::new(
            documents.flat_map(move |doc| tokenizer.sentences(&doc.content).collect::<Vec<_>>()),
        )
    }

    fn document_count(&self) -> Option<usize> {
        Some(self.paths.len())
    }
}

/// Iterator over lines from a file.
pub struct LineIterator {
    reader: BufReader<File>,
    normalizer: Normalizer,
}

impl LineIterator {
    /// Create a new line iterator for a file.
    pub fn new(path: impl AsRef<Path>, normalizer: Normalizer) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self { reader, normalizer })
    }
}

impl Iterator for LineIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    let normalized = self.normalizer.normalize(&line);
                    if !normalized.is_empty() {
                        return Some(normalized);
                    }
                    // Skip empty lines
                }
                Err(_) => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_file(dir: &Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).expect("Failed to create test file");
        write!(file, "{}", content).expect("Failed to write test file");
        path
    }

    #[test]
    fn test_read_single_file() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_file(dir.path(), "test.txt", "Hello world. This is a test.");

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let docs: Vec<_> = reader.documents().collect();

        assert_eq!(docs.len(), 1);
        // Normalizer preserves case; Tokenizer lowercases
        assert!(docs[0].content.contains("Hello world"));
    }

    #[test]
    fn test_read_directory() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        create_test_file(dir.path(), "a.txt", "First file.");
        create_test_file(dir.path(), "b.txt", "Second file.");

        let reader = PlaintextReader::from_directory(dir.path()).expect("Failed to create reader");
        let docs: Vec<_> = reader.documents().collect();

        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_sentences() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let path = create_test_file(
            dir.path(),
            "test.txt",
            "First sentence. Second sentence! Third sentence?",
        );

        let reader = PlaintextReader::from_file(&path).expect("Failed to create reader");
        let sentences: Vec<_> = reader.sentences().collect();

        assert_eq!(sentences.len(), 3);
    }
}
