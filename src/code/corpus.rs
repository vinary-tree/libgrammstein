//! Corpus reading for programming language code.
//!
//! This module provides iterators over source code files for training
//! code language models and PCFGs.

use super::ast::{CodeParser, ParsedCode};
use super::language::CodeLanguage;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

/// A code snippet with metadata.
#[derive(Debug, Clone)]
pub struct CodeSnippet {
    /// The source code content
    pub content: String,
    /// File path (if from a file)
    pub path: Option<PathBuf>,
    /// Language name
    pub language: String,
    /// Line count
    pub line_count: usize,
    /// Whether parsing succeeded
    pub is_valid: bool,
}

impl CodeSnippet {
    /// Creates a new code snippet from content.
    pub fn new(content: impl Into<String>, language: impl Into<String>) -> Self {
        let content = content.into();
        let line_count = content.lines().count();
        Self {
            content,
            path: None,
            language: language.into(),
            line_count,
            is_valid: true,
        }
    }

    /// Sets the file path.
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Marks the snippet as invalid.
    pub fn with_validity(mut self, valid: bool) -> Self {
        self.is_valid = valid;
        self
    }
}

/// Trait for reading code corpora.
pub trait CodeCorpusReader: Send + Sync {
    /// Returns an iterator over code snippets.
    fn snippets(&self) -> Box<dyn Iterator<Item = CodeSnippet> + Send + '_>;

    /// Returns an iterator over parsed code (ASTs).
    fn parsed(&self) -> Box<dyn Iterator<Item = ParsedCode> + Send + '_>;

    /// Returns the language being read.
    fn language_name(&self) -> &str;

    /// Returns the estimated number of snippets, if known.
    fn estimated_count(&self) -> Option<usize> {
        None
    }
}

/// Reads code from a directory tree.
///
/// **Note:** For large codebases (100K+ files), prefer [`StreamingDirectoryCorpusReader`]
/// which uses lazy iteration instead of collecting all paths upfront.
pub struct DirectoryCorpusReader<L: CodeLanguage> {
    root: PathBuf,
    language: Arc<L>,
    files: Vec<PathBuf>,
}

impl<L: CodeLanguage> DirectoryCorpusReader<L> {
    /// Creates a new reader for the given directory.
    ///
    /// **Warning:** This method collects all file paths upfront, which can use
    /// significant memory for large codebases. For 100K+ files, use
    /// [`StreamingDirectoryCorpusReader::new`] instead.
    pub fn new(root: impl AsRef<Path>, language: Arc<L>) -> io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let extensions: Vec<&str> = language.file_extensions().to_vec();
        let mut files = Vec::new();

        Self::collect_files(&root, &extensions, &mut files)?;

        Ok(Self {
            root,
            language,
            files,
        })
    }

    fn collect_files(dir: &Path, extensions: &[&str], files: &mut Vec<PathBuf>) -> io::Result<()> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    Self::collect_files(&path, extensions, files)?;
                } else if let Some(ext) = path.extension() {
                    if extensions.iter().any(|e| ext == *e) {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    /// Returns the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the number of files found.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

/// Streaming code corpus reader that uses lazy directory traversal.
///
/// Unlike [`DirectoryCorpusReader`], this reader uses `walkdir` to traverse
/// the directory tree lazily, processing files as they are discovered without
/// collecting all paths into memory first. This is crucial for large codebases
/// (100K+ files) where path collection alone could consume 10-100MB.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use libgrammstein::code::corpus::StreamingDirectoryCorpusReader;
/// use libgrammstein::code::languages::rust_lang::RustLanguage;
///
/// let language = Arc::new(RustLanguage::new());
/// let reader = StreamingDirectoryCorpusReader::new("/path/to/codebase", language);
///
/// // Files are processed lazily as the iterator is consumed
/// for snippet in reader.snippets() {
///     println!("Processing: {:?}", snippet.path);
/// }
/// ```
pub struct StreamingDirectoryCorpusReader<L: CodeLanguage> {
    root: PathBuf,
    language: Arc<L>,
    /// Extensions stored as owned strings for iterator lifetime
    extensions: HashSet<String>,
}

impl<L: CodeLanguage> StreamingDirectoryCorpusReader<L> {
    /// Creates a new streaming reader for the given directory.
    ///
    /// Unlike [`DirectoryCorpusReader::new`], this method does NOT traverse
    /// the directory upfront - traversal happens lazily during iteration.
    pub fn new(root: impl AsRef<Path>, language: Arc<L>) -> Self {
        let extensions: HashSet<String> = language
            .file_extensions()
            .iter()
            .map(|s| s.to_string())
            .collect();

        Self {
            root: root.as_ref().to_path_buf(),
            language,
            extensions,
        }
    }

    /// Returns the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns a lazy iterator over matching file paths.
    ///
    /// This uses `walkdir` for efficient directory traversal without
    /// buffering all paths in memory.
    fn file_iter(&self) -> impl Iterator<Item = PathBuf> + '_ {
        WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().is_file())
            .filter(move |entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| self.extensions.contains(ext))
                    .unwrap_or(false)
            })
            .map(|entry| entry.into_path())
    }
}

impl<L: CodeLanguage + 'static> CodeCorpusReader for StreamingDirectoryCorpusReader<L> {
    fn snippets(&self) -> Box<dyn Iterator<Item = CodeSnippet> + Send + '_> {
        let language_name = self.language.name().to_string();
        Box::new(self.file_iter().filter_map(move |path| {
            let content = fs::read_to_string(&path).ok()?;
            Some(CodeSnippet::new(content, &language_name).with_path(path))
        }))
    }

    fn parsed(&self) -> Box<dyn Iterator<Item = ParsedCode> + Send + '_> {
        let language = self.language.clone();
        Box::new(self.file_iter().filter_map(move |path| {
            let content = fs::read_to_string(&path).ok()?;
            let mut parser = CodeParser::new(language.clone()).ok()?;
            parser.parse(&content).ok()
        }))
    }

    fn language_name(&self) -> &str {
        self.language.name()
    }

    // Note: estimated_count() returns None because we don't traverse upfront
}

impl<L: CodeLanguage + 'static> CodeCorpusReader for DirectoryCorpusReader<L> {
    fn snippets(&self) -> Box<dyn Iterator<Item = CodeSnippet> + Send + '_> {
        let language_name = self.language.name().to_string();
        Box::new(self.files.iter().filter_map(move |path| {
            let content = fs::read_to_string(path).ok()?;
            Some(CodeSnippet::new(content, &language_name).with_path(path.clone()))
        }))
    }

    fn parsed(&self) -> Box<dyn Iterator<Item = ParsedCode> + Send + '_> {
        let language = self.language.clone();
        Box::new(self.files.iter().filter_map(move |path| {
            let content = fs::read_to_string(path).ok()?;
            let mut parser = CodeParser::new(language.clone()).ok()?;
            parser.parse(&content).ok()
        }))
    }

    fn language_name(&self) -> &str {
        self.language.name()
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.files.len())
    }
}

/// Reads code from a list of files.
pub struct FileListCorpusReader<L: CodeLanguage> {
    files: Vec<PathBuf>,
    language: Arc<L>,
}

impl<L: CodeLanguage> FileListCorpusReader<L> {
    /// Creates a new reader from a list of file paths.
    pub fn new(files: Vec<PathBuf>, language: Arc<L>) -> Self {
        Self { files, language }
    }

    /// Creates a reader from a file containing paths (one per line).
    pub fn from_file_list(list_file: impl AsRef<Path>, language: Arc<L>) -> io::Result<Self> {
        let content = fs::read_to_string(list_file)?;
        let files: Vec<PathBuf> = content
            .lines()
            .map(|s| PathBuf::from(s.trim()))
            .filter(|p| p.exists())
            .collect();
        Ok(Self { files, language })
    }
}

impl<L: CodeLanguage + 'static> CodeCorpusReader for FileListCorpusReader<L> {
    fn snippets(&self) -> Box<dyn Iterator<Item = CodeSnippet> + Send + '_> {
        let language_name = self.language.name().to_string();
        Box::new(self.files.iter().filter_map(move |path| {
            let content = fs::read_to_string(path).ok()?;
            Some(CodeSnippet::new(content, &language_name).with_path(path.clone()))
        }))
    }

    fn parsed(&self) -> Box<dyn Iterator<Item = ParsedCode> + Send + '_> {
        let language = self.language.clone();
        Box::new(self.files.iter().filter_map(move |path| {
            let content = fs::read_to_string(path).ok()?;
            let mut parser = CodeParser::new(language.clone()).ok()?;
            parser.parse(&content).ok()
        }))
    }

    fn language_name(&self) -> &str {
        self.language.name()
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.files.len())
    }
}

/// In-memory corpus reader for testing.
pub struct InMemoryCorpusReader<L: CodeLanguage> {
    snippets: Vec<CodeSnippet>,
    language: Arc<L>,
}

impl<L: CodeLanguage> InMemoryCorpusReader<L> {
    /// Creates a new in-memory reader.
    pub fn new(language: Arc<L>) -> Self {
        Self {
            snippets: Vec::new(),
            language,
        }
    }

    /// Adds a snippet to the corpus.
    pub fn add_snippet(&mut self, content: impl Into<String>) {
        let snippet = CodeSnippet::new(content, self.language.name());
        self.snippets.push(snippet);
    }

    /// Adds multiple snippets.
    pub fn add_snippets<I, S>(&mut self, contents: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for content in contents {
            self.add_snippet(content);
        }
    }
}

impl<L: CodeLanguage + 'static> CodeCorpusReader for InMemoryCorpusReader<L> {
    fn snippets(&self) -> Box<dyn Iterator<Item = CodeSnippet> + Send + '_> {
        Box::new(self.snippets.iter().cloned())
    }

    fn parsed(&self) -> Box<dyn Iterator<Item = ParsedCode> + Send + '_> {
        let language = self.language.clone();
        Box::new(self.snippets.iter().filter_map(move |snippet| {
            let mut parser = CodeParser::new(language.clone()).ok()?;
            parser.parse(&snippet.content).ok()
        }))
    }

    fn language_name(&self) -> &str {
        self.language.name()
    }

    fn estimated_count(&self) -> Option<usize> {
        Some(self.snippets.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_snippet_creation() {
        let snippet = CodeSnippet::new("fn main() {}", "rust").with_path("/test/main.rs");

        assert_eq!(snippet.content, "fn main() {}");
        assert_eq!(snippet.language, "rust");
        assert_eq!(snippet.line_count, 1);
        assert!(snippet.is_valid);
        assert_eq!(snippet.path, Some(PathBuf::from("/test/main.rs")));
    }

    #[test]
    fn test_code_snippet_multiline() {
        let code = "fn main() {\n    println!(\"hello\");\n}";
        let snippet = CodeSnippet::new(code, "rust");

        assert_eq!(snippet.line_count, 3);
    }
}
