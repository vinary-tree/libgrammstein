//! Model registry for organizing models by language.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::LanguageTag;
use crate::error::Result;

/// Type of language model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// N-gram model only.
    Ngram,
    /// Embedding model only.
    Embedding,
    /// Hybrid model (N-gram + Embedding).
    Hybrid,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Ngram => write!(f, "ngram"),
            ModelType::Embedding => write!(f, "embedding"),
            ModelType::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Metadata stored with a trained model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Language tag (BCP 47).
    pub language: LanguageTag,

    /// Model type.
    pub model_type: ModelType,

    /// Training corpus sources.
    pub corpus_sources: Vec<String>,

    /// Training date.
    pub trained_at: DateTime<Utc>,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// N-gram order (for ngram and hybrid models).
    pub ngram_order: Option<usize>,

    /// Embedding dimension (for embedding and hybrid models).
    pub embedding_dim: Option<usize>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new metadata for an N-gram model.
    pub fn ngram(language: LanguageTag, vocab_size: usize, order: usize) -> Self {
        Self {
            language,
            model_type: ModelType::Ngram,
            corpus_sources: Vec::new(),
            trained_at: Utc::now(),
            vocab_size,
            ngram_order: Some(order),
            embedding_dim: None,
            extra: HashMap::new(),
        }
    }

    /// Create new metadata for an embedding model.
    pub fn embedding(language: LanguageTag, vocab_size: usize, dim: usize) -> Self {
        Self {
            language,
            model_type: ModelType::Embedding,
            corpus_sources: Vec::new(),
            trained_at: Utc::now(),
            vocab_size,
            ngram_order: None,
            embedding_dim: Some(dim),
            extra: HashMap::new(),
        }
    }

    /// Create new metadata for a hybrid model.
    pub fn hybrid(language: LanguageTag, vocab_size: usize, order: usize, dim: usize) -> Self {
        Self {
            language,
            model_type: ModelType::Hybrid,
            corpus_sources: Vec::new(),
            trained_at: Utc::now(),
            vocab_size,
            ngram_order: Some(order),
            embedding_dim: Some(dim),
            extra: HashMap::new(),
        }
    }

    /// Add a corpus source.
    pub fn with_corpus_source(mut self, source: impl Into<String>) -> Self {
        self.corpus_sources.push(source.into());
        self
    }

    /// Add extra metadata.
    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Entry in the model registry.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    /// Path to the model file.
    pub path: PathBuf,

    /// Language tag.
    pub language: LanguageTag,

    /// Model type.
    pub model_type: ModelType,

    /// File size in bytes.
    pub size_bytes: u64,

    /// Model metadata (if available).
    pub metadata: Option<ModelMetadata>,
}

/// Registry for discovering and managing installed models.
///
/// Scans a directory structure for model files and organizes them
/// by language for easy lookup.
#[derive(Debug)]
pub struct ModelRegistry {
    /// Root directory for models.
    root: PathBuf,

    /// Index of discovered models by language.
    index: HashMap<String, Vec<ModelEntry>>,
}

impl ModelRegistry {
    /// Scan a directory for models and build an index.
    ///
    /// Expected directory structure:
    /// ```text
    /// root/
    /// ├── en/
    /// │   ├── en-US/
    /// │   │   ├── ngram.bin
    /// │   │   └── hybrid.bin
    /// │   └── en-GB/
    /// │       └── ngram.bin
    /// ├── de/
    /// │   └── de-DE/
    /// │       └── hybrid.bin
    /// ```
    pub fn scan(root: &Path) -> Result<Self> {
        let mut index: HashMap<String, Vec<ModelEntry>> = HashMap::new();

        if !root.exists() {
            return Ok(Self {
                root: root.to_path_buf(),
                index,
            });
        }

        // Scan top-level language directories
        for lang_entry in fs::read_dir(root)? {
            let lang_entry = lang_entry?;
            if !lang_entry.file_type()?.is_dir() {
                continue;
            }

            let lang_name = lang_entry.file_name().to_string_lossy().to_string();

            // Scan dialect/region subdirectories
            for dialect_entry in fs::read_dir(lang_entry.path())? {
                let dialect_entry = dialect_entry?;
                let dialect_path = dialect_entry.path();

                if dialect_entry.file_type()?.is_dir() {
                    // Look for model files in dialect directory
                    Self::scan_model_files(&mut index, &dialect_path, &lang_name)?;
                } else if dialect_path.extension().map_or(false, |e| e == "bin") {
                    // Model file directly in language directory
                    Self::add_model_file(&mut index, &dialect_path, &lang_name, None)?;
                }
            }
        }

        Ok(Self {
            root: root.to_path_buf(),
            index,
        })
    }

    fn scan_model_files(
        index: &mut HashMap<String, Vec<ModelEntry>>,
        dir: &Path,
        lang: &str,
    ) -> Result<()> {
        let dialect = dir.file_name().map(|n| n.to_string_lossy().to_string());

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |e| e == "bin") {
                Self::add_model_file(index, &path, lang, dialect.as_deref())?;
            }
        }

        Ok(())
    }

    fn add_model_file(
        index: &mut HashMap<String, Vec<ModelEntry>>,
        path: &Path,
        lang: &str,
        dialect: Option<&str>,
    ) -> Result<()> {
        let metadata = fs::metadata(path)?;
        let size_bytes = metadata.len();

        // Infer model type from filename
        let model_type = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| {
                if s.contains("hybrid") {
                    ModelType::Hybrid
                } else if s.contains("embedding") || s.contains("embed") {
                    ModelType::Embedding
                } else {
                    ModelType::Ngram
                }
            })
            .unwrap_or(ModelType::Ngram);

        // Build language tag
        let language = if let Some(d) = dialect {
            d.parse().unwrap_or_else(|_| LanguageTag::new(lang))
        } else {
            LanguageTag::new(lang)
        };

        let entry = ModelEntry {
            path: path.to_path_buf(),
            language: language.clone(),
            model_type,
            size_bytes,
            metadata: None, // TODO: Load metadata from model file
        };

        index
            .entry(lang.to_string())
            .or_default()
            .push(entry);

        Ok(())
    }

    /// Get the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Find models by language (exact match).
    pub fn find(&self, lang: &LanguageTag) -> Vec<&ModelEntry> {
        self.index
            .get(lang.language())
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.language == *lang)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find best matching model (falls back to base language).
    pub fn find_best_match(&self, lang: &LanguageTag) -> Option<&ModelEntry> {
        // First try exact match
        let exact = self.find(lang);
        if !exact.is_empty() {
            // Prefer hybrid over ngram over embedding
            return exact
                .iter()
                .find(|e| e.model_type == ModelType::Hybrid)
                .or_else(|| exact.iter().find(|e| e.model_type == ModelType::Ngram))
                .or_else(|| exact.first())
                .copied();
        }

        // Fall back to base language
        let base = lang.base();
        if base != *lang {
            return self.find_best_match(&base);
        }

        // Try any model with matching base language
        self.index.get(lang.language()).and_then(|entries| {
            entries
                .iter()
                .find(|e| e.model_type == ModelType::Hybrid)
                .or_else(|| entries.iter().find(|e| e.model_type == ModelType::Ngram))
                .or_else(|| entries.first())
        })
    }

    /// List all available languages.
    pub fn languages(&self) -> Vec<&str> {
        self.index.keys().map(String::as_str).collect()
    }

    /// List all models.
    pub fn all_models(&self) -> Vec<&ModelEntry> {
        self.index.values().flat_map(|v| v.iter()).collect()
    }

    /// Get total number of models.
    pub fn count(&self) -> usize {
        self.index.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_metadata_ngram() {
        let meta = ModelMetadata::ngram(LanguageTag::new("en"), 10000, 5);
        assert_eq!(meta.model_type, ModelType::Ngram);
        assert_eq!(meta.ngram_order, Some(5));
        assert_eq!(meta.embedding_dim, None);
    }

    #[test]
    fn test_model_metadata_hybrid() {
        let meta = ModelMetadata::hybrid(LanguageTag::new("en"), 10000, 5, 100);
        assert_eq!(meta.model_type, ModelType::Hybrid);
        assert_eq!(meta.ngram_order, Some(5));
        assert_eq!(meta.embedding_dim, Some(100));
    }

    #[test]
    fn test_empty_registry() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::scan(temp_dir.path()).unwrap();
        assert_eq!(registry.count(), 0);
        assert!(registry.languages().is_empty());
    }

    #[test]
    fn test_registry_scan() {
        let temp_dir = TempDir::new().unwrap();

        // Create directory structure
        let en_us = temp_dir.path().join("en").join("en-US");
        fs::create_dir_all(&en_us).unwrap();
        fs::write(en_us.join("ngram.bin"), b"test").unwrap();
        fs::write(en_us.join("hybrid.bin"), b"test").unwrap();

        let de_de = temp_dir.path().join("de").join("de-DE");
        fs::create_dir_all(&de_de).unwrap();
        fs::write(de_de.join("ngram.bin"), b"test").unwrap();

        let registry = ModelRegistry::scan(temp_dir.path()).unwrap();

        assert_eq!(registry.count(), 3);
        assert_eq!(registry.languages().len(), 2);

        let en_models = registry.find(&LanguageTag::with_region("en", "US"));
        assert_eq!(en_models.len(), 2);
    }
}
