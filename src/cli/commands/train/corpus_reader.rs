//! Corpus-reader factory used by multiple train subcommands.

use std::path::Path;

use crate::cli::args::CorpusFormat;
use crate::cli::error::{CliError, CliResult};
use crate::corpus::{CorpusReader, GutenbergReader, PlaintextReader, WikipediaReader};

pub(super) fn create_corpus_reader(
    path: &str,
    format: CorpusFormat,
) -> CliResult<Box<dyn CorpusReader>> {
    let path_obj = Path::new(path);

    match format {
        CorpusFormat::Plaintext => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    PlaintextReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    PlaintextReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Wikipedia => {
            // Check if it's an HTTP URL
            #[cfg(feature = "http-corpus")]
            if path.starts_with("http://") || path.starts_with("https://") {
                return Ok(Box::new(
                    WikipediaReader::from_url(path, crate::corpus::WikipediaConfig::default())
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ));
            }

            // Local file
            if path_obj.exists() {
                Ok(Box::new(
                    WikipediaReader::new(path_obj).map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
        CorpusFormat::Gutenberg => {
            if path_obj.is_dir() {
                Ok(Box::new(
                    GutenbergReader::from_directory(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else if path_obj.exists() {
                Ok(Box::new(
                    GutenbergReader::from_file(path_obj)
                        .map_err(|e| CliError::corpus(e.to_string()))?,
                ))
            } else {
                Err(CliError::file_not_found(path_obj))
            }
        }
    }
}
