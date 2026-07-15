//! Cache-file mode: download `.gz` files to local disk before parsing.
//!
//! `--cache-files` mode decouples the HTTP download from the parser. A
//! worker first calls [`download_to_cache`] to write the raw `.gz` to a
//! local cache directory (atomically renaming `.gz.downloading` → `.gz`),
//! then streams from that cached file via
//! [`super::super::reader::stream_aggregated_from_cached_file`]. On
//! success or final failure the caller calls [`cleanup_cache_file`].

#![cfg(feature = "google-books")]

use std::path::Path;

use super::super::reader::ReaderError;
use super::super::task_manager::RetryAfter;
use super::ImportError;

/// Download a raw `.gz` file to local cache for later import.
///
/// Downloads to a `.gz.downloading` suffix first, then atomically renames
/// to the final path. This prevents partial downloads from being treated
/// as complete cached files.
///
/// If the cached file already exists at `cache_path`, the download is skipped.
///
/// Handles HTTP 429 (rate limiting) by extracting the Retry-After header and
/// returning `ImportError::Reader(ReaderError::RateLimited{..})` so existing
/// `is_retryable_error()` works unchanged.
#[cfg(feature = "google-books")]
pub(super) async fn download_to_cache(
    url: &str,
    cache_path: &Path,
    client: &reqwest::Client,
) -> Result<(), ImportError> {
    use tokio::io::AsyncWriteExt;

    // Skip if already cached
    if cache_path.exists() {
        log::debug!("Cache hit: {}", cache_path.display());
        return Ok(());
    }

    // Create parent directory
    if let Some(parent) = cache_path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            ImportError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to create cache directory {}: {}",
                    parent.display(),
                    e
                ),
            ))
        })?;
    }

    // Download to temporary suffix
    let downloading_path = cache_path.with_extension("gz.downloading");

    // Check for a partial download from a previous interrupted attempt
    let existing_len = tokio::fs::metadata(&downloading_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);

    // Send GET with conditional Range header for resume
    let mut request = client.get(url);
    if existing_len > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={}-", existing_len));
        log::debug!("Requesting resume from byte {} for {}", existing_len, url);
    }

    let response = request
        .send()
        .await
        .map_err(|e| ImportError::Reader(ReaderError::Http(format!("GET {}: {}", url, e))))?;

    let status = response.status();

    // Handle rate limiting
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(|s| {
                if let Ok(secs) = s.parse::<u64>() {
                    RetryAfter::Seconds(secs)
                } else {
                    RetryAfter::Seconds(60) // Default fallback
                }
            });

        return Err(ImportError::Reader(ReaderError::RateLimited {
            url: url.to_string(),
            retry_after,
        }));
    }

    // Handle 416 Range Not Satisfiable: partial file is at or beyond the full content.
    // Delete the stale partial file and re-request from scratch.
    let (response, status, existing_len) =
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && existing_len > 0 {
            log::warn!(
                "Range not satisfiable (partial file {} bytes) for {} — restarting from scratch",
                existing_len,
                url
            );
            let _ = tokio::fs::remove_file(&downloading_path).await;

            // Re-request without Range header
            let retry_resp = client.get(url).send().await.map_err(|e| {
                ImportError::Reader(ReaderError::Http(format!("GET {} (retry): {}", url, e)))
            })?;
            let retry_status = retry_resp.status();
            (retry_resp, retry_status, 0u64)
        } else {
            (response, status, existing_len)
        };

    if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(ImportError::Reader(ReaderError::Http(format!(
            "HTTP {} for {}",
            status, url
        ))));
    }

    // Open file in append mode (206 resume) or create mode (200 fresh download)
    use tokio::io::BufWriter;
    use tokio_stream::StreamExt;
    use tokio_util::io::StreamReader;

    let file = if status == reqwest::StatusCode::PARTIAL_CONTENT && existing_len > 0 {
        // 206 Partial Content — server confirmed resume; append to existing partial file
        log::info!("Resuming download from byte {} for {}", existing_len, url);
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&downloading_path)
            .await
            .map_err(|e| {
                ImportError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to open partial cache file {}: {}",
                        downloading_path.display(),
                        e
                    ),
                ))
            })?
    } else {
        // 200 OK — either fresh download or server ignored Range header; start from scratch
        if existing_len > 0 {
            log::debug!(
                "Server returned 200 (not 206) despite Range request — restarting download for {}",
                url
            );
        }
        tokio::fs::File::create(&downloading_path)
            .await
            .map_err(|e| {
                ImportError::Io(std::io::Error::new(
                    e.kind(),
                    format!(
                        "Failed to create cache file {}: {}",
                        downloading_path.display(),
                        e
                    ),
                ))
            })?
    };
    let mut writer = BufWriter::with_capacity(256 * 1024, file); // 256 KB buffer

    // Convert reqwest bytes stream to AsyncRead via StreamReader
    // (same pattern as reader.rs:876)
    let byte_stream = response.bytes_stream();
    let url_for_errors = url.to_string();
    let mapped_stream = byte_stream.map(move |result| {
        result.map_err(|e| {
            std::io::Error::other(format!("HTTP stream error for {}: {}", url_for_errors, e))
        })
    });
    let mut reader = StreamReader::new(mapped_stream);

    // Single efficient copy with internal buffering
    tokio::io::copy(&mut reader, &mut writer)
        .await
        .map_err(|e| {
            ImportError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to download {} to {}: {}",
                    url,
                    downloading_path.display(),
                    e
                ),
            ))
        })?;

    writer.flush().await.map_err(|e| {
        ImportError::Io(std::io::Error::new(
            e.kind(),
            format!(
                "Failed to flush cache file {}: {}",
                downloading_path.display(),
                e
            ),
        ))
    })?;
    drop(writer);

    // Atomic rename: .gz.downloading → .gz
    tokio::fs::rename(&downloading_path, cache_path)
        .await
        .map_err(|e| {
            ImportError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "Failed to rename {} → {}: {}",
                    downloading_path.display(),
                    cache_path.display(),
                    e
                ),
            ))
        })?;

    log::debug!("Cached {} → {}", url, cache_path.display());
    Ok(())
}

/// Clean up cached file and its `.downloading` remnant.
///
/// Best-effort: logs warnings on failure but does not return errors.
#[cfg(feature = "google-books")]
pub(super) async fn cleanup_cache_file(cache_path: &Path) {
    // Remove the cached .gz file
    if cache_path.exists() {
        if let Err(e) = tokio::fs::remove_file(cache_path).await {
            log::warn!(
                "Failed to remove cached file {}: {}",
                cache_path.display(),
                e
            );
        }
    }
    // Remove any partial .downloading remnant
    let downloading_path = cache_path.with_extension("gz.downloading");
    if downloading_path.exists() {
        if let Err(e) = tokio::fs::remove_file(&downloading_path).await {
            log::warn!(
                "Failed to remove downloading remnant {}: {}",
                downloading_path.display(),
                e
            );
        }
    }
}
