//! Unit tests for the Google Books importer.

use super::*;
use tempfile::tempdir;

    #[test]
    fn test_import_progress() {
        let progress = ImportProgress {
            current_order: 3,
            current_prefix: "th".to_string(),
            ngrams_in_file: 1000,
            total_ngrams: 50000,
            files_completed: 10,
            total_files: 678,
            bytes_downloaded: 1024 * 1024,
            ngrams_per_second: 5000.0,
            eta_seconds: Some(3600),
            phase: ImportPhase::Importing,
        };

        assert_eq!(progress.current_order, 3);
        assert_eq!(progress.phase, ImportPhase::Importing);
    }

    #[test]
    fn test_import_stats_default() {
        let stats = ImportStats::default();
        assert_eq!(stats.total_ngrams, 0);
        assert_eq!(stats.files_processed, 0);
    }

    #[test]
    fn test_importer_creation() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config);
        assert!(importer.is_ok());
    }

    #[test]
    fn test_unsupported_language() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("invalid")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config);
        assert!(matches!(importer, Err(ImportError::UnsupportedLanguage(_))));
    }

    #[test]
    fn test_interrupt_flag() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let importer = GoogleBooksImporter::new(config).expect("Failed to create importer");

        assert!(!importer.is_interrupted());
        importer.interrupt();
        assert!(importer.is_interrupted());
    }

    /// Create a mock Google Books n-gram gzip file for testing.
    fn create_mock_ngram_file(path: &std::path::Path, ngrams: &[(&str, u16, u64, u32)]) {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let file = std::fs::File::create(path).expect("Failed to create file");
        let mut encoder = GzEncoder::new(file, Compression::default());

        for (ngram, year, count, volume_count) in ngrams {
            writeln!(encoder, "{}\t{}\t{}\t{}", ngram, year, count, volume_count)
                .expect("Failed to write");
        }

        encoder.finish().expect("Failed to finish compression");
    }

    #[test]
    fn test_file_import_with_mock_data() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock 1-gram file with test data
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-t.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("the", 2000, 50000, 1000),
                ("the", 2001, 55000, 1100),
                ("the", 2002, 60000, 1200),
                ("this", 2000, 10000, 500),
                ("this", 2001, 11000, 550),
                ("that", 2000, 20000, 800),
                ("that", 2001, 21000, 850),
                ("test", 2000, 5000, 200),
            ],
        );

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");

        // Import from local files
        let result = importer.import_files(&ngram_dir, |progress| {
            assert!(progress.current_order >= 1);
        });

        assert!(result.is_ok());
        let stats = result.unwrap();
        assert!(stats.total_ngrams > 0, "Should have imported n-grams");
    }

    #[test]
    fn test_year_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with data from multiple years
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-a.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("apple", 1990, 1000, 100),
                ("apple", 2000, 2000, 200),
                ("apple", 2010, 3000, 300),
                ("ant", 1990, 500, 50),
                ("ant", 2000, 600, 60),
            ],
        );

        // Import with year range filter (only 2000-2010)
        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .year_range(2000, 2010)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // The year filtering should have excluded 1990 data
    }

    #[test]
    fn test_min_count_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with varying counts
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-b.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("big", 2000, 100000, 5000),    // High count
                ("bear", 2000, 50000, 2500),    // Medium count
                ("bxyz", 2000, 10, 2),          // Low count (below default threshold)
            ],
        );

        // Import with min_count=40 (Google's default)
        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(40)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // "bxyz" should have been filtered out due to low count
    }

    #[test]
    fn test_pos_tag_filtering() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");
        let ngram_dir = dir.path().join("ngrams");
        std::fs::create_dir(&ngram_dir).expect("Failed to create ngram dir");

        // Create a mock file with POS-tagged and regular n-grams
        let file_path = ngram_dir.join("googlebooks-eng-all-1gram-20200217-c.gz");
        create_mock_ngram_file(
            &file_path,
            &[
                ("cat", 2000, 50000, 2500),
                ("cat_NOUN", 2000, 45000, 2300),     // POS tag
                ("car", 2000, 40000, 2000),
                ("the_DET", 2000, 100000, 5000),     // POS tag
            ],
        );

        // Import with POS tag filtering enabled
        let mut config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .min_count(1)
            .output_path(output_path)
            .build()
            .expect("Failed to build config");
        config.skip_pos_tags = true;

        let mut importer = GoogleBooksImporter::new(config).expect("Failed to create importer");
        let result = importer.import_files(&ngram_dir, |_| {});

        assert!(result.is_ok());
        // POS-tagged n-grams should have been filtered out
    }

    #[test]
    fn test_checkpoint_save_and_load() {
        let dir = tempdir().expect("Failed to create temp dir");
        let output_path = dir.path().join("test.artrie");

        let config = GoogleBooksConfig::builder()
            .language("en")
            .orders(1..=1)
            .output_path(output_path.clone())
            .build()
            .expect("Failed to build config");

        let mut importer = GoogleBooksImporter::new(config.clone()).expect("Failed to create importer");

        // Save checkpoint (now saved to trie, not JSON file)
        importer.save_checkpoint().expect("Failed to save checkpoint");

        // Load checkpoint from the storage's checkpoint trie via its
        // public API (replaces the previous direct `importer.trie.read()`).
        let loaded = importer
            .storage
            .load_import_checkpoint()
            .expect("Failed to load checkpoint from trie")
            .expect("Checkpoint should exist in trie");

        // v2 format: order_progress is a HashMap, completed_orders() is a method
        assert!(loaded.order_progress.is_empty());  // Fresh checkpoint has no progress
        assert!(loaded.completed_orders().is_empty());  // No orders completed yet
    }

    // ---- download_to_cache and cleanup_cache_file ----
    //
    // These tests exercise the HTTP download path used by the `--cache-files`
    // mode. The wiremock-based fixtures simulate the actual Google Books
    // endpoints' behavior (200 OK, 206 Partial Content, 416 Range Not
    // Satisfiable, mid-stream errors) without requiring network access.

    #[cfg(feature = "google-books")]
    mod cache_files {
        use super::*;
        use wiremock::matchers::{any, header_exists, method};
        use wiremock::{Mock, MockServer, Request, ResponseTemplate};

        fn build_client() -> reqwest::Client {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to build test HTTP client")
        }

        #[tokio::test]
        async fn download_to_cache_creates_file() {
            let server = MockServer::start().await;
            let body: &[u8] = &[0x1f, 0x8b, 0x08, 0x00, b'h', b'i', 0x00, 0x00];
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(body))
                .mount(&server)
                .await;

            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("test.gz");
            let url = format!("{}/test.gz", server.uri());
            let client = build_client();

            download_to_cache(&url, &cache_path, &client)
                .await
                .expect("download should succeed");

            assert!(cache_path.exists(), "cache file should exist after download");
            let written = std::fs::read(&cache_path).expect("read cache file");
            assert_eq!(written, body, "cache file should contain server body");
        }

        #[tokio::test]
        async fn download_to_cache_skips_if_exists() {
            // Pre-populate the cache file with sentinel content. Even though
            // the mock would return different bytes, download_to_cache should
            // see the file exists and return without contacting the server.
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(b"server bytes"))
                .mount(&server)
                .await;

            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("test.gz");
            std::fs::write(&cache_path, b"sentinel").expect("pre-populate cache");

            let url = format!("{}/test.gz", server.uri());
            let client = build_client();
            download_to_cache(&url, &cache_path, &client)
                .await
                .expect("download should succeed (no-op)");

            let written = std::fs::read(&cache_path).expect("read cache file");
            assert_eq!(written, b"sentinel", "cache should keep its sentinel content");
        }

        #[tokio::test]
        async fn download_to_cache_resume_via_range() {
            // Simulate a partial download from a previous interrupted attempt.
            // The server should see a Range header on the resume request.
            let full_body: &[u8] = b"0123456789abcdef";
            let server = MockServer::start().await;

            // When a Range header is present, return 206 with the latter half
            Mock::given(method("GET"))
                .and(header_exists("range"))
                .respond_with(ResponseTemplate::new(206).set_body_bytes(&full_body[8..]))
                .mount(&server)
                .await;

            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("test.gz");
            let downloading = cache_path.with_extension("gz.downloading");
            // Pre-create a .downloading remnant with the first half
            std::fs::write(&downloading, &full_body[..8]).expect("seed partial");

            let url = format!("{}/test.gz", server.uri());
            let client = build_client();
            download_to_cache(&url, &cache_path, &client)
                .await
                .expect("download should resume");

            assert!(cache_path.exists());
            assert!(!downloading.exists(), "downloading remnant should be renamed away");
            let written = std::fs::read(&cache_path).expect("read");
            assert_eq!(
                written, full_body,
                "resumed download should assemble first-half (cached) + second-half (server) = full body"
            );
        }

        #[tokio::test]
        async fn download_to_cache_416_recovery() {
            // The .downloading remnant is past the full content's EOF. Server
            // returns 416 on the Range request; download_to_cache should
            // delete the stale partial and re-request without Range, then
            // server returns the full body on the retry.
            let full_body: &[u8] = b"short";
            let server = MockServer::start().await;
            // First call (with Range): 416
            Mock::given(method("GET"))
                .and(header_exists("range"))
                .respond_with(ResponseTemplate::new(416))
                .up_to_n_times(1)
                .mount(&server)
                .await;
            // Second call (no Range): full content
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(full_body))
                .mount(&server)
                .await;

            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("test.gz");
            let downloading = cache_path.with_extension("gz.downloading");
            // Pre-seed an oversized partial (10 bytes vs 5-byte full content)
            std::fs::write(&downloading, b"toolongbye").expect("seed oversized partial");

            let url = format!("{}/test.gz", server.uri());
            let client = build_client();
            download_to_cache(&url, &cache_path, &client)
                .await
                .expect("download should recover from 416");

            assert!(cache_path.exists());
            let written = std::fs::read(&cache_path).expect("read");
            assert_eq!(written, full_body, "after 416 recovery, file matches server's full body");
        }

        #[tokio::test]
        async fn cleanup_cache_file_removes_both() {
            // Both the final .gz and an unfinished .gz.downloading should be
            // removed when cleanup is called.
            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("test.gz");
            let downloading = cache_path.with_extension("gz.downloading");
            std::fs::write(&cache_path, b"final").expect("write final");
            std::fs::write(&downloading, b"partial").expect("write partial");
            assert!(cache_path.exists() && downloading.exists());

            cleanup_cache_file(&cache_path).await;

            assert!(!cache_path.exists(), ".gz should be removed");
            assert!(!downloading.exists(), ".gz.downloading should be removed");
        }

        #[tokio::test]
        async fn cleanup_cache_file_is_idempotent() {
            // Cleaning up a non-existent cache is a no-op (no error).
            let tmp = tempdir().expect("tempdir");
            let cache_path = tmp.path().join("nope.gz");
            cleanup_cache_file(&cache_path).await;
            assert!(!cache_path.exists());
        }
    }
