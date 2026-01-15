//! Terminal UI for long-running operations.
//!
//! This module provides a ratatui-based terminal UI for Google Books import
//! and other long-running operations. It subscribes to domain events and
//! renders them without coupling to import internals.
//!
//! ## Architecture
//!
//! The TUI follows a reactive, event-driven architecture:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                           Import Layer                               │
//! │  GoogleBooksImporter emits ImportEvent via broadcast channel         │
//! └───────────────────────────────────────┬─────────────────────────────┘
//!                                         │ ImportEvent
//!                                         ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                           TUI Layer                                  │
//! │  - TuiState: Applies events to derive display state                  │
//! │  - ImportTui: Renders state with ratatui, handles key input          │
//! │  - Sends ImportCommand back to importer via mpsc channel             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Bindings
//!
//! | Key | Action |
//! |-----|--------|
//! | `p` | Pause/resume import |
//! | `q` | Graceful quit (save checkpoint) |
//! | `+` | Increase parallelism |
//! | `-` | Decrease parallelism |
//! | `↑`/`↓` | Scroll log |
//! | `Esc` | Force quit (no checkpoint) |
//!
//! ## Example
//!
//! ```ignore
//! use tokio::sync::{broadcast, mpsc};
//! use libgrammstein::sources::google_books::{ImportEvent, ImportCommand};
//! use libgrammstein::cli::tui::ImportTui;
//!
//! let (event_tx, _) = broadcast::channel::<ImportEvent>(1024);
//! let (command_tx, command_rx) = mpsc::channel::<ImportCommand>(16);
//!
//! // Run import in background
//! let import_handle = tokio::spawn(async move {
//!     importer.import_http_reactive(event_tx.clone(), command_rx).await
//! });
//!
//! // Run TUI in foreground (subscribes to events)
//! let tui = ImportTui::new(command_tx, parallel_downloads);
//! tui.run(event_tx.subscribe())?;
//!
//! // TUI exits when import completes or user quits
//! let stats = import_handle.await??;
//! ```

mod app;
mod state;
mod widgets;

pub use app::ImportTui;
pub use state::{TuiState, WorkerState, WorkerStatus};
