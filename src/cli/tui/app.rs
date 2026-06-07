//! Main TUI application.
//!
//! Handles the render loop, key input processing, and event subscription.

use std::io;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table};
use tokio::sync::{broadcast, mpsc};

use super::state::{TuiState, WorkerStatus};
use super::widgets;
use crate::sources::google_books::{ImportCommand, ImportEvent};

/// Import TUI application.
pub struct ImportTui {
    /// TUI state.
    state: TuiState,

    /// Command sender for controlling the import.
    command_tx: mpsc::Sender<ImportCommand>,
}

impl ImportTui {
    /// Create a new TUI application.
    pub fn new(
        command_tx: mpsc::Sender<ImportCommand>,
        language: &str,
        min_order: u8,
        max_order: u8,
        parallel_downloads: usize,
    ) -> Self {
        Self {
            state: TuiState::new(language, min_order, max_order, parallel_downloads),
            command_tx,
        }
    }

    /// Run the TUI until import completes or user quits.
    ///
    /// This method blocks the current thread and handles:
    /// - Event subscription via broadcast channel
    /// - Key input handling
    /// - Terminal rendering at ~60 FPS
    ///
    /// Returns `Ok(true)` if import completed normally, `Ok(false)` if cancelled.
    pub fn run(mut self, mut event_rx: broadcast::Receiver<ImportEvent>) -> io::Result<bool> {
        // Initialize terminal
        let mut terminal = ratatui::init();

        // Clear any pre-existing terminal content
        terminal.clear()?;

        let result = self.run_loop(&mut terminal, &mut event_rx);

        // Restore terminal
        ratatui::restore();

        result
    }

    /// Main event loop.
    fn run_loop(
        &mut self,
        terminal: &mut ratatui::DefaultTerminal,
        event_rx: &mut broadcast::Receiver<ImportEvent>,
    ) -> io::Result<bool> {
        loop {
            // Drain all pending events (non-blocking)
            loop {
                match event_rx.try_recv() {
                    Ok(import_event) => {
                        // Log phase-critical events
                        match &import_event {
                            ImportEvent::PhaseChanged { phase } => {
                                log::debug!("[TUI-APP] Received PhaseChanged: '{}'", phase);
                            }
                            ImportEvent::MknStarted { .. } => {
                                log::debug!("[TUI-APP] Received MknStarted");
                            }
                            ImportEvent::MknCompleted { .. } => {
                                log::debug!("[TUI-APP] Received MknCompleted");
                            }
                            ImportEvent::MergeStarted { shard_count, .. } => {
                                log::debug!(
                                    "[TUI-APP] Received MergeStarted: {} shards",
                                    shard_count
                                );
                            }
                            ImportEvent::MergeCompleted { .. } => {
                                log::debug!("[TUI-APP] Received MergeCompleted");
                            }
                            ImportEvent::AllWorkCompleted { .. } => {
                                log::debug!("[TUI-APP] Received AllWorkCompleted");
                            }
                            ImportEvent::WorkerExited { worker_id } => {
                                log::debug!("[TUI-APP] Received WorkerExited: {}", worker_id);
                            }
                            _ => {}
                        }
                        self.state.apply_event(&import_event);

                        // Note: We no longer auto-exit on is_finished().
                        // Instead, we wait for AllWorkCompleted which sets
                        // show_completion_dialog, requiring user acknowledgment.
                    }
                    Err(broadcast::error::TryRecvError::Empty) => break,
                    Err(broadcast::error::TryRecvError::Lagged(n)) => {
                        // We missed some events due to slow rendering, log it
                        log::warn!("[TUI-APP] LAGGED: missed {} events!", n);
                    }
                    Err(broadcast::error::TryRecvError::Closed) => {
                        log::debug!(
                            "[TUI-APP] Channel closed, show_completion_dialog={}, is_finished={}",
                            self.state.show_completion_dialog,
                            self.state.is_finished()
                        );
                        // Channel closed - if we have a completion dialog, show it
                        // Otherwise, the import must have finished abruptly
                        if !self.state.show_completion_dialog && self.state.is_finished() {
                            // Import completed without AllWorkCompleted event
                            // (e.g., non-sharded mode or error)
                            // Brief pause to show final state, then exit
                            terminal.draw(|frame| self.render(frame))?;
                            std::thread::sleep(Duration::from_millis(500));
                            return Ok(self.state.completed);
                        }
                        // If dialog is showing, continue loop to handle user input
                    }
                }
            }

            // Render current state
            terminal.draw(|frame| self.render(frame))?;

            // Handle key input (non-blocking, with timeout for ~60 FPS)
            if event::poll(Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    // Only handle key press events, not release
                    if key.kind == KeyEventKind::Press {
                        if self.handle_key(key.code)? {
                            return Ok(self.state.completed);
                        }
                    }
                }
            }
        }
    }

    /// Handle a key press.
    ///
    /// Returns `true` if the TUI should exit.
    fn handle_key(&mut self, key: KeyCode) -> io::Result<bool> {
        // If completion dialog is showing, only accept exit keys
        if self.state.show_completion_dialog {
            match key {
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                    return Ok(true); // Exit TUI
                }
                _ => return Ok(false), // Ignore other keys
            }
        }

        match key {
            KeyCode::Char('p') | KeyCode::Char('P') => {
                let cmd = if self.state.paused {
                    ImportCommand::Resume
                } else {
                    ImportCommand::Pause
                };
                let _ = self.command_tx.try_send(cmd);
            }

            KeyCode::Char('q') | KeyCode::Char('Q') => {
                // Graceful quit
                let _ = self.command_tx.try_send(ImportCommand::Cancel);
                return Ok(true);
            }

            KeyCode::Esc => {
                // Force quit
                let _ = self.command_tx.try_send(ImportCommand::ForceQuit);
                return Ok(true);
            }

            KeyCode::Char('+') | KeyCode::Char('=') => {
                let new_parallelism = self.state.parallel_downloads + 1;
                let _ = self
                    .command_tx
                    .try_send(ImportCommand::SetParallelism(new_parallelism));
                self.state.parallel_downloads = new_parallelism;

                // Add a new worker slot
                self.state.workers.push(super::state::WorkerState {
                    id: new_parallelism - 1,
                    status: WorkerStatus::Idle,
                    prefix: None,
                    order: None,
                    progress: 0.0,
                    bytes_downloaded: 0,
                    total_bytes: None,
                    ngram_count: 0,
                });
            }

            KeyCode::Char('-') | KeyCode::Char('_') => {
                if self.state.parallel_downloads > 1 {
                    let new_parallelism = self.state.parallel_downloads - 1;
                    let _ = self
                        .command_tx
                        .try_send(ImportCommand::SetParallelism(new_parallelism));
                    self.state.parallel_downloads = new_parallelism;
                    // Worker slot is removed when WorkerExited event is received,
                    // not immediately here. This keeps the display in sync with
                    // the actual worker state (worker finishes current job first).
                }
            }

            KeyCode::Up => {
                self.state.scroll_log_up();
            }

            KeyCode::Down => {
                self.state.scroll_log_down();
            }

            _ => {}
        }

        Ok(false)
    }

    /// Render the TUI.
    fn render(&self, frame: &mut Frame) {
        let area = frame.area();

        // Calculate height needed for per-order progress display
        // Each order takes 1 line, plus 2 for borders
        let order_count = (self.state.max_order - self.state.min_order + 1) as u16;
        let progress_height = (order_count + 2).max(3);

        // Create main layout
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),               // Header
                Constraint::Length(progress_height), // Per-order progress
                Constraint::Min(8),                  // Main content (stats + workers)
                Constraint::Length(8),               // Log
            ])
            .split(area);

        // Header
        self.render_header(frame, chunks[0]);

        // Overall progress bar
        self.render_progress(frame, chunks[1]);

        // Main content area (stats on left, workers on right)
        let main_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[2]);

        self.render_stats(frame, main_chunks[0]);
        self.render_workers(frame, main_chunks[1]);

        // Log panel
        self.render_log(frame, chunks[3]);

        // Render completion dialog on top if needed
        if self.state.show_completion_dialog {
            self.render_completion_dialog(frame, area);
        }
    }

    /// Render the header.
    fn render_header(&self, frame: &mut Frame, area: Rect) {
        // Check for error first - show prominently
        if let Some(ref error) = self.state.error_message {
            let title = format!(" ERROR: {} - Press [Q] to quit", error);

            let header = Paragraph::new(title)
                .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Red)),
                );

            frame.render_widget(header, area);
            return;
        }

        let status = if self.state.paused {
            " [PAUSED]"
        } else if self.state.completed {
            " [COMPLETE]"
        } else if self.state.cancelled {
            " [CANCELLED]"
        } else {
            ""
        };

        // Add warning indicator if there are failed prefixes
        let warning = if self.state.failed_prefixes_count > 0 {
            format!(" [!{} FAILED]", self.state.failed_prefixes_count)
        } else {
            String::new()
        };

        let title = format!(
            " Google Books Import - {} {}-{}grams | {}  [P]ause  [Q]uit  [+/-] Workers{}{}",
            self.state.language.to_uppercase(),
            self.state.min_order,
            self.state.max_order,
            self.state.display_phase(),
            status,
            warning
        );

        // Use yellow/warning color if there are failed prefixes
        let header_style = if self.state.failed_prefixes_count > 0 {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        };

        let header = Paragraph::new(title)
            .style(header_style)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(header, area);
    }

    /// Render the overall progress bar.
    fn render_progress(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Order Progress ")
            .borders(Borders::ALL);

        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Build progress lines for each order
        let mut lines: Vec<Line> = Vec::new();

        for order in self.state.min_order..=self.state.max_order {
            let (files_completed, total_files, is_complete, files_succeeded, files_skipped) =
                if let Some(progress) = self.state.order_progress.get(&order) {
                    (
                        progress.files_completed,
                        progress.total_files,
                        progress.is_complete,
                        progress.files_succeeded,
                        progress.files_skipped,
                    )
                } else {
                    // Order hasn't started yet
                    (0, 0, false, 0, 0)
                };

            let percent = if total_files > 0 {
                (files_completed as f64 / total_files as f64 * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };

            // Create segmented progress bar with colors for success/skipped
            let bar_width = 20;
            let bar_spans = widgets::segmented_progress_bar(
                files_succeeded,
                files_skipped,
                total_files,
                bar_width,
            );

            // Determine status text and color
            let (status, status_style) = if is_complete {
                ("DONE", Style::default().fg(Color::Green))
            } else if files_completed > 0 || total_files > 0 {
                ("ACTIVE", Style::default().fg(Color::Yellow))
            } else {
                ("PENDING", Style::default().fg(Color::DarkGray))
            };

            // Format: "2-grams: ████░░░░░░ 340/676 (50%) ACTIVE"
            // The progress bar now has colored segments: green=success, yellow=skipped
            let mut spans = vec![Span::styled(
                format!("{}-grams: ", order),
                Style::default().fg(Color::Cyan),
            )];
            spans.extend(bar_spans);
            spans.push(Span::raw(format!(
                " {}/{} ({:.0}%) ",
                files_completed, total_files, percent
            )));
            spans.push(Span::styled(status, status_style));

            let line = Line::from(spans);

            lines.push(line);
        }

        let text = Text::from(lines);
        let paragraph = Paragraph::new(text);
        frame.render_widget(paragraph, inner);
    }

    /// Render statistics panel.
    fn render_stats(&self, frame: &mut Frame, area: Rect) {
        let stats_block = Block::default().title(" Statistics ").borders(Borders::ALL);

        let inner = stats_block.inner(area);
        frame.render_widget(stats_block, area);

        // Split into stats text and sparkline
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(3)])
            .split(inner);

        // Stats table - create bindings for formatted strings to extend lifetimes
        let total_ngrams_str = format_count(self.state.total_ngrams);
        let unique_ngrams_str = format_count(self.state.unique_ngrams);
        // Use derived total to stay in sync with Order Progress display
        let files_completed_str = format!("{}", self.state.total_files_completed());
        let rate_str = format!(
            "{}/s",
            widgets::format_si(self.state.ngrams_per_second.round() as u64)
        );
        let elapsed_str = format_duration(self.state.elapsed);
        let workers_str = format!("{}", self.state.parallel_downloads);
        let warnings_str = self.state.prefix_warnings_summary().unwrap_or_default();

        let mut rows = vec![
            Row::new(vec!["Total n-grams:", total_ngrams_str.as_str()]),
            Row::new(vec!["Unique n-grams:", unique_ngrams_str.as_str()]),
            Row::new(vec!["Files completed:", files_completed_str.as_str()]),
            Row::new(vec!["Rate:", rate_str.as_str()]),
            Row::new(vec!["Elapsed:", elapsed_str.as_str()]),
            Row::new(vec!["Workers:", workers_str.as_str()]),
        ];

        // Add backoff queue row (jobs waiting on exponential backoff in current session)
        let backoff_queue_str = format!("{}", self.state.backoff_queue_count);
        let backoff_queue_row = if self.state.backoff_queue_count > 0 {
            Row::new(vec!["Backoff queue:", backoff_queue_str.as_str()])
                .style(Style::default().fg(Color::Yellow))
        } else {
            Row::new(vec!["Backoff queue:", backoff_queue_str.as_str()])
        };
        rows.push(backoff_queue_row);

        // Add checkpoint retries row (from previous session)
        let checkpoint_retries = self.state.failed_prefixes_count
            + self.state.retrying_prefixes_count
            + self.state.recovering_prefixes_count;
        let checkpoint_retries_str = format!("{}", checkpoint_retries);
        let checkpoint_retries_row = if checkpoint_retries > 0 {
            Row::new(vec!["Checkpoint retries:", checkpoint_retries_str.as_str()])
                .style(Style::default().fg(Color::Yellow))
        } else {
            Row::new(vec!["Checkpoint retries:", checkpoint_retries_str.as_str()])
        };
        rows.push(checkpoint_retries_row);

        // Add warnings row if there are failed/recovering prefixes
        if self.state.has_prefix_warnings() {
            rows.push(
                Row::new(vec!["Warnings:", warnings_str.as_str()])
                    .style(Style::default().fg(Color::Yellow)),
            );
        }

        // Pre-allocate formatted strings for MKN progress (lifetime extension)
        let mkn_phase_str: String;
        let mkn_progress_str: String;

        // Add MKN progress if active
        if self.state.mkn_in_progress {
            if let Some(ref mkn) = self.state.mkn_progress {
                mkn_phase_str = format!("Phase {}/{}", mkn.phase, mkn.total_phases);
                mkn_progress_str = format!("{:.1}%", mkn.percent_complete);
                rows.push(
                    Row::new(vec!["MKN Progress:", mkn_phase_str.as_str()])
                        .style(Style::default().fg(Color::Magenta)),
                );
                rows.push(
                    Row::new(vec!["  Completion:", mkn_progress_str.as_str()])
                        .style(Style::default().fg(Color::Magenta)),
                );
            }
        }

        // Pre-allocate formatted strings for merge progress (lifetime extension)
        let merge_shards_str: String;
        let merge_progress_str: String;

        // Add merge progress if active
        if self.state.merge_in_progress {
            if let Some(ref merge) = self.state.merge_progress {
                merge_shards_str =
                    format!("{}/{} shards", merge.shards_processed, merge.total_shards);
                merge_progress_str = format!("{:.1}%", merge.percent_complete);
                rows.push(
                    Row::new(vec!["Merge Progress:", merge_shards_str.as_str()])
                        .style(Style::default().fg(Color::Blue)),
                );
                rows.push(
                    Row::new(vec!["  Completion:", merge_progress_str.as_str()])
                        .style(Style::default().fg(Color::Blue)),
                );
            }
        }

        let table =
            Table::new(rows, [Constraint::Length(20), Constraint::Fill(1)]).column_spacing(1);

        frame.render_widget(table, chunks[0]);

        // Throughput sparkline
        if !self.state.throughput_history.is_empty() {
            let data = widgets::sparkline_data(&self.state.throughput_history);
            let sparkline = ratatui::widgets::Sparkline::default()
                .block(Block::default().title("Throughput"))
                .data(&data)
                .style(Style::default().fg(Color::Cyan));
            frame.render_widget(sparkline, chunks[1]);
        }
    }

    /// Render workers panel.
    fn render_workers(&self, frame: &mut Frame, area: Rect) {
        let workers_block = Block::default().title(" Workers ").borders(Borders::ALL);

        let inner = workers_block.inner(area);
        frame.render_widget(workers_block, area);

        let items: Vec<ListItem> = self
            .state
            .workers
            .iter()
            .map(|worker| {
                // Format order indicator if available
                let order_str = worker.order.map(|o| format!("{}g:", o)).unwrap_or_default();

                let (status, style) = match &worker.status {
                    WorkerStatus::Idle => {
                        ("idle".to_string(), Style::default().fg(Color::DarkGray))
                    }
                    WorkerStatus::Downloading => {
                        let prefix = worker.prefix.as_deref().unwrap_or("...");
                        let progress_bar = widgets::mini_progress_bar(worker.progress, 10);
                        let status_text = if worker.ngram_count > 0 {
                            // Show n-gram count once processing has started
                            format!(
                                "{}{} {} {} ngrams",
                                order_str,
                                progress_bar,
                                prefix,
                                format_count(worker.ngram_count)
                            )
                        } else {
                            // Still connecting/downloading before processing starts
                            format!(
                                "{}{} {} {}",
                                order_str,
                                progress_bar,
                                prefix,
                                format_bytes(worker.bytes_downloaded)
                            )
                        };
                        (status_text, Style::default().fg(Color::Yellow))
                    }
                    WorkerStatus::Completed => {
                        let prefix = worker.prefix.as_deref().unwrap_or("...");
                        (
                            format!(
                                "{}done: {} ({} ngrams)",
                                order_str,
                                prefix,
                                format_count(worker.ngram_count)
                            ),
                            Style::default().fg(Color::Green),
                        )
                    }
                    WorkerStatus::Retrying {
                        attempt,
                        max_attempts,
                    } => {
                        let prefix = worker.prefix.as_deref().unwrap_or("...");
                        (
                            format!(
                                "{}retry {}/{}: {}",
                                order_str, attempt, max_attempts, prefix
                            ),
                            Style::default().fg(Color::Red),
                        )
                    }
                };

                ListItem::new(format!("[{}] {}", worker.id, status)).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render log panel.
    fn render_log(&self, frame: &mut Frame, area: Rect) {
        let log_block = Block::default()
            .title(format!(" Log ({} entries) ", self.state.log_entries.len()))
            .borders(Borders::ALL);

        let inner = log_block.inner(area);
        frame.render_widget(log_block, area);

        let items: Vec<ListItem> = self
            .state
            .log_entries
            .iter()
            .skip(self.state.log_scroll)
            .take(inner.height as usize)
            .map(|entry| {
                let style = match entry.level {
                    // Use Gray instead of DarkGray for visibility on dark backgrounds
                    crate::sources::google_books::LogLevel::Debug => {
                        Style::default().fg(Color::Gray)
                    }
                    crate::sources::google_books::LogLevel::Info => {
                        Style::default().fg(Color::White)
                    }
                    crate::sources::google_books::LogLevel::Warn => {
                        Style::default().fg(Color::Yellow)
                    }
                    crate::sources::google_books::LogLevel::Error => {
                        Style::default().fg(Color::Red)
                    }
                };

                ListItem::new(format!("[{}] {}", entry.timestamp, entry.message)).style(style)
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    /// Render the completion dialog.
    fn render_completion_dialog(&self, frame: &mut Frame, area: Rect) {
        // Calculate dialog size (60% width, 14 lines height)
        let dialog_width = (area.width as f32 * 0.6).max(50.0).min(80.0) as u16;
        let dialog_height = 14u16;

        // Center the dialog
        let x = area.x + (area.width.saturating_sub(dialog_width)) / 2;
        let y = area.y + (area.height.saturating_sub(dialog_height)) / 2;
        let dialog_area = Rect::new(x, y, dialog_width, dialog_height);

        // Clear the area behind the dialog
        frame.render_widget(Clear, dialog_area);

        // Determine title and border color based on state
        let (title, border_style) = if self.state.error_message.is_some() {
            (" Import Failed ", Style::default().fg(Color::Red))
        } else if self.state.cancelled {
            (" Import Cancelled ", Style::default().fg(Color::Yellow))
        } else {
            (" Import Complete ", Style::default().fg(Color::Green))
        };

        // Build content lines
        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(""));

        if let Some(ref stats) = self.state.completion_stats {
            lines.push(Line::from(vec![
                Span::styled("  Total n-grams:  ", Style::default().fg(Color::Cyan)),
                Span::raw(format_count(stats.total_ngrams)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Files processed: ", Style::default().fg(Color::Cyan)),
                Span::raw(format!("{}", stats.files_processed)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Duration:        ", Style::default().fg(Color::Cyan)),
                Span::raw(format_duration(stats.total_duration)),
            ]));

            if stats.merge_performed {
                lines.push(Line::from(vec![
                    Span::styled("  Shards:          ", Style::default().fg(Color::Cyan)),
                    Span::raw(if stats.shards_kept {
                        "kept (--keep-shards)"
                    } else {
                        "deleted after merge"
                    }),
                ]));
            }
        } else {
            // Fallback if no stats available
            lines.push(Line::from(vec![
                Span::styled("  Total n-grams:  ", Style::default().fg(Color::Cyan)),
                Span::raw(format_count(self.state.total_ngrams)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Files processed: ", Style::default().fg(Color::Cyan)),
                // Use derived total to stay in sync with Order Progress display
                Span::raw(format!("{}", self.state.total_files_completed())),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  Duration:        ", Style::default().fg(Color::Cyan)),
                Span::raw(format_duration(self.state.elapsed)),
            ]));
        }

        // Add error message if present
        if let Some(ref error) = self.state.error_message {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("  Error: ", Style::default().fg(Color::Red)),
                Span::raw(error.clone()),
            ]));
        }

        // Add warnings if there were failures
        if self.state.failed_prefixes_count > 0 {
            lines.push(Line::from(""));
            lines.push(Line::from(vec![Span::styled(
                format!(
                    "  {} prefixes failed (will retry next run)",
                    self.state.failed_prefixes_count
                ),
                Style::default().fg(Color::Yellow),
            )]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![Span::styled(
            "          Press [Enter] or [Q] to exit",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )]));

        let dialog = Paragraph::new(lines)
            .block(
                Block::default()
                    .title(title)
                    .title_style(border_style.add_modifier(Modifier::BOLD))
                    .borders(Borders::ALL)
                    .border_style(border_style),
            )
            .alignment(Alignment::Left);

        frame.render_widget(dialog, dialog_area);
    }
}

/// Format a duration as HH:MM:SS.
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let secs = secs % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    } else {
        format!("{:02}:{:02}", mins, secs)
    }
}

/// Format a count with commas.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

/// Format bytes as human-readable.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00");
        assert_eq!(format_duration(Duration::from_secs(65)), "01:05");
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }
}
