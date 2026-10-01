/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::Cell;

use chrono::Local;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use tokio::sync::oneshot;

use crate::overlay::Overlay;
use crate::theme::Labels;
use crate::theme::Theme;

/// An in-progress overlay-producing async job.
///
/// Exactly one variant is live at a time. `App` holds
/// `active_job: Option<ActiveJob>` which is always `Some` iff
/// `overlay` is `Some` (TUI-21). Enforced by `set_job`/`dismiss_job`.
pub(crate) enum ActiveJob {
    /// A single py-spy HTTP fetch.
    ///
    /// Use `rx.is_some()` — not `lines.is_empty()` — to distinguish
    /// "loading" from "completed with no output". An empty `lines`
    /// with `rx == None` means the fetch completed but returned nothing.
    PySpy {
        /// `Some` while the HTTP fetch is in flight; `None` after the
        /// oneshot fires.
        rx: Option<oneshot::Receiver<Vec<Line<'static>>>>,
        short: String,
        /// Populated when the oneshot result arrives.
        lines: Vec<Line<'static>>,
        /// Set to a formatted timestamp when the result arrives.
        completed_at: Option<String>,
    },
    /// A single config dump HTTP fetch.
    ///
    /// Same rx/lines semantics as PySpy: `rx.is_some()` → loading.
    Config {
        rx: Option<oneshot::Receiver<Vec<Line<'static>>>>,
        short: String,
        lines: Vec<Line<'static>>,
        completed_at: Option<String>,
    },
}

impl ActiveJob {
    /// Build the display overlay for this job's current state.
    ///
    /// Takes `(&self, &Theme)` not `(&self, &App)` because
    /// `rebuild_overlay` borrows `self.active_job` (shared) while
    /// holding `&mut self` for the overlay assignment. The borrow
    /// checker rejects `&App` here.
    pub(crate) fn build_overlay(&self, theme: &Theme) -> Overlay {
        match self {
            ActiveJob::PySpy {
                rx,
                short,
                lines,
                completed_at,
            } => {
                let scheme = &theme.scheme;
                let labels = &theme.labels;
                let sep = labels.separator;
                // Use rx.is_some() — not lines.is_empty() — to detect
                // loading state. An empty lines with rx == None means
                // "completed with no output".
                let loading = rx.is_some();

                let title = if loading {
                    Line::from(vec![
                        Span::styled(
                            format!("py-spy: {short}"),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{sep}{}", labels.overlay_running), scheme.info),
                    ])
                } else if let Some(ts) = completed_at {
                    Line::from(vec![
                        Span::styled(
                            format!("py-spy: {short}"),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{sep}{}", labels.overlay_completed_at),
                            scheme.detail_label,
                        ),
                        Span::raw(" "),
                        Span::styled(ts.clone(), scheme.stat_timing),
                    ])
                } else {
                    Line::from(Span::styled(
                        format!("py-spy: {short}"),
                        Style::default().add_modifier(Modifier::BOLD),
                    ))
                };

                let status_line = if loading {
                    Some(Line::from(vec![
                        Span::styled(labels.overlay_running, scheme.info),
                        Span::styled(format!("{sep}fetching stack trace"), scheme.detail_label),
                    ]))
                } else {
                    None
                };

                Overlay {
                    title,
                    status_line,
                    lines: lines.clone(),
                    loading,
                    scroll: Cell::new(0),
                    max_scroll: Cell::new(u16::MAX),
                }
            }
            ActiveJob::Config {
                rx,
                short,
                lines,
                completed_at,
            } => {
                let scheme = &theme.scheme;
                let labels = &theme.labels;
                let sep = labels.separator;
                let loading = rx.is_some();

                let title = if loading {
                    Line::from(vec![
                        Span::styled(
                            format!("config: {short}"),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{sep}{}", labels.overlay_running), scheme.info),
                    ])
                } else if let Some(ts) = completed_at {
                    Line::from(vec![
                        Span::styled(
                            format!("config: {short}"),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{sep}{}", labels.overlay_completed_at),
                            scheme.detail_label,
                        ),
                        Span::raw(" "),
                        Span::styled(ts.clone(), scheme.stat_timing),
                    ])
                } else {
                    Line::from(Span::styled(
                        format!("config: {short}"),
                        Style::default().add_modifier(Modifier::BOLD),
                    ))
                };

                let status_line = if loading {
                    Some(Line::from(vec![
                        Span::styled(labels.overlay_running, scheme.info),
                        Span::styled(
                            format!("{sep}fetching config snapshot"),
                            scheme.detail_label,
                        ),
                    ]))
                } else {
                    None
                };

                Overlay {
                    title,
                    status_line,
                    lines: lines.clone(),
                    loading,
                    scroll: Cell::new(0),
                    max_scroll: Cell::new(u16::MAX),
                }
            }
        }
    }

    /// Return the footer help text for the current overlay state.
    /// Falls back to the default topology help text when no job is active.
    pub(crate) fn footer_text<'a>(job: &Option<Self>, labels: &'a Labels) -> &'a str {
        match job {
            Some(ActiveJob::PySpy { .. }) => labels.footer_pyspy_help_text,
            Some(ActiveJob::Config { .. }) => labels.footer_config_help_text,
            None => labels.footer_help_text,
        }
    }

    /// Process an event from the async receiver, updating job state.
    /// Caller always rebuilds the overlay afterward.
    pub(crate) fn on_event(&mut self, event: ActiveJobEvent) {
        match event {
            ActiveJobEvent::PySpyResult(new_lines) => {
                if let ActiveJob::PySpy {
                    rx,
                    lines,
                    completed_at,
                    ..
                } = self
                {
                    *rx = None;
                    *lines = new_lines;
                    *completed_at = Some(Local::now().format("%H:%M:%S").to_string());
                } else {
                    debug_assert!(false, "PySpyResult delivered to non-PySpy job");
                }
            }
            ActiveJobEvent::ConfigResult(new_lines) => {
                if let ActiveJob::Config {
                    rx,
                    lines,
                    completed_at,
                    ..
                } = self
                {
                    *rx = None;
                    *lines = new_lines;
                    *completed_at = Some(Local::now().format("%H:%M:%S").to_string());
                } else {
                    debug_assert!(false, "ConfigResult delivered to non-Config job");
                }
            }
        }
    }
}

/// Result of the active overlay-producing job completing one event.
pub(crate) enum ActiveJobEvent {
    PySpyResult(Vec<Line<'static>>),
    ConfigResult(Vec<Line<'static>>),
}
