//! Minimal Claude Code–style layout. Full width. No sidebar.

use crate::theme::Theme;
use crate::ui::commands;
use crate::ui::markdown;
use crate::ui::paths;
use crate::ui::state::{AppState, MessageKind, ToolStatus};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::path::Path;

pub fn draw(frame: &mut Frame, theme: &Theme, state: &AppState) {
    let area = frame.size();
    frame.render_widget(
        Block::default().style(ratatui::style::Style::default().bg(theme.bg)),
        area,
    );

    // header | body | status  — input sits in body bottom like Claude
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, theme, state, chunks[0]);
    draw_body(frame, theme, state, chunks[1]);
    draw_status(frame, theme, state, chunks[2]);
}

fn short_cwd(cwd: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Some(rest) = cwd.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    cwd.to_string()
}

fn draw_header(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    // Claude-like: mark · name version · model · path
    let lines = vec![
        Line::from(vec![
            Span::styled("  ▣  ", theme.style_accent()),
            Span::styled("0zcode ", theme.style_text()),
            Span::styled(format!("v{}", state.version), theme.style_muted()),
        ]),
        Line::from(Span::styled(
            format!("     {}", state.model_label),
            theme.style_dim(),
        )),
        Line::from(Span::styled(
            format!("     {}", short_cwd(&state.cwd)),
            theme.style_success(),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.style_border()),
        ),
        area,
    );
}

fn draw_body(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let has_chat = state.messages.iter().any(|m| {
        matches!(
            m.kind,
            MessageKind::User | MessageKind::Assistant | MessageKind::Tool { .. }
        )
    });

    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(2), Constraint::Length(3)])
        .split(area);

    if !has_chat {
        // Empty state — Claude style: quiet center + prompt
        let empty = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  tools-first · zero source",
                theme.style_muted(),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Try  weather in Mumbai   ·   time   ·   10 USD to INR   ·   /help",
                theme.style_dim(),
            )),
        ];
        frame.render_widget(Paragraph::new(empty), body[0]);
    } else {
        draw_transcript(frame, theme, state, body[0]);
    }

    draw_prompt(frame, theme, state, body[1]);
}

fn draw_transcript(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    for m in &state.messages {
        match &m.kind {
            MessageKind::System => continue,
            MessageKind::User => {
                lines.push(Line::from(""));
                lines.push(Line::from(vec![
                    Span::styled("❯ ", theme.style_accent()),
                    Span::styled(m.text.clone(), theme.style_text()),
                ]));
            }
            MessageKind::Assistant => {
                let md = markdown::render_markdown(theme, &m.text);
                let mut first = true;
                for line in md {
                    if first {
                        let mut spans = vec![Span::styled("● ", theme.style_text())];
                        spans.extend(line.spans.iter().cloned());
                        lines.push(Line::from(spans));
                        first = false;
                    } else {
                        let mut spans = vec![Span::raw("  ")];
                        spans.extend(line.spans.iter().cloned());
                        lines.push(Line::from(spans));
                    }
                }
            }
            MessageKind::Activity { summary } => {
                lines.push(Line::from(vec![
                    Span::styled("  → ", theme.style_muted()),
                    Span::styled(summary.clone(), theme.style_dim()),
                ]));
            }
            MessageKind::Thought { duration } => {
                let label = if duration.as_secs() >= 1 {
                    format!("{:.1}s", duration.as_secs_f32())
                } else {
                    format!("{}ms", duration.as_millis())
                };
                lines.push(Line::from(vec![
                    Span::styled("  + ", theme.style_warning()),
                    Span::styled(format!("Thought · {label}"), theme.style_warning()),
                ]));
            }
            MessageKind::Tool { name, status } => {
                let st = match status {
                    ToolStatus::Running => ("running", theme.style_warning()),
                    ToolStatus::Done => ("done", theme.style_success()),
                    ToolStatus::Failed => ("failed", theme.style_danger()),
                };
                lines.push(Line::from(vec![
                    Span::styled("  $ ", theme.style_success()),
                    Span::styled(name.clone(), theme.style_success()),
                    Span::raw("  "),
                    Span::styled(st.0, st.1),
                ]));
                for part in m.text.lines() {
                    let p = part.trim().trim_start_matches('│').trim();
                    if !p.is_empty() {
                        lines.push(Line::from(Span::styled(
                            format!("    {p}"),
                            theme.style_dim(),
                        )));
                    }
                }
            }
            MessageKind::Error => {
                for part in m.text.lines() {
                    lines.push(Line::from(vec![
                        Span::styled("● ", theme.style_danger()),
                        Span::styled(part.to_string(), theme.style_danger()),
                    ]));
                }
            }
        }
    }

    let visible = area.height as usize;
    let max_off = lines.len().saturating_sub(visible);
    let y = max_off.saturating_sub(state.scroll as usize) as u16;
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((y, 0)),
        area,
    );
}

fn draw_prompt(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let mut lines = Vec::new();

    if let Some((tool, _)) = &state.pending_permission {
        lines.push(Line::from(Span::styled(
            format!("  Allow network `{tool}`?  y once · a always · n deny"),
            theme.style_warning(),
        )));
    }

    // Prompt line — Claude style
    if state.input.is_empty() && state.pending_permission.is_none() {
        lines.push(Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(
                "Try \"weather in Mumbai\"",
                theme.style_muted(),
            ),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(state.input.clone(), theme.style_text()),
            Span::styled("█", theme.style_accent()),
        ]));
    }

    // Completions only when typing / or @
    if state.input.starts_with('/') {
        let m = commands::filter_commands(&state.input);
        let h = m.iter().take(6).map(|c| c.name).collect::<Vec<_>>().join("  ");
        lines.push(Line::from(Span::styled(format!("  {h}"), theme.style_muted())));
    } else if let Some(tok) = paths::at_token(&state.input) {
        let m = paths::fuzzy_paths(Path::new(&state.cwd), tok);
        let h = if m.is_empty() {
            "no matches".into()
        } else {
            m.join("  ")
        };
        lines.push(Line::from(Span::styled(format!("  {h}"), theme.style_success())));
    } else {
        lines.push(Line::from(Span::styled(
            "  ? /help",
            theme.style_muted(),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(theme.style_border()),
        ),
        area,
    );
}

fn draw_status(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let left = if state.working {
        format!(" ● {}", state.status_line)
    } else {
        " · ready".into()
    };
    let style = if state.working {
        theme.style_warning()
    } else {
        theme.style_muted()
    };
    // Claude puts hints on the right conceptually — we keep one quiet line
    let text = format!("{left}                              esc quit");
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(text, style))),
        area,
    );
}

pub fn buffer_text(width: u16, height: u16, theme: &Theme, state: &AppState) -> String {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    terminal.draw(|f| draw(f, theme, state)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..height {
        for x in 0..width {
            out.push_str(buf.get(x, y).symbol());
        }
        out.push('\n');
    }
    out
}
