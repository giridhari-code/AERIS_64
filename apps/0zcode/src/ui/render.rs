//! Claude Code visual clone (layout only). 0zcode branding · zero source policy.

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

    let has_chat = state
        .messages
        .iter()
        .any(|m| matches!(m.kind, MessageKind::User | MessageKind::Assistant | MessageKind::Tool { .. }));

    if !has_chat {
        draw_empty_state(frame, theme, state, area);
    } else {
        draw_chat_state(frame, theme, state, area);
    }
}

fn short_cwd(cwd: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Some(rest) = cwd.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    cwd.to_string()
}

/// Empty session — matches Claude: header, separator, prompt near TOP, vast empty.
fn draw_empty_state(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // header + border
            Constraint::Length(1), // prompt
            Constraint::Length(1), // hints row
            Constraint::Min(1),    // empty void
        ])
        .split(area);

    draw_header(frame, theme, state, chunks[0]);
    draw_prompt_line(frame, theme, state, chunks[1]);
    draw_hint_row(frame, theme, state, chunks[2]);
    // chunks[3] stays empty background
}

/// Active chat — transcript fills, prompt pinned bottom (Claude session pattern).
fn draw_chat_state(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let input_h = if state.pending_permission.is_some() { 3 } else { 2 };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(3),
            Constraint::Length(input_h),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, theme, state, chunks[0]);
    draw_transcript(frame, theme, state, chunks[1]);
    draw_prompt_block(frame, theme, state, chunks[2]);
    draw_hint_row(frame, theme, state, chunks[3]);
}

fn draw_header(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    // Logo left, text like Claude Code screenshot
    let line1 = Line::from(vec![
        Span::styled("  ▣  ", theme.style_accent()),
        Span::styled("0zcode ", theme.style_text()),
        Span::styled(format!("v{}", state.version), theme.style_muted()),
    ]);
    let line2 = Line::from(vec![
        Span::raw("     "),
        Span::styled(&state.model_label, theme.style_dim()),
    ]);
    let line3 = Line::from(vec![
        Span::raw("     "),
        Span::styled(short_cwd(&state.cwd), theme.style_success()),
    ]);
    frame.render_widget(
        Paragraph::new(vec![line1, line2, line3]).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(theme.style_border()),
        ),
        area,
    );
}

fn draw_prompt_line(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let line = if state.input.is_empty() {
        Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(r#"Try "weather in Mumbai""#, theme.style_muted()),
        ])
    } else {
        Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(state.input.clone(), theme.style_text()),
            Span::styled("█", theme.style_accent()),
        ])
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_prompt_block(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let mut lines = Vec::new();
    if let Some((tool, _)) = &state.pending_permission {
        lines.push(Line::from(Span::styled(
            format!("  Allow `{tool}` network?  y · a always · n"),
            theme.style_warning(),
        )));
    }
    if state.input.is_empty() && state.pending_permission.is_none() {
        lines.push(Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(r#"Try "weather in Mumbai""#, theme.style_muted()),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("❯ ", theme.style_accent()),
            Span::styled(state.input.clone(), theme.style_text()),
            Span::styled("█", theme.style_accent()),
        ]));
    }
    if state.input.starts_with('/') {
        let m = commands::filter_commands(&state.input);
        let h = m.iter().take(6).map(|c| c.name).collect::<Vec<_>>().join("  ");
        lines.push(Line::from(Span::styled(format!("  {h}"), theme.style_muted())));
    } else if let Some(tok) = paths::at_token(&state.input) {
        let m = paths::fuzzy_paths(Path::new(&state.cwd), tok);
        let h = if m.is_empty() { "no matches".into() } else { m.join("  ") };
        lines.push(Line::from(Span::styled(format!("  {h}"), theme.style_success())));
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

fn draw_hint_row(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    // Claude: left "? for shortcuts" · right status
    let left = Span::styled("  ? for shortcuts", theme.style_muted());
    let right = if state.working {
        Span::styled(
            format!("{}  ", state.status_line),
            theme.style_warning(),
        )
    } else {
        Span::styled("coding disabled  ", theme.style_muted())
    };
    // manual left/right via full width padding
    let w = area.width as usize;
    let left_s = "  ? for shortcuts";
    let right_s = if state.working {
        state.status_line.clone()
    } else {
        "coding disabled".into()
    };
    let gap = w.saturating_sub(left_s.len() + right_s.len() + 2);
    let line = Line::from(vec![
        Span::styled(left_s, theme.style_muted()),
        Span::raw(" ".repeat(gap)),
        Span::styled(right_s, if state.working { theme.style_warning() } else { theme.style_muted() }),
    ]);
    let _ = (left, right);
    frame.render_widget(Paragraph::new(line), area);
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
                let (label, st) = match status {
                    ToolStatus::Running => ("running", theme.style_warning()),
                    ToolStatus::Done => ("done", theme.style_success()),
                    ToolStatus::Failed => ("failed", theme.style_danger()),
                };
                lines.push(Line::from(vec![
                    Span::styled("  $ ", theme.style_success()),
                    Span::styled(name.clone(), theme.style_success()),
                    Span::raw("  "),
                    Span::styled(label, st),
                ]));
                for part in m.text.lines() {
                    let p = part.trim().trim_start_matches('│').trim();
                    if !p.is_empty() {
                        lines.push(Line::from(Span::styled(format!("    {p}"), theme.style_dim())));
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
        Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((y, 0)),
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
