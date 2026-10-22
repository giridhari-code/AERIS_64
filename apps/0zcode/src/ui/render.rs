//! Ratatui widgets — colors only from Theme.

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
    let area = frame.size(); // ratatui 0.26; 0.28+ uses area()
    frame.render_widget(
        Block::default().style(ratatui::style::Style::default().bg(theme.bg)),
        area,
    );

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(5),
            Constraint::Length(if state.pending_permission.is_some() {
                5
            } else {
                3
            }),
            Constraint::Length(1),
        ])
        .split(area);

    draw_header(frame, theme, state, chunks[0]);
    draw_transcript(frame, theme, state, chunks[1]);
    draw_input(frame, theme, state, chunks[2]);
    draw_footer(frame, theme, state, chunks[3]);
}


fn draw_header(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let lines = vec![
        Line::from(vec![
            Span::styled(" ▣ ", theme.style_accent()),
            Span::styled(
                format!("0zcode  v{} ", state.version),
                theme.style_text(),
            ),
        ]),
        Line::from(Span::styled(
            format!("  {} ", state.cwd),
            theme.style_success(),
        )),
        Line::from(Span::styled(
            format!("  {} · {} ", state.model_label, state.mode_hint),
            theme.style_muted(),
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

fn draw_transcript(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    for m in &state.messages {
        match &m.kind {
            MessageKind::User => {
                lines.push(Line::from(vec![
                    Span::styled(" ❯ ", theme.style_accent()),
                    Span::styled(m.text.clone(), theme.style_text()),
                ]));
            }
            MessageKind::Assistant => {
                let md = markdown::render_markdown(theme, &m.text);
                let mut first = true;
                for line in md {
                    if first {
                        let mut spans = vec![Span::styled(" ● ", theme.style_text())];
                        spans.extend(line.spans.iter().cloned());
                        lines.push(Line::from(spans));
                        first = false;
                    } else {
                        lines.push(line);
                    }
                }
                lines.push(Line::from(""));
            }
            MessageKind::Activity { summary } => {
                lines.push(Line::from(vec![
                    Span::styled(" → ", theme.style_muted()),
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
                    Span::styled(" + ", theme.style_warning()),
                    Span::styled(format!("Thought · {label}"), theme.style_warning()),
                ]));
            }
            MessageKind::Tool { name, status } => {
                let st = match status {
                    ToolStatus::Running => Span::styled("running", theme.style_warning()),
                    ToolStatus::Done => Span::styled("done", theme.style_success()),
                    ToolStatus::Failed => Span::styled("failed", theme.style_danger()),
                };
                lines.push(Line::from(vec![
                    Span::styled(" $ ", theme.style_success()),
                    Span::styled(name.clone(), theme.style_success()),
                    Span::raw("  "),
                    st,
                ]));
                for part in m.text.lines() {
                    let p = part.trim().trim_start_matches('│').trim();
                    if p.is_empty() {
                        continue;
                    }
                    lines.push(Line::from(Span::styled(
                        format!("   {p}"),
                        theme.style_dim(),
                    )));
                }
                lines.push(Line::from(""));
            }
            MessageKind::System => {
                lines.push(Line::from(Span::styled(
                    format!("  {} ", m.text),
                    theme.style_muted(),
                )));
            }
            MessageKind::Error => {
                for part in m.text.lines() {
                    lines.push(Line::from(vec![
                        Span::styled(" ● ", theme.style_danger()),
                        Span::styled(part.to_string(), theme.style_danger()),
                    ]));
                }
                lines.push(Line::from(""));
            }
        }
    }

    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll, 0));
    frame.render_widget(para, area);
}

fn draw_input(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    if let Some((tool, _)) = &state.pending_permission {
        lines.push(Line::from(Span::styled(
            format!(" Network tool `{tool}` — allow once [y] / always [a] / deny [n] "),
            theme.style_warning(),
        )));
    }

    lines.push(Line::from(vec![
        Span::styled(" ❯ ", theme.style_accent()),
        Span::styled(state.input.clone(), theme.style_text()),
        Span::styled("█", theme.style_accent()),
    ]));

    if state.input.starts_with('/') {
        let matches = commands::filter_commands(&state.input);
        let hint = matches
            .iter()
            .take(4)
            .map(|c| format!("{} — {}", c.name, c.description))
            .collect::<Vec<_>>()
            .join("  ·  ");
        lines.push(Line::from(Span::styled(
            format!("  {hint} "),
            theme.style_muted(),
        )));
    } else if let Some(tok) = paths::at_token(&state.input) {
        let cwd = Path::new(&state.cwd);
        let matches = paths::fuzzy_paths(cwd, tok);
        let hint = if matches.is_empty() {
            "no path matches".into()
        } else {
            matches.join("  ·  ")
        };
        lines.push(Line::from(Span::styled(
            format!("  @{hint} "),
            theme.style_success(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("  {} ", state.model_label),
            theme.style_muted(),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(theme.style_border())
                .style(theme.style_panel()),
        ),
        area,
    );
}

fn draw_footer(frame: &mut Frame, theme: &Theme, state: &AppState, area: Rect) {
    let spin = if state.working { "●" } else { "·" };
    let style = if state.working {
        theme.style_warning()
    } else {
        theme.style_muted()
    };
    let text = format!(
        " {spin} {}  ·  esc quit  ·  tools-first  ·  coding disabled ",
        state.status_line
    );
    frame.render_widget(Paragraph::new(Line::from(Span::styled(text, style))), area);
}

pub fn buffer_text(width: u16, height: u16, theme: &Theme, state: &AppState) -> String {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    terminal
        .draw(|f| draw(f, theme, state))
        .expect("draw");
    let buf = terminal.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..height {
        for x in 0..width {
            let cell = buf.get(x, y);
            out.push_str(cell.symbol());
        }
        out.push('\n');
    }
    out
}
