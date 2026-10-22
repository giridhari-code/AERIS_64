//! Lightweight markdown → ratatui Lines (no external md crate).
//! Supports: headings, **bold**, `code`, fenced ``` blocks, - lists.

use crate::theme::Theme;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

pub fn render_markdown(theme: &Theme, text: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut in_fence = false;
    let mut fence_buf: Vec<String> = Vec::new();

    for raw in text.lines() {
        let line = raw;
        if line.trim_start().starts_with("```") {
            if in_fence {
                for code_line in &fence_buf {
                    out.push(Line::from(Span::styled(
                        format!("  {code_line}"),
                        Style::default()
                            .fg(theme.text_dim)
                            .add_modifier(Modifier::DIM),
                    )));
                }
                fence_buf.clear();
                in_fence = false;
            } else {
                in_fence = true;
            }
            continue;
        }
        if in_fence {
            fence_buf.push(line.to_string());
            continue;
        }

        if let Some(rest) = line.strip_prefix("### ") {
            out.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default()
                    .fg(theme.text)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(rest) = line.strip_prefix("## ") {
            out.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default()
                    .fg(theme.text)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ") {
            out.push(Line::from(Span::styled(
                rest.to_string(),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }
        if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            let mut spans = vec![Span::styled(" · ".to_string(), theme.style_muted())];
            spans.extend(inline_spans(theme, rest));
            out.push(Line::from(spans));
            continue;
        }
        if line.trim().is_empty() {
            out.push(Line::from(""));
            continue;
        }
        out.push(Line::from(inline_spans(theme, line)));
    }
    if in_fence {
        for code_line in &fence_buf {
            out.push(Line::from(Span::styled(
                format!("  {code_line}"),
                Style::default().fg(theme.text_dim),
            )));
        }
    }
    out
}

fn inline_spans(theme: &Theme, s: &str) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut buf = String::new();
    while i < chars.len() {
        // `code`
        if chars[i] == '`' {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), theme.style_text()));
            }
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != '`' {
                i += 1;
            }
            let code: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                code,
                Style::default()
                    .fg(theme.success)
                    .add_modifier(Modifier::ITALIC),
            ));
            if i < chars.len() {
                i += 1;
            }
            continue;
        }
        // **bold**
        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), theme.style_text()));
            }
            i += 2;
            let start = i;
            while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '*') {
                i += 1;
            }
            let bold: String = chars[start..i].iter().collect();
            spans.push(Span::styled(
                bold,
                Style::default()
                    .fg(theme.text)
                    .add_modifier(Modifier::BOLD),
            ));
            if i + 1 < chars.len() {
                i += 2;
            }
            continue;
        }
        buf.push(chars[i]);
        i += 1;
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, theme.style_text()));
    }
    if spans.is_empty() {
        spans.push(Span::raw(""));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    #[test]
    fn bold_and_heading() {
        let t = Theme::blood_and_ash();
        let lines = render_markdown(&t, "# Title\nHello **world** and `code`");
        assert!(lines.len() >= 2);
    }
}
