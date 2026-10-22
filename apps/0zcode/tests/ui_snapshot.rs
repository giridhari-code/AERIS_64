//! Renders a whole session transcript through the public `tui` API and checks
//! the Claude Code–style layout at phone, laptop and wide sizes.

use zeroz::term::visible_width;
use zeroz::tui::{Footer, Theme};

fn theme() -> Theme {
    Theme {
        color: false,
        cwd: "~/Downloads".into(),
    }
}

fn footer() -> Footer {
    Footer {
        left: "? for shortcuts".into(),
        right: "AERIS-64 · tools-first".into(),
        warn: false,
    }
}

/// welcome → echo band → reply → live prompt, exactly as the REPL emits them.
fn transcript(width: usize) -> String {
    let t = theme();
    let mut out = t.render_welcome(width, "AERIS-64 · tools-first", false);
    out.push('\n');
    let echo: Vec<char> = "weather in Mumbai".chars().collect();
    let band = t.prompt_frame(width, &echo, echo.len(), &[], 0, &footer(), true);
    out.push_str(&band.lines.join("\n"));
    out.push_str("\n\n");
    out.push_str(&t.render_tool(
        "weather",
        "weather in Mumbai",
        &["Mumbai: 28.0°C · humidity 78%".into(), "source: Open-Meteo".into()],
        Some(0.8),
        width,
        true,
    ));
    out.push_str("\n\n");
    let live = t.prompt_frame(width, &[], 0, &[], 0, &footer(), false);
    out.push_str(&live.lines.join("\n"));
    out
}

#[test]
fn snapshot_80x24() {
    let text = transcript(80);
    eprintln!("--- 80 cols ---\n{text}");
    assert!(text.contains("0zcode"));
    assert!(text.contains("❯ weather in Mumbai"));
    assert!(text.contains("● weather(weather in Mumbai)  0.8s"));
    assert!(text.contains("⎿  Mumbai: 28.0°C"));
    assert!(text.contains("? for shortcuts"));
    // two rules around the prompt
    assert_eq!(text.lines().filter(|l| *l == "─".repeat(80)).count(), 2);
}

#[test]
fn never_wider_than_terminal() {
    for width in [30usize, 40, 80, 120, 200] {
        for l in transcript(width).lines() {
            assert!(visible_width(l) <= width, "width {width}: {l:?}");
        }
    }
}
