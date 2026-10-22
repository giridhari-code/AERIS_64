use zeroz::theme::Theme;
use zeroz::ui::render::buffer_text;
use zeroz::ui::state::{AppState, Message, ToolStatus};

fn sample_state() -> AppState {
    let mut s = AppState::new(
        "local / tools-first".into(),
        "~/AERIS_64".into(),
        "1.1.0".into(),
    );
    s.push(Message::user("weather in Mumbai"));
    s.push(Message::activity("Explored: 1 tool"));
    s.push(Message::tool(
        "weather",
        "Mumbai: 28.0°C · humidity 78%\nsource: Open-Meteo",
        ToolStatus::Done,
    ));
    s
}

#[test]
fn snapshot_80x24() {
    let theme = Theme::blood_and_ash();
    let text = buffer_text(80, 24, &theme, &sample_state());
    assert!(text.contains("0zcode"), "{text}");
    assert!(text.contains("weather") || text.contains("Mumbai"), "{text}");
    // dump for human review in test output
    eprintln!("--- 80x24 ---\n{text}");
}

#[test]
fn snapshot_160x50() {
    let theme = Theme::blood_and_ash();
    let text = buffer_text(160, 50, &theme, &sample_state());
    assert!(text.contains("0zcode"), "{text}");
    eprintln!("--- 160x50 (first 30 lines) ---");
    for (i, line) in text.lines().take(30).enumerate() {
        eprintln!("{i:02}|{line}");
    }
}
