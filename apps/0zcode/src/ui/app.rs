//! Async render loop; agent runs on blocking pool, updates via channel.

use crate::agent::handle;
use crate::config::Config;
use crate::policy::PRODUCT_VERSION;
use crate::theme::Theme;
use crate::ui::render::{self, draw};
use crate::session;
use crate::ui::net_gate;
use crate::ui::paths;
use crate::ui::state::{
    explored_label, AgentUpdate, AppState, Message, ToolStatus, UiEvent,
};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::env;
use std::io::{self, Stdout};
use std::panic;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

pub fn run_app(cfg: &Config) -> io::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    rt.block_on(run_async(cfg))
}

async fn run_async(cfg: &Config) -> io::Result<()> {
    install_panic_hook();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let theme = Theme::from_env();
    let cwd = env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "~".into());
    let mut state = AppState::new(cfg.model_label.clone(), cwd, PRODUCT_VERSION.to_string());

    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiEvent>();
    let (ag_tx, mut ag_rx) = mpsc::unbounded_channel::<AgentUpdate>();

    // Agent worker: receives Submit, runs handle on blocking pool
    let cfg_worker = cfg.clone();
    tokio::spawn(async move {
        while let Some(ev) = ui_rx.recv().await {
            match ev {
                UiEvent::Quit => break,
                UiEvent::Cancel => {
                    // UI already bumped gen; nothing to run
                }
                UiEvent::Submit { line, gen } => {
                    let cfg = cfg_worker.clone();
                    let ag_tx = ag_tx.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        let t0 = Instant::now();
                        let r = handle(&line, &cfg);
                        let elapsed = t0.elapsed();
                        if r.refused {
                            let _ = ag_tx.send(AgentUpdate::Refused {
                                text: r.text,
                                gen,
                            });
                        } else if !r.tool_names.is_empty() {
                            let name = r.tool_names.join("+");
                            let _ = ag_tx.send(AgentUpdate::ToolStarted {
                                name: name.clone(),
                                gen,
                            });
                            let _ = ag_tx.send(AgentUpdate::ToolFinished {
                                name,
                                body: r.text,
                                status: ToolStatus::Done,
                                elapsed,
                                gen,
                            });
                            let _ = ag_tx.send(AgentUpdate::Idle { gen });
                        } else {
                            let _ = ag_tx.send(AgentUpdate::Assistant {
                                text: r.text,
                                gen,
                            });
                        }
                    })
                    .await;
                }
            }
        }
    });

    let mut result = Ok(());
    loop {
        terminal.draw(|f| draw(f, &theme, &state))?;

        // Drain agent updates without blocking
        while let Ok(u) = ag_rx.try_recv() {
            if let AgentUpdate::ToolFinished { name, .. } = &u {
                state.push(crate::ui::state::Message::activity(explored_label(1)));
                let _ = name;
            }
            state.apply_update(u);
            let _ = session::save_all(&session::last_session_path(), &state.messages);
        }

        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            let _ = ui_tx.send(UiEvent::Quit);
                            break;
                        }
                        KeyCode::Esc => {
                            if state.pending_permission.is_some() {
                                state.pending_permission = None;
                                state.push(Message::system("Network tool denied."));
                                continue;
                            }
                            if state.working {
                                state.cancel_in_flight();
                                let _ = ui_tx.send(UiEvent::Cancel);
                                state.push(Message::system(
                                    "Cancelled — in-flight result will be ignored.",
                                ));
                                continue;
                            }
                            let _ = ui_tx.send(UiEvent::Quit);
                            break;
                        }
                        KeyCode::Char('y') | KeyCode::Char('Y') if state.pending_permission.is_some() => {
                            if let Some((_tool, line)) = state.pending_permission.take() {
                                state.push(Message::user(line.clone()));
                                state.request_gen = state.request_gen.wrapping_add(1);
                                let gen = state.request_gen;
                                state.working = true;
                                state.status_line = "working".into();
                                let _ = ui_tx.send(UiEvent::Submit { line, gen });
                            }
                            continue;
                        }
                        KeyCode::Char('a') | KeyCode::Char('A') if state.pending_permission.is_some() => {
                            state.network_allowed_always = true;
                            if let Some((_tool, line)) = state.pending_permission.take() {
                                state.push(Message::user(line.clone()));
                                state.request_gen = state.request_gen.wrapping_add(1);
                                let gen = state.request_gen;
                                state.working = true;
                                state.status_line = "working".into();
                                let _ = ui_tx.send(UiEvent::Submit { line, gen });
                            }
                            continue;
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') if state.pending_permission.is_some() => {
                            state.pending_permission = None;
                            state.push(Message::system("Network tool denied."));
                            continue;
                        }
                        KeyCode::Enter => {
                            let line = state.take_input();
                            if line.is_empty() {
                                continue;
                            }
                            if matches!(line.as_str(), "/quit" | "/exit" | "/q") {
                                let _ = ui_tx.send(UiEvent::Quit);
                                break;
                            }
                            if line == "/clear" {
                                state.clear_transcript();
                                state.push(Message::system("Transcript cleared."));
                                let _ = session::save_all(&session::last_session_path(), &state.messages);
                                continue;
                            }
                            if line == "/compact" {
                                let keep = 40;
                                if state.messages.len() > keep {
                                    let drop_n = state.messages.len() - keep;
                                    state.messages.drain(0..drop_n);
                                    state.push(Message::system(format!("Compacted; kept last {keep} lines.")));
                                } else {
                                    state.push(Message::system("Nothing to compact."));
                                }
                                let _ = session::save_all(&session::last_session_path(), &state.messages);
                                continue;
                            }
                            if line == "/resume" {
                                match session::load_all(&session::last_session_path()) {
                                    Ok(msgs) if !msgs.is_empty() => {
                                        state.messages = msgs;
                                        state.push(Message::system("Session resumed."));
                                    }
                                    Ok(_) => state.push(Message::system("No saved session.")),
                                    Err(e) => state.push(Message::error(format!("resume failed: {e}"))),
                                }
                                continue;
                            }
                            if line == "/config" {
                                state.push(Message::user(line.clone()));
                                state.push(Message::assistant(format!(
                                    "config: {}",
                                    Config::config_path_display()
                                )));
                                continue;
                            }
                            if line == "/help" || line == "?" {
                                state.push(Message::user(line));
                                state.push(Message::assistant(
                                    "Commands: /help /clear /tools /status /model /quit\n\
Try: weather in Mumbai · time · 10 USD to INR",
                                ));
                                continue;
                            }
                            if line == "/tools" {
                                state.push(Message::user(line));
                                state.push(Message::assistant(crate::tools::catalog()));
                                continue;
                            }
                            if line == "/status" {
                                state.push(Message::user(line));
                                state.push(Message::assistant(format!(
                                    "model {} · cwd {} · up {}s",
                                    state.model_label,
                                    state.cwd,
                                    state.started.elapsed().as_secs()
                                )));
                                continue;
                            }
                            if line.starts_with("/model") {
                                let rest = line["/model".len()..].trim();
                                if !rest.is_empty() {
                                    state.model_label = rest.to_string();
                                }
                                state.push(Message::user(line));
                                state.push(Message::assistant(format!(
                                    "Set model label to {}",
                                    state.model_label
                                )));
                                continue;
                            }
                            if let Some(tool) = net_gate::needs_network(&line) {
                                if !state.network_allowed_always {
                                    state.pending_permission = Some((tool.to_string(), line));
                                    continue;
                                }
                            }
                            state.push(Message::user(line.clone()));
                            state.request_gen = state.request_gen.wrapping_add(1);
                            let gen = state.request_gen;
                            state.working = true;
                            state.status_line = "working".into();
                            let _ = ui_tx.send(UiEvent::Submit { line, gen });
                        }
                        KeyCode::Backspace => state.backspace(),
                        KeyCode::Tab => {
                            if state.input.starts_with('/') {
                                let matches = crate::ui::commands::filter_commands(&state.input);
                                if let Some(c) = matches.first() {
                                    state.input = c.name.to_string();
                                    state.cursor = state.input.len();
                                }
                            } else if let Some(tok) = paths::at_token(&state.input) {
                                let cwd = std::path::Path::new(&state.cwd);
                                let matches = paths::fuzzy_paths(cwd, tok);
                                if let Some(m) = matches.first() {
                                    state.input = paths::apply_path_completion(&state.input, m);
                                    state.cursor = state.input.len();
                                }
                            }
                        }
                        KeyCode::Char(c) => state.insert_char(c),
                        KeyCode::Up => state.scroll = state.scroll.saturating_add(1),
                        KeyCode::Down => state.scroll = state.scroll.saturating_sub(1),
                        _ => {}
                    }
                }
                Event::Resize(_, _) => {
                    // next draw uses new size — no flicker beyond one frame
                }
                _ => {}
            }
        }
    }

    restore_terminal(&mut terminal)?;
    result
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn install_panic_hook() {
    let original = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let mut out = io::stdout();
        let _ = execute!(out, LeaveAlternateScreen);
        original(info);
    }));
}
