//! Interactive session for the `0zcode` binary.
//!
//! * **TUI mode** (real terminal): raw-mode prompt with line editing, history,
//!   slash-command menu, bracketed paste, live spinner, Esc/Ctrl-C interrupt.
//! * **Plain mode** (pipe, `TERM=dumb`, `ZEROZ_PLAIN=1`, no `stty`): the same
//!   renderers, line-at-a-time input, no cursor control.

use crate::agent::{handle_with, llm_linked, AgentResponse, ChatCtx};
use crate::config::Config;
use crate::editor::{self, Editor};
use crate::history::History;
use crate::llm;
use crate::locale;
use crate::term::{self, emit, Input, Key, RawMode, Term};
use crate::tui::{split_frames, Footer, Frame, Screen, StreamWriter, Theme};
use std::env;
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Entry point used by `main`.
pub fn run(cfg: &Config) -> ExitCode {
    let force_plain = env::var("ZEROZ_PLAIN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !force_plain && term::is_interactive() {
        if let Some(raw) = RawMode::enter() {
            return run_tui(cfg, raw);
        }
    }
    run_plain(cfg)
}

/// Which tool a message most likely hits (progress label only — routing is in `agent`).
pub fn tool_hint(msg: &str) -> &'static str {
    let l = msg.to_lowercase();
    if l.contains("weather") || l.contains("mausam") {
        "weather"
    } else if l.contains("time") || l.contains("tarikh") {
        "time"
    } else if l.contains("wiki") || l.starts_with("who ") {
        "wikipedia"
    } else if l.contains("search") || l.contains("news") {
        "search"
    } else if l.contains("usd") || l.contains("inr") {
        "currency"
    } else if l.contains("github") {
        "github"
    } else if l.contains("btc") || l.contains("crypto") {
        "crypto"
    } else if l.contains("headline") || l == "news" {
        "headlines"
    } else if l.contains(" km") || l.contains(" to mi") || l.contains(" to f") {
        "units"
    } else if l.chars().any(|c| "+-*/".contains(c)) {
        "calculator"
    } else {
        "tools"
    }
}

// ───────────────────────── session state ─────────────────────────

enum Dispatch {
    /// Handled locally — print and prompt again.
    Print(String),
    /// Needs the agent.
    Agent,
}

enum Finish {
    Text(String),
    /// The reply was already printed live, token by token.
    Streamed,
    /// Nothing to show (e.g. repeated offline notice suppressed).
    Silent,
    Clear,
    Quit,
}

struct LastRun {
    query: String,
    text: String,
    secs: f32,
}

struct Session<'a> {
    cfg: &'a Config,
    theme: Theme,
    history: History,
    last_reply: String,
    last_tool: String,
    last_run: Option<LastRun>,
    started: Instant,
    /// AERIS-64 conversation id (server keeps the history); `/clear` rotates it.
    llm_session: String,
}

impl<'a> Session<'a> {
    fn new(cfg: &'a Config, color: bool) -> Self {
        let mut theme = Theme::from_config(cfg);
        theme.color = color;
        Self {
            cfg,
            theme,
            history: History::new(cfg.history_size),
            last_reply: String::new(),
            last_tool: String::new(),
            last_run: None,
            started: Instant::now(),
            llm_session: llm::new_session_id(),
        }
    }

    /// Blocking agent call for plain mode (no live tokens, same AERIS session).
    fn ask(&self, line: &str) -> AgentResponse {
        let cancel = AtomicBool::new(false);
        let mut sink = |_: &str| {};
        let mut ctx = ChatCtx {
            session: &self.llm_session,
            cancel: &cancel,
            on_token: &mut sink,
        };
        handle_with(line, self.cfg, Some(&mut ctx))
    }

    /// One line for `/status` and `/model`: link state, with a quick ping.
    fn llm_state(&self) -> String {
        let cfg = self.cfg;
        let Some(url) = cfg.aeris_url.as_deref() else {
            return "not configured — LLM auto-start uses http://127.0.0.1:8000".into();
        };
        if cfg.offline {
            return "off (offline mode)".into();
        }
        // Live probe (honest): Down / ServerOnly / Ready
        match crate::aeris_boot::probe_status(url) {
            crate::aeris_boot::AerisStatus::Ready => match llm::ping(cfg) {
                Ok(model) => format!("online · {model} · {url}"),
                Err(_) => format!("online · {url}"),
            },
            crate::aeris_boot::AerisStatus::ServerOnly => format!(
                "server up at {url} but model not loaded — set NEUROFIELD_CHECKPOINT (model.safetensors)"
            ),
            crate::aeris_boot::AerisStatus::Down => format!(
                "offline at {url} — start backend or check ZEROZ_REPO / checkpoint"
            ),
        }
    }

    fn welcome(&self, width: usize) -> String {
        self.theme
            .render_welcome(width, &self.cfg.model_label, self.cfg.offline)
    }

    fn expand(&self, width: usize) -> String {
        match &self.last_run {
            Some(r) => self.render_framed(&r.query, &r.text, r.secs, width, false),
            None => self.theme.render_notice("Nothing to expand yet."),
        }
    }

    /// Tool output arrives framed (`┌ name … └`). One block per frame; a frame
    /// titled `error` becomes an error block.
    fn render_framed(
        &self,
        query: &str,
        text: &str,
        secs: f32,
        width: usize,
        collapse: bool,
    ) -> String {
        let t = &self.theme;
        let sections = split_frames(text);
        let multi = sections.len() > 1;
        let mut blocks: Vec<String> = sections
            .iter()
            .map(|s| {
                if s.title == "error" {
                    t.render_error(&s.lines.join("\n"), width)
                } else {
                    t.render_tool(
                        &s.title,
                        query,
                        &s.lines,
                        if multi { None } else { Some(secs) },
                        width,
                        collapse,
                    )
                }
            })
            .collect();
        if multi {
            blocks.push(t.render_dim(&format!("{} tools · {secs:.1}s", sections.len())));
        }
        blocks.join("\n")
    }

    fn dispatch(&self, line: &str, width: usize) -> Dispatch {
        let t = &self.theme;
        let lower = line.to_lowercase();
        match lower.as_str() {
            "/help" | "?" | "help" => Dispatch::Print(t.render_help(width)),
            m if m == "/mcp" || m == "mcp" || m.starts_with("/mcp ") => {
                let text = crate::mcp_store::handle_command(line).unwrap_or_else(|| {
                    format!(
                        "{}\n\nBuilt-in schema:\n{}",
                        crate::mcp_store::list_text(),
                        crate::registry::tools_markdown()
                    )
                });
                Dispatch::Print(t.render_assistant(&text, width))
            }
            "/history" | "history" => {
                Dispatch::Print(t.render_assistant(&self.history.render(), width))
            }
            "/copy" | "copy" => {
                if self.last_reply.is_empty() {
                    return Dispatch::Print(t.render_error("Nothing to copy yet.", width));
                }
                let path = env::temp_dir().join("0zcode_last.txt");
                Dispatch::Print(match fs::write(&path, &self.last_reply) {
                    Ok(()) => t.render_ok(&format!("Saved last reply → {}", path.display())),
                    Err(e) => t.render_error(&format!("Could not write: {e}"), width),
                })
            }
            "/expand" | "expand" => Dispatch::Print(self.expand(width)),
            "/status" | "status" => Dispatch::Print(t.render_status(
                width,
                &self.cfg.model_label,
                if self.cfg.offline {
                    "offline"
                } else {
                    "agent (no code)"
                },
                self.cfg.offline,
                self.history.len(),
                &self.last_tool,
                self.started.elapsed().as_secs(),
                &Config::config_path_display(),
                &self.llm_state(),
            )),
            "/model" | "model" => Dispatch::Print(t.render_assistant(
                &format!(
                    "Active label: {}\nAERIS-64: {}\nSet ZEROZ_MODEL_LABEL or model_label in config.toml.",
                    self.cfg.model_label,
                    self.llm_state()
                ),
                width,
            )),
            _ => Dispatch::Agent,
        }
    }

    /// Bookkeeping for a reply that was already written to the screen live.
    fn finish_streamed(&mut self, r: AgentResponse) -> Finish {
        self.last_reply = r.text.clone();
        self.last_tool = r.tool_names.join("+");
        self.history.push("assistant", &r.text);
        Finish::Streamed
    }

    fn finish(&mut self, line: &str, r: AgentResponse, secs: f32, width: usize) -> Finish {
        if r.text == "__QUIT__" {
            return Finish::Quit;
        }
        if r.text == "__CLEAR__" {
            self.history.clear();
            self.last_reply.clear();
            self.last_tool.clear();
            self.last_run = None;
            self.llm_session = llm::new_session_id();
            return Finish::Clear;
        }
        if r.text.trim().is_empty() || r.tool_names == ["_silent"] {
            return Finish::Silent;
        }
        self.last_reply = r.text.clone();
        self.last_tool = r.tool_names.join("+");
        self.history.push("assistant", &r.text);

        let rendered = if r.refused {
            self.theme.render_error(&r.text, width)
        } else if r.text.trim_start().starts_with('┌') {
            let out = self.render_framed(line, &r.text, secs, width, true);
            if split_frames(&r.text).iter().any(|s| s.title != "error") {
                self.last_run = Some(LastRun {
                    query: line.to_string(),
                    text: r.text.clone(),
                    secs,
                });
            }
            out
        } else {
            self.theme.render_assistant(&r.text, width)
        };
        Finish::Text(rendered)
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

// ───────────────────────── plain mode ─────────────────────────

fn run_plain(cfg: &Config) -> ExitCode {
    let color = cfg.color && io::stdout().is_terminal();
    let mut session = Session::new(cfg, color);
    let width = term::env_width();
    println!("{}", session.welcome(width));

    let stdin = io::stdin();
    let tty = stdin.is_terminal();
    loop {
        if tty {
            print!("❯ ");
            let _ = io::stdout().flush();
        }
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break, // EOF
            Ok(_) => {}
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match session.dispatch(line, width) {
            Dispatch::Print(s) => {
                println!("{s}\n");
                continue;
            }
            Dispatch::Agent => {}
        }
        session.history.push("user", line);
        let start = Instant::now();
        let r = session.ask(line);
        match session.finish(line, r, start.elapsed().as_secs_f32(), width) {
            Finish::Text(s) => println!("{s}\n"),
            Finish::Streamed => println!(),
            Finish::Silent => {}
            Finish::Clear => println!("{}", session.welcome(width)),
            Finish::Quit => break,
        }
    }
    ExitCode::SUCCESS
}

// ───────────────────────── TUI mode ─────────────────────────

enum Msg {
    Token(String),
    Done(AgentResponse),
}

enum Run {
    /// Reply, seconds, and whether it was already printed live.
    Done(AgentResponse, f32, bool),
    Interrupted,
    Failed,
}

/// A successful model reply (as opposed to a tool answer or an error frame).
fn is_model_reply(r: &AgentResponse) -> bool {
    !r.refused && r.tool_names == ["aeris"]
}

struct Tui<'a> {
    session: Session<'a>,
    term: Term,
    input: Input,
    screen: Screen,
    editor: Editor,
    /// Set after a first Ctrl-C on an empty prompt.
    armed: Option<Instant>,
    /// Keys typed while a tool was running; replayed into the editor.
    stash: Vec<Key>,
    last_size: (usize, usize),
    /// Completed chat lines. Header is fixed; only this region scrolls.
    log: Vec<String>,
}

fn run_tui(cfg: &Config, raw: RawMode) -> ExitCode {
    let _raw = raw; // restored on drop, including on early return
    let mut ui = Tui {
        session: Session::new(cfg, cfg.color),
        term: Term::new(),
        input: Input::spawn(),
        screen: Screen::new(),
        editor: Editor::new(),
        armed: None,
        stash: Vec::new(),
        last_size: (0, 0),
        log: Vec::new(),
    };
    ui.editor.load_history(editor::load_history());
    ui.run();
    ExitCode::SUCCESS
}

impl<'a> Tui<'a> {
    fn run(&mut self) {
        emit("\x1b[?2004h"); // bracketed paste
        let w = self.term.size().0;
        emit(Theme::clear_seq());
        emit(&self.session.welcome(w));
        emit("\n");

        while let Some(line) = self.prompt() {
            let width = self.term.size().0;
            match self.session.dispatch(&line, width) {
                Dispatch::Print(s) => {
                    emit(&format!("{s}\n\n"));
                    continue;
                }
                Dispatch::Agent => {}
            }
            self.session.history.push("user", &line);
            match self.run_agent(&line) {
                Run::Done(r, secs, streamed) => {
                    let fin = if streamed && is_model_reply(&r) {
                        self.session.finish_streamed(r)
                    } else {
                        if streamed {
                            emit("\r\n"); // partial text, then the error block
                        }
                        self.session.finish(&line, r, secs, width)
                    };
                    match fin {
                        Finish::Text(s) => {
                            for line in s.lines() {
                                self.log.push(line.to_string());
                            }
                            self.log.push(String::new());
                            self.workspace_redraw();
                        }
                        Finish::Streamed => {
                            // stream already pushed tokens into log via on_token? 
                            // streamed path prints live — capture last_reply
                            if !self.session.last_reply.is_empty() {
                                for line in self.session.last_reply.lines() {
                                    self.log.push(line.to_string());
                                }
                                self.log.push(String::new());
                            }
                            self.workspace_redraw();
                        }
                        Finish::Silent => {
                            self.workspace_redraw();
                        }
                        Finish::Clear => {
                            self.log.clear();
                            self.workspace_redraw();
                        }
                        Finish::Quit => break,
                    }
                }
                Run::Interrupted => {
                    let n = self.session.theme.render_notice("Interrupted");
                    emit(&format!("{n}\n\n"));
                }
                Run::Failed => {
                    let e = self
                        .session
                        .theme
                        .render_error("The tool stopped unexpectedly. Try again.", width);
                    emit(&format!("{e}\n\n"));
                }
            }
        }
        let bye = self.session.theme.paint("38;5;245", "bye.");
        emit(&format!("  {bye}\n"));
    }

    fn footer(&self) -> Footer {
        let cfg = self.session.cfg;
        // Claude Code parity: left shortcuts hint · right model / status
        let right = if cfg.offline {
            format!("{} · offline", cfg.model_label)
        } else if let Some(url) = cfg.aeris_url.as_deref() {
            match crate::aeris_boot::probe_status(url) {
                crate::aeris_boot::AerisStatus::Ready => {
                    format!("{} · online", cfg.model_label)
                }
                crate::aeris_boot::AerisStatus::ServerOnly => {
                    format!("{} · no model", cfg.model_label)
                }
                crate::aeris_boot::AerisStatus::Down => {
                    format!("{} · offline", cfg.model_label)
                }
            }
        } else {
            cfg.model_label.clone()
        };
        if self.armed.is_some() {
            Footer {
                left: "Press Ctrl-C again to exit".into(),
                right,
                warn: true,
            }
        } else {
            Footer {
                left: "? for shortcuts".into(),
                right,
                warn: false,
            }
        }
    }

    fn frame(&self, width: usize, submitted: bool) -> Frame {
        let menu = self.editor.suggestions();
        self.session.theme.prompt_frame(
            width,
            self.editor.chars(),
            self.editor.cursor(),
            &menu,
            self.editor.selected_index(),
            &self.footer(),
            submitted,
        )
    }

    fn close_frame(&mut self) {
        let w = self.term.size().0;
        let f = self.frame(w, true);
        self.screen.finish(&f, w);
    }

    fn apply_typeahead(&mut self, k: Key) {
        match k {
            Key::Char(c) => self.editor.insert(c),
            Key::Paste(s) => self.editor.insert_str(&s),
            Key::Backspace => self.editor.backspace(),
            _ => {}
        }
    }

    /// Full-screen chrome: fixed header + recent log body (header never scrolls away).
    fn workspace_redraw(&mut self) {
        let (cols, rows) = self.term.size();
        let header = self.session.welcome(cols);
        let header_h = header.lines().count().max(1) + 1;
        let prompt_h = 5usize;
        let body_h = rows.saturating_sub(header_h + prompt_h).max(3);

        emit("\x1b[H\x1b[2J");
        emit(&header);
        if !header.ends_with('\n') {
            emit("\n");
        }
        emit("\n");

        let start = self.log.len().saturating_sub(body_h);
        for line in &self.log[start..] {
            emit(line);
            emit("\r\n");
        }
        self.screen.drawn = false;
        self.last_size = (cols, rows);
    }

    /// Read one line. `None` means the user quit (Ctrl-D, double Ctrl-C, EOF).
    fn prompt(&mut self) -> Option<String> {
        for k in std::mem::take(&mut self.stash) {
            self.apply_typeahead(k);
        }
        let mut dirty = true;
        loop {
            let size = self.term.size();
            if size != self.last_size {
                self.workspace_redraw();
                dirty = true;
            }
            if dirty {
                let f = self.frame(size.0, false);
                self.screen.draw(&f, size.0);
                self.last_size = size;
                dirty = false;
            }
            let Some(key) = self.input.next_key(Duration::from_millis(600)) else {
                if self.armed.is_some_and(|t| t.elapsed() > Duration::from_secs(2)) {
                    self.armed = None;
                    dirty = true;
                }
                continue;
            };
            dirty = true;
            if key != Key::Ctrl('c') {
                self.armed = None;
            }
            let ed = &mut self.editor;
            match key {
                Key::Char(c) => ed.insert(c),
                Key::Paste(s) => ed.insert_str(&s),
                Key::Backspace => ed.backspace(),
                Key::Delete => ed.delete(),
                Key::Left | Key::Ctrl('b') => ed.left(),
                Key::Right | Key::Ctrl('f') => ed.right(),
                Key::Home | Key::Ctrl('a') => ed.home(),
                Key::End | Key::Ctrl('e') => ed.end(),
                Key::WordLeft | Key::Alt('b') => ed.word_left(),
                Key::WordRight | Key::Alt('f') => ed.word_right(),
                Key::Ctrl('w') => ed.kill_word_back(),
                Key::Ctrl('u') => ed.kill_to_start(),
                Key::Ctrl('k') => ed.kill_to_end(),
                Key::Up | Key::Ctrl('p') => {
                    if ed.menu_visible() {
                        ed.menu_move(-1);
                    } else {
                        ed.history_prev();
                    }
                }
                Key::Down | Key::Ctrl('n') => {
                    if ed.menu_visible() {
                        ed.menu_move(1);
                    } else {
                        ed.history_next();
                    }
                }
                Key::Tab => {
                    ed.apply_selected();
                }
                Key::BackTab => {
                    ed.menu_move(-1);
                    ed.apply_selected();
                }
                Key::Esc => ed.dismiss_menu(),
                Key::Enter => {
                    if ed.menu_visible() {
                        ed.apply_selected();
                    }
                    let line = ed.text();
                    if line.trim().is_empty() {
                        ed.clear();
                        continue;
                    }
                    ed.push_history(&line);
                    editor::append_history(&line);
                    ed.clear();
                    // draw the submitted text (no menu/footer) into scrollback
                    let w = self.term.size().0;
                    let chars: Vec<char> = line.chars().collect();
                    let f = self.session.theme.prompt_frame(
                        w,
                        &chars,
                        chars.len(),
                        &[],
                        0,
                        &self.footer(),
                        true,
                    );
                    self.screen.finish(&f, w);
                    emit("\n"); // blank row between the echo band and the reply
                    return Some(line.trim().to_string());
                }
                Key::Ctrl('c') => {
                    if !ed.is_empty() {
                        ed.clear();
                        self.armed = None;
                    } else if self.armed.is_some() {
                        self.close_frame();
                        return None;
                    } else {
                        self.armed = Some(Instant::now());
                    }
                }
                Key::Ctrl('d') => {
                    if ed.is_empty() {
                        self.close_frame();
                        return None;
                    }
                    ed.delete();
                }
                Key::Eof => {
                    self.close_frame();
                    return None;
                }
                Key::Ctrl('l') => {
                    self.log.clear();
                    self.workspace_redraw();
                }
                Key::Ctrl('o') => {
                    let w = self.term.size().0;
                    self.screen.clear(w);
                    let out = self.session.expand(w);
                    emit(&format!("{out}\n\n"));
                }
                _ => {}
            }
        }
    }

    /// Run the agent on a worker thread. Until the first word of a model reply
    /// arrives a spinner animates; then the reply is written live under `●`.
    /// Esc / Ctrl-C interrupts (and tells the worker to drop the connection).
    fn run_agent(&mut self, line: &str) -> Run {
        let (tx, rx) = mpsc::channel();
        let cfg = self.session.cfg.clone();
        let msg = line.to_string();
        let sid = self.session.llm_session.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        thread::spawn(move || {
            let tok_tx = tx.clone();
            let mut sink = move |t: &str| {
                let _ = tok_tx.send(Msg::Token(t.to_string()));
            };
            let mut ctx = ChatCtx {
                session: &sid,
                cancel: &worker_cancel,
                on_token: &mut sink,
            };
            let r = handle_with(&msg, &cfg, Some(&mut ctx));
            let _ = tx.send(Msg::Done(r));
        });

        let lang = locale::detect(line);
        let mut hint = tool_hint(line);
        if hint == "tools" && llm_linked(self.session.cfg) {
            hint = "aeris";
        }
        let label = capitalize(
            locale::progress_label(lang, hint)
                .trim_start_matches('…')
                .trim(),
        ) + "…";
        let width = self.term.size().0;
        let mut writer = StreamWriter::new(width);
        let bullet = self.session.theme.paint("1", "●");
        let start = Instant::now();
        let mut tick = 0usize;
        let mut spinner_shown = false;
        let mut live = false; // reply text is on screen
        emit("\x1b[?25l");

        // Print the `● ` lead-in the first time there is text to show.
        let show = |text: &str, spinner_shown: &mut bool, live: &mut bool| {
            if text.is_empty() {
                return;
            }
            if !*live {
                if *spinner_shown {
                    emit("\r\x1b[2K");
                    *spinner_shown = false;
                }
                emit(&format!("{bullet} "));
                *live = true;
            }
            emit(text);
        };

        let outcome = loop {
            match rx.try_recv() {
                Ok(Msg::Token(t)) => {
                    let out = writer.push(&t);
                    show(&out, &mut spinner_shown, &mut live);
                    continue;
                }
                Ok(Msg::Done(r)) => {
                    let tail = writer.finish();
                    if is_model_reply(&r) {
                        show(&tail, &mut spinner_shown, &mut live);
                    }
                    break Run::Done(r, start.elapsed().as_secs_f32(), live);
                }
                Err(TryRecvError::Disconnected) => break Run::Failed,
                Err(TryRecvError::Empty) => {}
            }
            // instant tools (calc, meta) never flash a spinner
            if !live && start.elapsed() > Duration::from_millis(180) {
                let s = self
                    .session
                    .theme
                    .render_spinner(tick, &label, start.elapsed().as_secs());
                emit(&s);
                spinner_shown = true;
                tick += 1;
            }
            let wait = if start.elapsed() < Duration::from_millis(150) {
                Duration::from_millis(10)
            } else {
                Duration::from_millis(80)
            };
            match self.input.next_key(wait) {
                Some(Key::Esc) | Some(Key::Ctrl('c')) => {
                    cancel.store(true, Ordering::Relaxed);
                    if live {
                        emit(&writer.finish());
                        emit("\r\n");
                    }
                    break Run::Interrupted;
                }
                Some(k) => self.stash.push(k),
                None => {}
            }
        };
        if spinner_shown {
            emit("\r\x1b[2K");
        }
        emit("\x1b[?25h");
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::handle;

    fn cfg_offline() -> Config {
        Config {
            offline: true,
            color: false,
            ..Config::default()
        }
    }

    #[test]
    fn hints_route_labels() {
        assert_eq!(tool_hint("weather in Delhi"), "weather");
        assert_eq!(tool_hint("938+29"), "calculator");
        assert_eq!(tool_hint("10 usd to inr"), "currency");
        assert_eq!(tool_hint("hello"), "tools");
    }

    #[test]
    fn dispatch_handles_local_commands() {
        let cfg = cfg_offline();
        let s = Session::new(&cfg, false);
        for cmd in ["/help", "?", "/status", "/model", "/history", "/expand"] {
            assert!(matches!(s.dispatch(cmd, 80), Dispatch::Print(_)), "{cmd}");
        }
        assert!(matches!(s.dispatch("weather in Pune", 80), Dispatch::Agent));
        assert!(matches!(s.dispatch("/exit", 80), Dispatch::Agent));
    }

    #[test]
    fn finish_renders_plain_answer_and_framed_tool_for_expand() {
        let cfg = cfg_offline();
        let mut s = Session::new(&cfg, false);
        // plain answers (calc) render as a `●` block
        let r = handle("938+29", &cfg);
        let Finish::Text(out) = s.finish("938+29", r, 0.01, 80) else {
            panic!("expected text");
        };
        assert!(out.starts_with("● "), "{out}");
        assert!(out.contains("938+29 = 967"), "{out}");
        // framed tool output becomes `● tool(arg)` + `⎿` and is expandable
        let r = AgentResponse {
            text: "┌ calc ──\n│ 938+29 = 967\n└──".into(),
            refused: false,
            tool_names: vec!["calculator"],
        };
        let Finish::Text(out) = s.finish("938+29", r, 0.01, 80) else {
            panic!("expected text");
        };
        assert!(out.contains("● calc(938+29)"), "{out}");
        assert!(out.contains("⎿  938+29 = 967"), "{out}");
        assert!(s.expand(80).contains("967"));
    }

    #[test]
    fn finish_marks_policy_refusal_as_error_block() {
        let cfg = cfg_offline();
        let mut s = Session::new(&cfg, false);
        let r = handle("write python code", &cfg);
        assert!(r.refused);
        let Finish::Text(out) = s.finish("write python code", r, 0.0, 80) else {
            panic!("expected text");
        };
        assert!(out.starts_with('●'));
    }

    #[test]
    fn framed_error_becomes_error_block_without_borders() {
        let cfg = cfg_offline();
        let mut s = Session::new(&cfg, false);
        let r = AgentResponse {
            text: "┌ error ────\n│ network error — timed out\n│ (honest miss)\n└────".into(),
            refused: false,
            tool_names: vec![],
        };
        let Finish::Text(out) = s.finish("weather", r, 1.0, 80) else {
            panic!("expected text");
        };
        assert!(out.starts_with("● network error — timed out"), "{out}");
        assert!(!out.contains('┌') && !out.contains('│') && !out.contains('└'), "{out}");
        assert!(s.last_run.is_none(), "errors are not expandable");
    }

    #[test]
    fn multi_tool_reply_gets_one_block_per_tool() {
        let cfg = cfg_offline();
        let s = Session::new(&cfg, false);
        let text = "┌ weather ──\n│ Pune 30C\n└──\n\n┌ time ──\n│ 12:00\n└──";
        let out = s.render_framed("weather and time in Pune", text, 2.5, 80, true);
        assert!(out.contains("● weather(weather and time in Pune)"), "{out}");
        assert!(out.contains("● time("), "{out}");
        assert!(out.contains("2 tools · 2.5s"), "{out}");
        assert!(!out.contains('┌'));
    }

    #[test]
    fn quit_and_clear_markers() {
        let cfg = cfg_offline();
        let mut s = Session::new(&cfg, false);
        assert!(matches!(
            s.finish("/exit", handle("/exit", &cfg), 0.0, 80),
            Finish::Quit
        ));
        assert!(matches!(
            s.finish("/clear", handle("/clear", &cfg), 0.0, 80),
            Finish::Clear
        ));
    }

    #[test]
    fn capitalizes_label() {
        assert_eq!(capitalize("fetching weather"), "Fetching weather");
        assert_eq!(capitalize(""), "");
    }
}
