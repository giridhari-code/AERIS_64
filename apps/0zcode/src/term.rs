//! Terminal plumbing — std only, no libc, no `unsafe`.
//!
//! * raw mode via `stty` (saved with `stty -g`, restored on drop and on panic)
//! * a background stdin reader so the UI can poll keys with a timeout
//! * a pure byte → [`Key`] decoder (CSI / SS3 / UTF-8 / bracketed paste)
//! * display-width helpers (combining marks, CJK) so Devanagari and wide
//!   characters keep the cursor aligned

use std::io::{self, IsTerminal, Read, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// True when both stdin and stdout are a real terminal that can do cursor control.
pub fn is_interactive() -> bool {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return false;
    }
    !matches!(std::env::var("TERM").as_deref(), Ok("dumb") | Ok(""))
}

fn stty(args: &[&str]) -> Option<String> {
    let out = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Write to stdout and flush (the TUI never relies on line buffering).
pub fn emit(s: &str) {
    let mut o = io::stdout().lock();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

/// Write one frame atomically. DEC private mode 2026 (synchronized output)
/// makes the terminal hold back painting until the frame is complete, which
/// removes flicker on every redraw; terminals without it ignore the codes.
pub fn emit_sync(s: &str) {
    let mut buf = String::with_capacity(s.len() + 16);
    buf.push_str("\x1b[?2026h");
    buf.push_str(s);
    buf.push_str("\x1b[?2026l");
    emit(&buf);
}

static ALT_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Switch to the alternate screen so full repaints never leak into (or
/// duplicate in) the user's scrollback. Undone by [`leave_alt_screen`], which
/// the raw-mode guard and the panic hook also call.
pub fn enter_alt_screen() {
    ALT_ACTIVE.store(true, Ordering::Relaxed);
    emit("\x1b[?1049h\x1b[H");
}

static MOUSE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Ask the terminal to report wheel / touch scrolling as events (SGR mouse
/// mode). On the alternate screen Termux and most terminals otherwise turn a
/// swipe into ↑/↓ key presses, which would walk the input history.
pub fn enable_mouse() {
    MOUSE_ACTIVE.store(true, Ordering::Relaxed);
    emit("\x1b[?1000h\x1b[?1006h");
}

pub fn disable_mouse() {
    if MOUSE_ACTIVE.swap(false, Ordering::Relaxed) {
        emit("\x1b[?1006l\x1b[?1000l");
    }
}

pub fn leave_alt_screen() {
    if ALT_ACTIVE.swap(false, Ordering::Relaxed) {
        emit("\x1b[?1049l");
    }
}

// ───────────────────────── raw mode ─────────────────────────

/// RAII guard: terminal is in non-canonical, no-echo mode while it lives.
pub struct RawMode {
    saved: String,
}

impl RawMode {
    /// `None` when `stty` is unavailable or stdin is not a terminal —
    /// callers fall back to plain line input.
    pub fn enter() -> Option<RawMode> {
        let saved = stty(&["-g"])?;
        if saved.is_empty() {
            return None;
        }
        stty(&[
            "-icanon", "-echo", "-isig", "-ixon", "-iexten", "min", "1", "time", "0",
        ])?;
        let hook_saved = saved.clone();
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore(&hook_saved);
            prev(info);
        }));
        Some(RawMode { saved })
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        restore(&self.saved);
    }
}

fn restore(saved: &str) {
    // sync output off, alt screen off, bracketed paste off, cursor visible,
    // then original termios
    emit("\x1b[?2026l");
    disable_mouse();
    leave_alt_screen();
    emit("\x1b[?2004l\x1b[?25h");
    let _ = stty(&[saved]);
}

// ───────────────────────── size ─────────────────────────

/// Terminal size. Querying spawns `stty`, which is slow on phones (Termux), so
/// the first call asks once and a background thread then refreshes the value;
/// the UI thread only ever reads it and never waits on a child process.
pub struct Term {
    shared: Option<Arc<Mutex<(usize, usize)>>>,
}

impl Default for Term {
    fn default() -> Self {
        Self::new()
    }
}

impl Term {
    pub fn new() -> Self {
        Self { shared: None }
    }

    /// `(columns, rows)`
    pub fn size(&mut self) -> (usize, usize) {
        if let Some(shared) = &self.shared {
            return *shared.lock().unwrap_or_else(|e| e.into_inner());
        }
        let first = query_size();
        let shared = Arc::new(Mutex::new(first));
        let bg = Arc::clone(&shared);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(400));
            let s = query_size();
            *bg.lock().unwrap_or_else(|e| e.into_inner()) = s;
        });
        self.shared = Some(shared);
        first
    }
}

fn query_size() -> (usize, usize) {
    if let Some(out) = stty(&["size"]) {
        let mut it = out.split_whitespace();
        if let (Some(r), Some(c)) = (it.next(), it.next()) {
            if let (Ok(r), Ok(c)) = (r.parse::<usize>(), c.parse::<usize>()) {
                if r > 0 && c > 0 {
                    return (c, r);
                }
            }
        }
    }
    (env_width(), 24)
}

/// Width for non-interactive output: `$COLUMNS` or 80.
pub fn env_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 20)
        .unwrap_or(80)
}

// ───────────────────────── keys ─────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Paste(String),
    Enter,
    Tab,
    BackTab,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    WordLeft,
    WordRight,
    Esc,
    /// Mouse wheel / touch scroll (only when mouse reporting is on).
    ScrollUp,
    ScrollDown,
    /// `Ctrl(a)` … `Ctrl(z)`
    Ctrl(char),
    Alt(char),
    /// stdin closed
    Eof,
    Unknown,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Decoded {
    Key(Key, usize),
    /// Consume `n` bytes without producing a key.
    Skip(usize),
    /// Need more bytes.
    Incomplete,
}

const PASTE_END: &[u8] = b"\x1b[201~";

/// Decode the first key from `b`. Pure; the unit tests drive this directly.
pub fn decode(b: &[u8]) -> Decoded {
    let Some(&first) = b.first() else {
        return Decoded::Incomplete;
    };
    match first {
        0x1b => decode_escape(b),
        b'\r' | b'\n' => Decoded::Key(Key::Enter, 1),
        b'\t' => Decoded::Key(Key::Tab, 1),
        0x7f | 0x08 => Decoded::Key(Key::Backspace, 1),
        0x01..=0x1a => Decoded::Key(Key::Ctrl((b'a' + first - 1) as char), 1),
        0x00 | 0x1c..=0x1f => Decoded::Skip(1),
        _ => decode_utf8(b),
    }
}

fn decode_utf8(b: &[u8]) -> Decoded {
    let need = match b[0] {
        0x20..=0x7e => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => return Decoded::Skip(1),
    };
    if b.len() < need {
        return Decoded::Incomplete;
    }
    match std::str::from_utf8(&b[..need]) {
        Ok(s) => match s.chars().next() {
            Some(c) => Decoded::Key(Key::Char(c), need),
            None => Decoded::Skip(need),
        },
        Err(_) => Decoded::Skip(1),
    }
}

fn decode_escape(b: &[u8]) -> Decoded {
    if b.len() == 1 {
        return Decoded::Incomplete;
    }
    match b[1] {
        b'[' => decode_csi(b),
        b'O' => {
            if b.len() < 3 {
                return Decoded::Incomplete;
            }
            let k = match b[2] {
                b'A' => Key::Up,
                b'B' => Key::Down,
                b'C' => Key::Right,
                b'D' => Key::Left,
                b'H' => Key::Home,
                b'F' => Key::End,
                _ => Key::Unknown,
            };
            Decoded::Key(k, 3)
        }
        0x1b => Decoded::Key(Key::Esc, 1),
        0x7f | 0x08 => Decoded::Key(Key::Ctrl('w'), 2),
        b'\r' | b'\n' => Decoded::Key(Key::Enter, 2),
        c if (0x21..=0x7e).contains(&c) => Decoded::Key(Key::Alt(c as char), 2),
        _ => Decoded::Skip(2),
    }
}

fn decode_csi(b: &[u8]) -> Decoded {
    let mut i = 2;
    while i < b.len() && !(0x40..=0x7e).contains(&b[i]) {
        i += 1;
        if i > 40 {
            return Decoded::Skip(2);
        }
    }
    if i >= b.len() {
        return Decoded::Incomplete;
    }
    let params = &b[2..i];
    let fin = b[i];
    let used = i + 1;
    let modified = params.contains(&b';') && params.last().is_some_and(|m| matches!(m, b'3' | b'5'));
    // legacy X10 mouse report: ESC [ M then three raw bytes (button, col, row)
    if fin == b'M' && params.is_empty() {
        if b.len() < used + 3 {
            return Decoded::Incomplete;
        }
        let k = match b[used].wrapping_sub(32) & !28 {
            64 => Key::ScrollUp,
            65 => Key::ScrollDown,
            _ => Key::Unknown,
        };
        return Decoded::Key(k, used + 3);
    }
    let key = match fin {
        // SGR mouse report: ESC [ < button ; col ; row (M = press, m = release)
        b'M' | b'm' if params.first() == Some(&b'<') => {
            let button = std::str::from_utf8(&params[1..])
                .ok()
                .and_then(|p| p.split(';').next())
                .and_then(|n| n.parse::<u32>().ok());
            match button {
                // strip the shift / alt / ctrl modifier bits
                Some(n) if fin == b'M' => match n & !28 {
                    64 => Key::ScrollUp,
                    65 => Key::ScrollDown,
                    _ => Key::Unknown,
                },
                _ => Key::Unknown,
            }
        }
        b'A' => Key::Up,
        b'B' => Key::Down,
        b'C' if modified => Key::WordRight,
        b'D' if modified => Key::WordLeft,
        b'C' => Key::Right,
        b'D' => Key::Left,
        b'H' => Key::Home,
        b'F' => Key::End,
        b'Z' => Key::BackTab,
        b'~' => match params {
            b"1" | b"7" => Key::Home,
            b"4" | b"8" => Key::End,
            b"3" => Key::Delete,
            b"5" => Key::PageUp,
            b"6" => Key::PageDown,
            b"200" => return decode_paste(b, used),
            _ => Key::Unknown,
        },
        _ => Key::Unknown,
    };
    Decoded::Key(key, used)
}

fn decode_paste(b: &[u8], start: usize) -> Decoded {
    let rest = &b[start..];
    match rest.windows(PASTE_END.len()).position(|w| w == PASTE_END) {
        Some(pos) => {
            let text = String::from_utf8_lossy(&rest[..pos]).into_owned();
            Decoded::Key(Key::Paste(text), start + pos + PASTE_END.len())
        }
        None => Decoded::Incomplete,
    }
}

// ───────────────────────── input reader ─────────────────────────

/// Background stdin reader + key decoder with timeouts.
pub struct Input {
    rx: Receiver<Vec<u8>>,
    pending: Vec<u8>,
}

impl Input {
    pub fn spawn() -> Input {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        thread::spawn(move || {
            let mut stdin = io::stdin().lock();
            let mut buf = [0u8; 1024];
            loop {
                match stdin.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        Input {
            rx,
            pending: Vec::new(),
        }
    }

    /// Next key, or `None` if nothing arrived within `timeout`.
    pub fn next_key(&mut self, timeout: Duration) -> Option<Key> {
        let deadline = Instant::now() + timeout;
        loop {
            if !self.pending.is_empty() {
                match decode(&self.pending) {
                    Decoded::Key(k, n) => {
                        self.pending.drain(..n);
                        return Some(k);
                    }
                    Decoded::Skip(n) => {
                        self.pending.drain(..n);
                        continue;
                    }
                    Decoded::Incomplete => {
                        let wait = if self.pending.starts_with(b"\x1b[200~") {
                            Duration::from_millis(1500)
                        } else {
                            // a lone ESC is the Escape key, not the start of a sequence
                            Duration::from_millis(35)
                        };
                        match self.rx.recv_timeout(wait) {
                            Ok(more) => {
                                self.pending.extend(more);
                                continue;
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                let lone_esc = self.pending == [0x1b];
                                self.pending.clear();
                                return Some(if lone_esc { Key::Esc } else { Key::Unknown });
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                self.pending.clear();
                                return Some(Key::Eof);
                            }
                        }
                    }
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            match self.rx.recv_timeout(deadline - now) {
                Ok(bytes) => self.pending.extend(bytes),
                Err(RecvTimeoutError::Timeout) => return None,
                Err(RecvTimeoutError::Disconnected) => return Some(Key::Eof),
            }
        }
    }
}

// ───────────────────────── display width ─────────────────────────

const ZERO_WIDTH: &[(u32, u32)] = &[
    (0x0300, 0x036F), // combining diacritics
    (0x0900, 0x0902), // Devanagari signs
    (0x093A, 0x093A),
    (0x093C, 0x093C),
    (0x0941, 0x0948), // Devanagari vowel signs (Mn)
    (0x094D, 0x094D), // virama
    (0x0951, 0x0957),
    (0x0962, 0x0963),
    (0x200B, 0x200F), // zero-width space / joiners / marks
    (0x2060, 0x2064),
    (0xFE00, 0xFE0F), // variation selectors
];

const WIDE: &[(u32, u32)] = &[
    (0x1100, 0x115F),
    (0x2E80, 0xA4CF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE30, 0xFE6F),
    (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6),
    (0x1F300, 0x1F64F),
    (0x1F900, 0x1F9FF),
    (0x20000, 0x3FFFD),
];

pub fn char_width(c: char) -> usize {
    let u = c as u32;
    if u < 0x20 || (0x7f..0xa0).contains(&u) {
        return 0;
    }
    if u < 0x300 {
        return 1;
    }
    if ZERO_WIDTH.iter().any(|&(a, b)| u >= a && u <= b) {
        return 0;
    }
    if WIDE.iter().any(|&(a, b)| u >= a && u <= b) {
        return 2;
    }
    1
}

pub fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Width of `s` ignoring ANSI SGR escape sequences.
pub fn visible_width(s: &str) -> usize {
    let mut w = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for n in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&n) {
                        break;
                    }
                }
            }
            continue;
        }
        w += char_width(c);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(b: &[u8]) -> (Key, usize) {
        match decode(b) {
            Decoded::Key(k, n) => (k, n),
            other => panic!("expected key, got {other:?}"),
        }
    }

    #[test]
    fn decodes_mouse_wheel_and_swallows_clicks() {
        assert_eq!(key(b"\x1b[<64;10;5M"), (Key::ScrollUp, 11));
        assert_eq!(key(b"\x1b[<65;10;5M"), (Key::ScrollDown, 11));
        assert_eq!(key(b"\x1b[<68;10;5M").0, Key::ScrollUp); // with shift
        assert_eq!(key(b"\x1b[<0;10;5M").0, Key::Unknown); // tap down
        assert_eq!(key(b"\x1b[<0;10;5m").0, Key::Unknown); // tap up
        // legacy encoding: button byte 64 + 32, then column and row
        assert_eq!(key(&[0x1b, b'[', b'M', 96, 40, 40]), (Key::ScrollUp, 6));
        assert_eq!(decode(&[0x1b, b'[', b'M', 96]), Decoded::Incomplete);
    }

    #[test]
    fn decodes_plain_and_ctrl() {
        assert_eq!(key(b"a"), (Key::Char('a'), 1));
        assert_eq!(key(b"\r"), (Key::Enter, 1));
        assert_eq!(key(b"\x7f"), (Key::Backspace, 1));
        assert_eq!(key(b"\x01"), (Key::Ctrl('a'), 1));
        assert_eq!(key(b"\x04"), (Key::Ctrl('d'), 1));
        assert_eq!(key(b"\t"), (Key::Tab, 1));
    }

    #[test]
    fn decodes_arrows_and_navigation() {
        assert_eq!(key(b"\x1b[A"), (Key::Up, 3));
        assert_eq!(key(b"\x1b[D"), (Key::Left, 3));
        assert_eq!(key(b"\x1bOH"), (Key::Home, 3));
        assert_eq!(key(b"\x1b[3~"), (Key::Delete, 4));
        assert_eq!(key(b"\x1b[1;5C"), (Key::WordRight, 6));
        assert_eq!(key(b"\x1b[1;3D"), (Key::WordLeft, 6));
        assert_eq!(key(b"\x1b[Z"), (Key::BackTab, 3));
        assert_eq!(key(b"\x1bb"), (Key::Alt('b'), 2));
    }

    #[test]
    fn decodes_utf8_devanagari() {
        let s = "क".as_bytes();
        assert_eq!(key(s), (Key::Char('क'), 3));
        assert_eq!(decode(&s[..2]), Decoded::Incomplete);
    }

    #[test]
    fn lone_escape_is_incomplete() {
        assert_eq!(decode(b"\x1b"), Decoded::Incomplete);
        assert_eq!(decode(b"\x1b["), Decoded::Incomplete);
    }

    #[test]
    fn decodes_bracketed_paste() {
        let (k, n) = key(b"\x1b[200~hello\nworld\x1b[201~x");
        assert_eq!(k, Key::Paste("hello\nworld".into()));
        assert_eq!(n, b"\x1b[200~hello\nworld\x1b[201~".len());
        assert_eq!(decode(b"\x1b[200~partial"), Decoded::Incomplete);
    }

    #[test]
    fn widths() {
        assert_eq!(char_width('a'), 1);
        assert_eq!(char_width('中'), 2);
        assert_eq!(char_width('\u{094D}'), 0); // virama
        // न(1) म(1) स(1) ्(0) त(1) े(0)
        assert_eq!(str_width("नमस्ते"), 4);
        assert_eq!(visible_width("\x1b[31mred\x1b[0m"), 3);
    }
}
