//! Evidence receipts — per tool-call audit trail.
//!
//! Most terminal agents show answers and forget the chain of evidence.
//! 0zcode records an append-only, hash-chained receipt for every tool
//! result so operators can prove *what ran, when, and on what input*.
//!
//! Storage: `~/.local/state/0zcode/receipts.jsonl` (override `ZEROZ_RECEIPTS_FILE`)
//! Commands: `/receipts` · `/receipts last` · `/receipts verify`
//!
//! Design:
//! - content hash over (ts, tool, input_fp, ok, body_fp)
//! - prev hash links entries (tamper-evident chain within a file)
//! - no secrets: input is fingerprinted (length + FNV-1a), not full raw dump
//! - IST wall-clock in display; UTC nanos in record

use crate::config::Config;
use crate::tools::ToolOutput;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn receipts_path() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_RECEIPTS_FILE") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/state/0zcode/receipts.jsonl")
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// FNV-1a 64-bit — stable, no deps.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fingerprint(s: &str) -> String {
    let h = fnv1a64(s.as_bytes());
    format!("fnv1a64:{h:016x}:len={}", s.len())
}

fn last_hash_in_file(path: &PathBuf) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return "genesis".into();
    };
    let mut prev = "genesis".to_string();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(h) = string_field_simple(line, "hash") {
            prev = h;
        }
    }
    prev
}

fn string_field_simple(json_line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let i = json_line.find(&pat)?;
    let rest = &json_line[i + pat.len()..];
    let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
    if !rest.starts_with('"') {
        return None;
    }
    let s = &rest[1..];
    let end = s.find('"')?;
    Some(s[..end].to_string())
}

/// Record one tool outcome. Never panics; best-effort disk write.
pub fn record(tool: &str, input_hint: &str, ok: bool, body: &str, cfg: &Config) {
    if env::var("ZEROZ_RECEIPTS")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("off"))
        .unwrap_or(false)
    {
        return;
    }
    let path = receipts_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let ts = now_unix_ms();
    let prev = last_hash_in_file(&path);
    let body_fp = fingerprint(body);
    let input_fp = fingerprint(input_hint);
    let material = format!("{ts}|{tool}|{input_fp}|{ok}|{body_fp}|{prev}");
    let hash = format!("{:016x}", fnv1a64(material.as_bytes()));
    let product = cfg
        .user_agent
        .chars()
        .take(40)
        .collect::<String>();
    // minimal JSON object (hand-built)
    let ok_s = if ok { "true" } else { "false" };
    let line = format!(
        "{{\"ts_ms\":\"{}\",\"tool\":{},\"ok\":\"{}\",\"input_fp\":{},\"body_fp\":{},\"prev\":{},\"hash\":{},\"ua\":{}}}\n",
        ts,
        json_str(tool),
        ok_s,
        json_str(&input_fp),
        json_str(&body_fp),
        json_str(&prev),
        json_str(&hash),
        json_str(&product),
    );
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = f.write_all(line.as_bytes());
    }
}

fn json_str(s: &str) -> String {
    crate::http::json_string(s)
}

pub fn record_output(tool: &str, input_hint: &str, out: &ToolOutput, cfg: &Config) {
    record(tool, input_hint, true, &out.body, cfg);
}

pub fn record_err(tool: &str, input_hint: &str, err: &str, cfg: &Config) {
    record(tool, input_hint, false, err, cfg);
}

/// `/receipts` family
pub fn handle_command(line: &str) -> Option<String> {
    let lower = line.trim().to_lowercase();
    if lower != "/receipts"
        && lower != "/receipts last"
        && lower != "/receipts verify"
        && lower != "/receipts path"
        && !lower.starts_with("/receipts ")
    {
        return None;
    }
    let path = receipts_path();
    if lower == "/receipts path" {
        return Some(format!("receipts file:\n  {}", path.display()));
    }
    if lower == "/receipts verify" {
        return Some(verify_chain(&path));
    }
    if lower == "/receipts last" || lower == "/receipts" {
        return Some(tail_text(&path, if lower == "/receipts last" { 5 } else { 12 }));
    }
    if let Some(rest) = lower.strip_prefix("/receipts ") {
        if rest == "last" {
            return Some(tail_text(&path, 5));
        }
        if rest == "verify" {
            return Some(verify_chain(&path));
        }
        if rest == "path" {
            return Some(format!("receipts file:\n  {}", path.display()));
        }
        if let Ok(n) = rest.parse::<usize>() {
            return Some(tail_text(&path, n.clamp(1, 50)));
        }
    }
    Some(
        "Evidence receipts\n\
  /receipts          last 12 entries\n\
  /receipts last     last 5\n\
  /receipts N        last N (max 50)\n\
  /receipts verify   hash-chain integrity\n\
  /receipts path     storage location\n\
Env: ZEROZ_RECEIPTS=0 to disable · ZEROZ_RECEIPTS_FILE to override path"
            .into(),
    )
}

fn tail_text(path: &PathBuf, n: usize) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return format!(
            "No receipts yet.\nWill write to:\n  {}\n\nEvery tool/MCP call appends a hash-chained evidence line.",
            path.display()
        );
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return "Receipts file empty.".into();
    }
    let start = lines.len().saturating_sub(n);
    let mut out = format!(
        "Evidence receipts ({} of {} · {})\n\n",
        lines.len() - start,
        lines.len(),
        path.display()
    );
    for (i, line) in lines[start..].iter().enumerate() {
        let ts = string_field_simple(line, "ts_ms")
            .and_then(|s| s.parse::<u64>().ok())
            .map(format_ist)
            .unwrap_or_else(|| "?".into());
        let tool = string_field_simple(line, "tool").unwrap_or_else(|| "?".into());
        let ok = string_field_simple(line, "ok").as_deref() == Some("true");
        let hash = string_field_simple(line, "hash").unwrap_or_default();
        let short = if hash.len() >= 8 { &hash[..8] } else { &hash };
        out.push_str(&format!(
            "{}. [{}] {} {}  #{}\n",
            start + i + 1,
            ts,
            if ok { "ok" } else { "FAIL" },
            tool,
            short
        ));
    }
    out.push_str("\n/receipts verify  ·  chain integrity");
    out
}

fn format_ist(ts_ms: u64) -> String {
    // IST = UTC+5:30
    let secs = (ts_ms / 1000) as i64 + 5 * 3600 + 30 * 60;
    let s = secs.rem_euclid(86400) as u32;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let sec = s % 60;
    format!("{h:02}:{m:02}:{sec:02} IST")
}

fn verify_chain(path: &PathBuf) -> String {
    let Ok(text) = fs::read_to_string(path) else {
        return format!("No receipts file at {}", path.display());
    };
    let mut prev = "genesis".to_string();
    let mut n = 0u32;
    let mut bad = 0u32;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        n += 1;
        let Some(stored_prev) = string_field_simple(line, "prev") else {
            bad += 1;
            continue;
        };
        let Some(stored_hash) = string_field_simple(line, "hash") else {
            bad += 1;
            continue;
        };
        if stored_prev != prev {
            bad += 1;
        }
        // recompute
        let ts = string_field_simple(line, "ts_ms").unwrap_or_default();
        let tool = string_field_simple(line, "tool").unwrap_or_default();
        let input_fp = string_field_simple(line, "input_fp").unwrap_or_default();
        let body_fp = string_field_simple(line, "body_fp").unwrap_or_default();
        let ok = string_field_simple(line, "ok").unwrap_or_else(|| "false".into());
        let material = format!("{ts}|{tool}|{input_fp}|{ok}|{body_fp}|{stored_prev}");
        let expect = format!("{:016x}", fnv1a64(material.as_bytes()));
        if expect != stored_hash {
            bad += 1;
        }
        prev = stored_hash;
    }
    if n == 0 {
        return "No receipt entries to verify.".into();
    }
    if bad == 0 {
        format!("Receipt chain OK — {n} entries, unbroken hash link (genesis → tip).")
    } else {
        format!("Receipt chain PROBLEMS — {bad} issue(s) in {n} entries. File may be truncated or edited.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn chain_records_and_verifies() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join("0zcode_receipts_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.jsonl");
        env::set_var("ZEROZ_RECEIPTS_FILE", &path);
        env::remove_var("ZEROZ_RECEIPTS");
        let cfg = Config::from_env();
        record("calculator", "1+1", true, "2", &cfg);
        record("weather", "Delhi", true, "clear", &cfg);
        let v = verify_chain(&path);
        assert!(v.contains("OK"), "{v}");
        let tail = tail_text(&path, 10);
        assert!(tail.contains("calculator"));
        env::remove_var("ZEROZ_RECEIPTS_FILE");
        let _ = fs::remove_dir_all(&dir);
    }
}
