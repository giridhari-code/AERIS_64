//! Auto-start local AERIS-64 when `0zcode` launches and the LLM is down.
//!
//! Default: ON (`ZEROZ_AERIS_AUTO=0` to disable).
//! Default URL: `http://127.0.0.1:8000` when unset.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::llm;

const DEFAULT_URL: &str = "http://127.0.0.1:8000";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AerisStatus {
    /// Nothing listening
    Down,
    /// HTTP up but model not loaded (readyz 503)
    ServerOnly,
    /// readyz 200 — LLM usable
    Ready,
}

fn auto_enabled() -> bool {
    match env::var("ZEROZ_AERIS_AUTO") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true,
    }
}

fn http_get(url: &str, path: &str) -> Option<String> {
    let ep = llm::endpoint(url).ok()?;
    let addr = (ep.host.as_str(), ep.port)
        .to_socket_addrs()
        .ok()?
        .next()?;
    let mut sock = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        ep.host, ep.port
    );
    sock.write_all(req.as_bytes()).ok()?;
    sock.set_read_timeout(Some(Duration::from_millis(800))).ok();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 512];
    loop {
        match sock.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.len() > 4096 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Live probe — used by footer, /status, /model, boot.
pub fn probe_status(url: &str) -> AerisStatus {
    let Some(body) = http_get(url, "/readyz") else {
        // connection failed — try healthz to distinguish
        return if http_get(url, "/healthz").is_some() {
            AerisStatus::ServerOnly
        } else {
            AerisStatus::Down
        };
    };
    if body.contains("200") && (body.contains("ready") || body.contains("\"status\"")) {
        // 503 responses also have a body; require not 503
        if body.starts_with("HTTP/1.1 200") || body.starts_with("HTTP/1.0 200") {
            return AerisStatus::Ready;
        }
    }
    if body.contains("503") || body.contains("model not loaded") {
        return AerisStatus::ServerOnly;
    }
    // healthz only
    if http_get(url, "/healthz")
        .map(|h| h.contains("200"))
        .unwrap_or(false)
    {
        return AerisStatus::ServerOnly;
    }
    AerisStatus::Down
}

pub fn status_label(url: &str) -> String {
    match probe_status(url) {
        AerisStatus::Ready => format!("online at {url}"),
        AerisStatus::ServerOnly => {
            format!("server up at {url} but model not loaded (set NEUROFIELD_CHECKPOINT)")
        }
        AerisStatus::Down => format!("offline — not reachable at {url}"),
    }
}

fn find_repo_root() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(r) = env::var("ZEROZ_REPO") {
        cands.push(PathBuf::from(r));
    }
    if let Ok(cwd) = env::current_dir() {
        cands.push(cwd);
    }
    if let Ok(h) = env::var("HOME") {
        cands.push(PathBuf::from(&h).join("AERIS_64"));
        cands.push(PathBuf::from(&h).join("AERIS_64"));
    }
    for start in cands {
        let mut p = start;
        for _ in 0..10 {
            if p.join("packages/sentinel").is_dir() {
                return Some(p);
            }
            if !p.pop() {
                break;
            }
        }
    }
    None
}

fn find_checkpoint(root: &Path) -> Option<PathBuf> {
    if let Ok(c) = env::var("NEUROFIELD_CHECKPOINT") {
        let p = PathBuf::from(c);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
        if p.is_file() && p.file_name().map(|n| n == "model.safetensors").unwrap_or(false) {
            return p.parent().map(|x| x.to_path_buf());
        }
    }
    for d in ["checkpoints/local_run", "checkpoints/examples/AERIS_main"] {
        let p = root.join(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    let ck = root.join("checkpoints");
    if let Ok(walk) = fs::read_dir(&ck) {
        for e in walk.flatten() {
            let p = e.path();
            if p.join("model.safetensors").is_file() {
                return Some(p);
            }
            // one level deeper
            if let Ok(w2) = fs::read_dir(&p) {
                for e2 in w2.flatten() {
                    let p2 = e2.path();
                    if p2.join("model.safetensors").is_file() {
                        return Some(p2);
                    }
                }
            }
        }
    }
    None
}

fn python_bin() -> String {
    env::var("PYTHON").unwrap_or_else(|_| "python3".into())
}

/// Ensure URL is set; try start server; return human status for stderr.
pub fn ensure_aeris(cfg: &mut Config) -> Option<String> {
    if cfg.offline {
        return None;
    }
    if !auto_enabled() {
        if cfg.aeris_url.is_none() {
            return None;
        }
        let url = cfg.aeris_url.as_deref()?;
        return Some(format!("AERIS auto-start off — {}", status_label(url)));
    }

    if cfg.aeris_url.is_none() {
        cfg.aeris_url = Some(DEFAULT_URL.into());
    }
    let url = cfg.aeris_url.as_deref()?.to_string();

    match probe_status(&url) {
        AerisStatus::Ready => return Some(format!("AERIS-64 ready at {url}")),
        AerisStatus::ServerOnly => {
            return Some(format!(
                "AERIS server on {url} but model not loaded — set NEUROFIELD_CHECKPOINT to a folder with model.safetensors"
            ));
        }
        AerisStatus::Down => {}
    }

    let Some(root) = find_repo_root() else {
        return Some(format!(
            "AERIS offline at {url} — set ZEROZ_REPO=~/AERIS_64 (repo with packages/sentinel) or run scripts/zeroz/run_aeris.sh"
        ));
    };

    let ckpt = find_checkpoint(&root);
    if ckpt.is_none() {
        return Some(format!(
            "AERIS offline — no model.safetensors under {}/checkpoints (train or set NEUROFIELD_CHECKPOINT). Tools still work.",
            root.display()
        ));
    }

    let py = python_bin();
    let port = llm::endpoint(&url).map(|e| e.port).unwrap_or(8000);
    let log_path = env::temp_dir().join("aeris-0zcode-server.log");
    let log_file = fs::File::create(&log_path).ok();

    let mut cmd = Command::new(&py);
    cmd.args([
        "-m",
        "uvicorn",
        "sentinel.api.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--log-level",
        "warning",
    ]);
    cmd.env(
        "PYTHONPATH",
        format!(
            "{}{}",
            root.join("packages").display(),
            env::var("PYTHONPATH")
                .map(|p| format!(":{p}"))
                .unwrap_or_default()
        ),
    );
    if let Some(ref c) = ckpt {
        cmd.env("NEUROFIELD_CHECKPOINT", c);
    }
    cmd.current_dir(&root);
    cmd.stdin(Stdio::null());
    match log_file {
        Some(f) => match f.try_clone() {
            Ok(c) => {
                cmd.stdout(Stdio::from(c));
                cmd.stderr(Stdio::from(f));
            }
            Err(_) => {
                cmd.stdout(Stdio::null());
                cmd.stderr(Stdio::null());
            }
        },
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }

    match cmd.spawn() {
        Ok(_) => {
            let deadline = Instant::now() + Duration::from_secs(90);
            while Instant::now() < deadline {
                match probe_status(&url) {
                    AerisStatus::Ready => {
                        return Some(format!("AERIS-64 auto-started (ready) at {url}"));
                    }
                    AerisStatus::ServerOnly => {
                        // keep waiting for model load
                    }
                    AerisStatus::Down => {}
                }
                thread::sleep(Duration::from_millis(400));
            }
            match probe_status(&url) {
                AerisStatus::ServerOnly => Some(format!(
                    "AERIS server started but model not ready — check {} and NEUROFIELD_CHECKPOINT",
                    log_path.display()
                )),
                AerisStatus::Ready => Some(format!("AERIS-64 ready at {url}")),
                AerisStatus::Down => Some(format!(
                    "AERIS auto-start failed — see {} (uvicorn/python deps?)",
                    log_path.display()
                )),
            }
        }
        Err(e) => Some(format!(
            "AERIS auto-start spawn failed: {e} — pip install uvicorn torch && scripts/zeroz/run_aeris.sh"
        )),
    }
}
