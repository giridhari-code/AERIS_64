//! Auto-start local AERIS-64 when `0zcode` launches and the LLM is down.
//!
//! Default: ON (`ZEROZ_AERIS_AUTO=0` to disable).
//! Default URL: `http://127.0.0.1:8000` when unset.

use std::env;
use std::fs;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::Config;
use crate::llm;

const DEFAULT_URL: &str = "http://127.0.0.1:8000";

fn auto_enabled() -> bool {
    match env::var("ZEROZ_AERIS_AUTO") {
        Ok(s) => !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")),
        Err(_) => true, // default ON
    }
}

fn probe_ready(url: &str) -> bool {
    let Ok(ep) = llm::endpoint(url) else {
        return false;
    };
    let Ok(mut addrs) = (ep.host.as_str(), ep.port).to_socket_addrs_compat() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    let Ok(mut sock) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    use std::io::Write;
    let req = format!(
        "GET /readyz HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        ep.host, ep.port
    );
    if sock.write_all(req.as_bytes()).is_err() {
        return false;
    }
    sock.set_read_timeout(Some(Duration::from_millis(600))).ok();
    let mut buf = [0u8; 256];
    use std::io::Read;
    match sock.read(&mut buf) {
        Ok(n) if n > 12 => {
            let s = String::from_utf8_lossy(&buf[..n]);
            s.contains("200") || s.contains("ok") || s.contains("ready")
        }
        _ => false,
    }
}

trait ToSocketAddrsCompat {
    fn to_socket_addrs_compat(&self) -> std::io::Result<std::vec::IntoIter<std::net::SocketAddr>>;
}
impl ToSocketAddrsCompat for (&str, u16) {
    fn to_socket_addrs_compat(&self) -> std::io::Result<std::vec::IntoIter<std::net::SocketAddr>> {
        use std::net::ToSocketAddrs;
        Ok(self.to_socket_addrs()?.collect::<Vec<_>>().into_iter())
    }
}

fn find_repo_root() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        cands.push(cwd);
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(p) = exe.parent() {
            cands.push(p.to_path_buf());
            // cargo install: ~/.cargo/bin — walk up won't find repo; use ZEROZ_REPO / relative
        }
    }
    if let Ok(r) = env::var("ZEROZ_REPO") {
        cands.insert(0, PathBuf::from(r));
    }
    if let Ok(h) = env::var("HOME") {
        cands.push(PathBuf::from(h).join("AERIS_64"));
    }
    for start in cands {
        let mut p = start;
        for _ in 0..8 {
            if p.join("packages/sentinel").is_dir() && p.join("apps/0zcode").is_dir() {
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
        if p.join("model.safetensors").is_file() || p.is_file() {
            return Some(p);
        }
    }
    for d in [
        "checkpoints/local_run",
        "checkpoints/examples/AERIS_main",
        "checkpoints/examples",
    ] {
        let p = root.join(d);
        if p.join("model.safetensors").is_file() {
            return Some(p);
        }
    }
    // any model.safetensors under checkpoints/
    let ck = root.join("checkpoints");
    if let Ok(walk) = fs::read_dir(&ck) {
        for e in walk.flatten() {
            let p = e.path();
            if p.join("model.safetensors").is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn python_bin() -> String {
    env::var("PYTHON").unwrap_or_else(|_| "python3".into())
}

/// Ensure AERIS URL is set and server is up (start if needed). Returns status line for UI.
pub fn ensure_aeris(cfg: &mut Config) -> Option<String> {
    if cfg.offline || !auto_enabled() {
        return None;
    }
    // Default local URL so plain `0zcode` links without extra env
    if cfg.aeris_url.is_none() {
        cfg.aeris_url = Some(DEFAULT_URL.into());
    }
    let url = cfg.aeris_url.as_deref()?.to_string();

    if probe_ready(&url) {
        return Some(format!("AERIS-64 already up at {url}"));
    }

    let Some(root) = find_repo_root() else {
        return Some(
            "AERIS-64 down — set ZEROZ_REPO=~/AERIS_64 or start: scripts/zeroz/run_aeris.sh".into(),
        );
    };

    let ckpt = find_checkpoint(&root);
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
        Some(f) => {
            match f.try_clone() {
                Ok(c) => {
                    cmd.stdout(Stdio::from(c));
                    cmd.stderr(Stdio::from(f));
                }
                Err(_) => {
                    cmd.stdout(Stdio::null());
                    cmd.stderr(Stdio::null());
                }
            }
        }
        None => {
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }

    match cmd.spawn() {
        Ok(_child) => {
            // Detached: wait for ready (model load can take a while)
            let deadline = Instant::now() + Duration::from_secs(90);
            while Instant::now() < deadline {
                if probe_ready(&url) {
                    return Some(format!("AERIS-64 auto-started at {url}"));
                }
                thread::sleep(Duration::from_millis(400));
            }
            Some(format!(
                "AERIS-64 start timed out — see {}{}",
                log_path.display(),
                if ckpt.is_none() {
                    " (no model.safetensors found — train or set NEUROFIELD_CHECKPOINT)"
                } else {
                    ""
                }
            ))
        }
        Err(e) => Some(format!(
            "AERIS-64 auto-start failed ({e}) — install uvicorn / run scripts/zeroz/run_aeris.sh"
        )),
    }
}
