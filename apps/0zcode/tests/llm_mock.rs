//! The AERIS-64 client against a mock that speaks the real server's protocol
//! (`packages/sentinel/serving/server.py`): JSON chat, chunked SSE, errors.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use zeroz::agent::{handle_with, ChatCtx};
use zeroz::config::Config;
use zeroz::llm;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Chunked,
    Raw,
    Endless,
    Unauthorized,
    NoModel,
    NotReady,
}

struct Mock {
    port: u16,
    seen: Arc<Mutex<Vec<String>>>,
}

fn read_request(s: &mut TcpStream) -> (String, String) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let n = s.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..p]).to_string();
            let len = head
                .lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                .unwrap_or(0);
            while buf.len() < p + 4 + len {
                let n = s.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = String::from_utf8_lossy(&buf[p + 4..]).to_string();
            return (head, body);
        }
    }
    (String::new(), String::new())
}

fn chunk(s: &mut TcpStream, data: &str) -> bool {
    s.write_all(format!("{:x}\r\n{data}\r\n", data.len()).as_bytes()).is_ok() && s.flush().is_ok()
}

fn serve(mode: Mode) -> Mock {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = Arc::clone(&seen);
    thread::spawn(move || {
        for conn in l.incoming() {
            let Ok(mut s) = conn else { continue };
            let seen = Arc::clone(&seen2);
            thread::spawn(move || {
                let (head, body) = read_request(&mut s);
                seen.lock().unwrap().push(format!("{head}\n\n{body}"));
                let req = head.lines().next().unwrap_or("").to_string();
                let json = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nConnection: close\r\n\r\n";
                match mode {
                    Mode::Unauthorized => {
                        let _ = s.write_all(b"HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n{\"detail\": \"invalid api key\"}");
                    }
                    Mode::NoModel => {
                        let _ = s.write_all(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n{\"detail\": \"model not loaded\"}");
                    }
                    Mode::NotReady if req.contains("/v1/models") => {
                        let _ = s.write_all(format!("{json}{{\"object\": \"list\", \"data\": [{{\"id\": \"AERIS_main\", \"ready\": false}}]}}").as_bytes());
                    }
                    _ if req.starts_with("GET /v1/models") => {
                        let _ = s.write_all(format!("{json}{{\"object\": \"list\", \"data\": [{{\"id\": \"AERIS_main\", \"object\": \"model\", \"owned_by\": \"sentinel\", \"ready\": true}}]}}").as_bytes());
                    }
                    _ if req.starts_with("POST /v1/chat/stream") => {
                        let events = [
                            r#"{"session_id": "x", "event": "start"}"#.to_string(),
                            r#"{"token": "Namas", "id": 1}"#.to_string(),
                            r#"{"token": "te ", "id": 2}"#.to_string(),
                            r#"{"token": "\u0926\u094b\u0938\u094d\u0924", "id": 3}"#.to_string(),
                            r#"{"event": "done", "latency_ms": 4.0, "text": "Namaste \u0926\u094b\u0938\u094d\u0924"}"#.to_string(),
                            "[DONE]".to_string(),
                        ];
                        if mode == Mode::Chunked || mode == Mode::Endless {
                            let _ = s.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nConnection: close\r\n\r\n");
                        } else {
                            let _ = s.write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nConnection: close\r\n\r\n");
                        }
                        if mode == Mode::Endless {
                            for i in 0..400 {
                                if !chunk(&mut s, &format!("data: {{\"token\": \"w{i} \", \"id\": {i}}}\n\n")) {
                                    return;
                                }
                                thread::sleep(Duration::from_millis(20));
                            }
                            return;
                        }
                        for e in events {
                            let data = format!("data: {e}\n\n");
                            if mode == Mode::Chunked {
                                // split an event across two chunks to test reassembly
                                let (a, b) = data.split_at(data.len() / 2);
                                chunk(&mut s, a);
                                thread::sleep(Duration::from_millis(15));
                                chunk(&mut s, b);
                            } else {
                                let _ = s.write_all(data.as_bytes());
                                let _ = s.flush();
                                thread::sleep(Duration::from_millis(15));
                            }
                        }
                        if mode == Mode::Chunked {
                            let _ = s.write_all(b"0\r\n\r\n");
                        }
                    }
                    _ if req.starts_with("POST /v1/chat") => {
                        let _ = s.write_all(format!("{json}{{\"text\": \"Namaste \\u0926\\u094b\\u0938\\u094d\\u0924\", \"session_id\": \"x\", \"latency_ms\": 3.0, \"model\": \"AERIS_main\"}}").as_bytes());
                    }
                    _ => {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n{\"detail\": \"Not Found\"}");
                    }
                }
            });
        }
    });
    Mock { port, seen }
}

fn cfg(port: u16) -> Config {
    Config {
        aeris_url: Some(format!("http://127.0.0.1:{port}/chat")), // legacy path is tolerated
        aeris_key: Some("secret-key".into()),
        aeris_timeout: Duration::from_secs(3),
        offline: false,
        color: false,
        ..Config::default()
    }
}

#[test]
fn ping_reports_model_and_readiness() {
    let m = serve(Mode::Chunked);
    assert_eq!(llm::ping(&cfg(m.port)).unwrap(), "AERIS_main");
    let m = serve(Mode::NotReady);
    assert!(llm::ping(&cfg(m.port)).unwrap_err().to_string().contains("not loaded"));
}

#[test]
fn chat_sends_session_key_and_decodes_unicode() {
    let m = serve(Mode::Raw);
    let never = AtomicBool::new(false);
    let out = llm::chat(&cfg(m.port), "0zsession1234", "namaste \"AERIS\"", &never).unwrap();
    assert_eq!(out, "Namaste दोस्त");
    let req = m.seen.lock().unwrap()[0].clone();
    assert!(req.starts_with("POST /v1/chat HTTP/1.1"), "{req}");
    assert!(req.to_ascii_lowercase().contains("x-api-key: secret-key"));
    assert!(req.contains(r#""session_id":"0zsession1234""#));
    assert!(req.contains(r#""prompt":"namaste \"AERIS\"""#));
    assert!(req.contains(r#""max_new_tokens":256"#));
}

fn collect_stream(mode: Mode) -> (Vec<String>, String) {
    let m = serve(mode);
    let never = AtomicBool::new(false);
    let mut toks = Vec::new();
    let mut sink = |t: &str| toks.push(t.to_string());
    let fin = llm::stream(&cfg(m.port), "0zsession1234", "hi", &never, &mut sink).unwrap();
    assert!(m.seen.lock().unwrap()[0].starts_with("POST /v1/chat/stream"));
    (toks, fin)
}

#[test]
fn stream_chunked_sse_in_order_with_split_events() {
    let (toks, fin) = collect_stream(Mode::Chunked);
    assert_eq!(toks, ["Namas", "te ", "दोस्त"]);
    assert_eq!(fin, "Namaste दोस्त");
}

#[test]
fn stream_unchunked_close_delimited_sse() {
    let (toks, fin) = collect_stream(Mode::Raw);
    assert_eq!(toks.concat(), "Namaste दोस्त");
    assert_eq!(fin, "Namaste दोस्त");
}

#[test]
fn stream_can_be_interrupted() {
    let m = serve(Mode::Endless);
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = Arc::clone(&cancel);
    let mut n = 0;
    let mut sink = |_: &str| {
        n += 1;
        if n == 3 {
            c2.store(true, Ordering::Relaxed);
        }
    };
    let e = llm::stream(&cfg(m.port), "0zsession1234", "hi", &cancel, &mut sink).unwrap_err();
    assert!(e.to_string().contains("interrupted"), "{e}");
}

#[test]
fn server_errors_are_actionable() {
    let m = serve(Mode::Unauthorized);
    let never = AtomicBool::new(false);
    let e = llm::chat(&cfg(m.port), "0zsession1234", "hi", &never).unwrap_err().to_string();
    assert!(e.contains("ZEROZ_AERIS_KEY"), "{e}");
    let m = serve(Mode::NoModel);
    let mut sink = |_: &str| {};
    let e = llm::stream(&cfg(m.port), "0zsession1234", "hi", &never, &mut sink).unwrap_err().to_string();
    assert!(e.contains("model not loaded"), "{e}");
}

#[test]
fn unreachable_server_says_how_to_start_it() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    drop(l);
    let never = AtomicBool::new(false);
    let e = llm::chat(&cfg(port), "0zsession1234", "hi", &never).unwrap_err().to_string();
    assert!(e.contains("not reachable") && e.contains("run_aeris.sh"), "{e}");
}

#[test]
fn agent_routes_unmatched_text_and_questions_to_the_model() {
    let m = serve(Mode::Chunked);
    let c = cfg(m.port);
    let never = AtomicBool::new(false);
    let mut toks = String::new();
    let mut sink = |t: &str| toks.push_str(t);
    // free-form question: with the model linked it must NOT fall back to web search
    let mut ctx = ChatCtx { session: "0zsession1234", cancel: &never, on_token: &mut sink };
    let r = handle_with("kya tum meri madad kar sakte ho?", &c, Some(&mut ctx));
    assert_eq!(r.tool_names, ["aeris"], "{}", r.text);
    assert_eq!(r.text, "Namaste दोस्त");
    assert_eq!(toks, "Namaste दोस्त");
    // tools still win when one matches
    let mut sink = |_: &str| {};
    let mut ctx = ChatCtx { session: "0zsession1234", cancel: &never, on_token: &mut sink };
    let r = handle_with("938+29", &c, Some(&mut ctx));
    assert!(r.text.contains("967") && r.tool_names != ["aeris"]);
    // coding policy is enforced before the model is ever called
    let before = m.seen.lock().unwrap().len();
    let mut sink = |_: &str| {};
    let mut ctx = ChatCtx { session: "0zsession1234", cancel: &never, on_token: &mut sink };
    let r = handle_with("write python code for a snake game", &c, Some(&mut ctx));
    assert!(r.refused);
    assert_eq!(m.seen.lock().unwrap().len(), before);
}

#[test]
fn model_errors_become_error_frames_and_offline_skips_the_link() {
    let m = serve(Mode::Unauthorized);
    let never = AtomicBool::new(false);
    let mut sink = |_: &str| {};
    let mut ctx = ChatCtx { session: "0zsession1234", cancel: &never, on_token: &mut sink };
    let r = handle_with("tell me something nice", &cfg(m.port), Some(&mut ctx));
    assert!(r.text.starts_with("┌ error"), "{}", r.text);
    assert!(r.text.contains("ZEROZ_AERIS_KEY"));

    let mut c = cfg(m.port);
    c.offline = true;
    let n = m.seen.lock().unwrap().len();
    let r = handle_with("tell me something nice", &c, None);
    assert!(!r.text.contains("AERIS-64 —"));
    assert_eq!(m.seen.lock().unwrap().len(), n);
}
