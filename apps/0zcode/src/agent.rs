//! Request orchestration: policy → route → tools → direct answer.
//! Voice: honest, short, no corporate filler (Grok-style product behavior).

use crate::config::Config;
use crate::policy::{self, PRODUCT_NAME, PRODUCT_TAGLINE};
use crate::locale;
use crate::tools::{self, ToolOutput};
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub struct AgentResponse {
    pub text: String,
    pub refused: bool,
    pub tool_names: Vec<&'static str>,
}

/// Live-chat plumbing for the interactive TUI: the AERIS session, a flag the
/// UI sets on Esc, and a sink that receives tokens as they are generated.
pub struct ChatCtx<'a> {
    pub session: &'a str,
    pub cancel: &'a AtomicBool,
    pub on_token: &'a mut dyn FnMut(&str),
}

/// True when free-form messages should try the AERIS-64 model (URL configured).
/// Does **not** guarantee the server is up — call sites still handle errors.
pub fn llm_linked(cfg: &Config) -> bool {
    cfg.aeris_url.is_some() && !cfg.offline
}

/// True when /readyz reports the model is loaded (live probe).
pub fn llm_online(cfg: &Config) -> bool {
    if cfg.offline {
        return false;
    }
    let Some(url) = cfg.aeris_url.as_deref() else {
        return false;
    };
    crate::aeris_boot::probe_status(url) == crate::aeris_boot::AerisStatus::Ready
}

/// One-shot entry (no streaming, fresh AERIS session).
pub fn handle(user: &str, cfg: &Config) -> AgentResponse {
    handle_with(user, cfg, None)
}

pub fn handle_with(user: &str, cfg: &Config, chat: Option<&mut ChatCtx>) -> AgentResponse {
    let msg = user.trim();
    if msg.is_empty() {
        return AgentResponse {
            text: "Empty input — type a question or /help.".into(),
            refused: false,
            tool_names: vec![],
        };
    }

    let lower = msg.to_lowercase();

    if is_greeting(&lower) {
        return AgentResponse {
            text: greeting_reply(&lower),
            refused: false,
            tool_names: vec![],
        };
    }

    if policy::is_code_request(msg) {
        return AgentResponse {
            text: policy::refusal_message(),
            refused: true,
            tool_names: vec![],
        };
    }

    if let Some(mcp_msg) = crate::mcp_store::handle_command(msg) {
        return AgentResponse {
            text: mcp_msg,
            refused: false,
            tool_names: vec![],
        };
    }

    // —— meta / slash (deterministic) ————————————————————————————————
    if matches!(
        lower.as_str(),
        "/quit" | "/exit" | "quit" | "exit" | "q" | "/q"
    ) {
        return AgentResponse {
            text: "__QUIT__".into(),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/clear") {
        return AgentResponse {
            text: "__CLEAR__".into(),
            refused: false,
            tool_names: vec![],
        };
    }
    if matches!(lower.as_str(), "/help" | "help" | "?") {
        return AgentResponse {
            text: help_text(),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/mcp") || lower == "mcp" {
        return AgentResponse {
            text: crate::registry::tools_json(cfg),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/tools") {
        return AgentResponse {
            text: tools::catalog().into(),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/status") {
        return AgentResponse {
            text: format!(
                "{PRODUCT_NAME} v{}\n{PRODUCT_TAGLINE}\nmode: research tools only — no code gen\noffline: {}\nmodel label: {}\nconfig: {}",
                env!("CARGO_PKG_VERSION"),
                cfg.offline,
                cfg.model_label,
                Config::config_path_display()
            ),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/version") || lower == "version" {
        return AgentResponse {
            text: policy::banner_line(),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/model") {
        return AgentResponse {
            text: format!(
                "Display label: {}\n(This is cosmetic. Tools run locally via HTTP APIs; set ZEROZ_MODEL_LABEL to rename.)",
                cfg.model_label
            ),
            refused: false,
            tool_names: vec![],
        };
    }

    // —— tool plan (multi-intent OK) ————————————————————————————————
    let mut outputs: Vec<ToolOutput> = Vec::new();
    let mut names: Vec<&'static str> = Vec::new();

    let wants_weather = lower.contains("weather")
        || lower.contains("mausam")
        || lower.contains("temperature")
        || lower.contains("forecast");
    let wants_time = has_word(&lower, "time")
        || has_word(&lower, "tarikh")
        || lower.contains("what time")
        || lower.contains("time now")
        || lower.contains("current time")
        || lower.contains("date today")
        || lower.contains("kitna baja")
        || lower.contains("kya time");
    let wants_fx = (lower.contains("usd")
        || lower.contains("eur")
        || lower.contains("inr")
        || lower.contains("gbp")
        || lower.contains("jpy"))
        && (lower.contains("to") || lower.contains(" in ") || lower.contains("convert"));
    let wants_wiki = lower.contains("wikipedia")
        || lower.starts_with("who is ")
        || lower.starts_with("who was ")
        || lower.starts_with("what is ")
        || lower.contains("tell me about ")
        || lower.contains("ke bare mein")
        || lower.contains("kaun hai");
    let wants_search = lower.contains("search")
        || lower.contains("look up")
        || lower.contains("latest news")
        || lower.contains("headline")
        || lower.contains("khabar");
    let math_expr = extract_math_expr(msg);
    let wants_calc = lower.starts_with("calculate")
        || lower.starts_with("compute")
        || lower.starts_with("kitna")
        || looks_like_math(msg)
        || math_expr.is_some();

    // Strict units — avoid matching "to finds" / random English
    let wants_units = (lower.contains(" km ") || lower.contains(" km to") || lower.starts_with("km "))
        || lower.contains(" to mi")
        || lower.contains(" to km")
        || lower.contains("c to f")
        || lower.contains("f to c")
        || lower.contains("°c")
        || lower.contains("°f")
        || lower.contains("celsius")
        || lower.contains("fahrenheit")
        || lower.starts_with("convert ")
        || lower.starts_with("units ");
    let wants_crypto = lower.contains("btc")
        || lower.contains("bitcoin")
        || lower.contains("eth price")
        || lower.contains("ethereum")
        || lower.contains("crypto")
        || (lower.contains("price") && (lower.contains("btc") || lower.contains("eth") || lower.contains("sol")));
    let wants_github = lower.contains("github.com/")
        || lower.starts_with("github ")
        || (lower.contains("github") && lower.contains('/'));
    let wants_headlines = lower.contains("headlines")
        || lower == "news"
        || lower.starts_with("top news")
        || lower.contains("bbc news");
    let wants_email = lower.contains("email")
        || lower.contains("e-mail")
        || lower.contains("inbox")
        || lower.contains("aaj ke mail")
        || lower.contains("aaj ka email")
        || lower.contains("my mail");
    let wants_notes = lower.starts_with("note ")
        || lower.starts_with("notes ")
        || lower.starts_with("note:")
        || lower == "notes"
        || lower == "note list";

    if wants_calc {
        let calc_in = math_expr.as_deref().unwrap_or(msg);
        push_tool(&mut outputs, &mut names, tools::calculator(calc_in));
    }
    if wants_units {
        push_tool(&mut outputs, &mut names, tools::units(msg));
    }

    let net = |outputs: &mut Vec<ToolOutput>, names: &mut Vec<&'static str>, r: crate::error::Result<ToolOutput>| {
        if cfg.offline {
            push_tool(
                outputs,
                names,
                Err(crate::error::Error::Offline(
                    "network tools disabled (ZEROZ_OFFLINE or config offline=true); calc still works".into(),
                )),
            );
        } else {
            push_tool(outputs, names, r);
        }
    };

    if wants_fx {
        net(&mut outputs, &mut names, tools::currency(msg, cfg));
    }
    if wants_time {
        if let Some(place) = extract_place(msg) {
            net(&mut outputs, &mut names, tools::time_in_place(&place, cfg));
        } else {
            net(&mut outputs, &mut names, tools::time_now(cfg));
        }
    }
    if wants_crypto {
        let sym = extract_crypto_sym(msg);
        net(&mut outputs, &mut names, tools::crypto_price(&sym, cfg));
    }
    if wants_github {
        net(&mut outputs, &mut names, tools::github_repo(msg, cfg));
    }
    if wants_headlines {
        net(&mut outputs, &mut names, tools::headlines(cfg));
    }
    if wants_email {
        net(&mut outputs, &mut names, tools::email_status(cfg));
    }
    if wants_notes {
        push_tool(&mut outputs, &mut names, crate::notes::handle_notes(msg));
    }
    if wants_weather {
        let place = extract_place(msg).unwrap_or_else(|| "Delhi".into());
        net(&mut outputs, &mut names, tools::weather(&place, cfg));
    }
    if wants_wiki && !wants_search {
        let topic = strip_wiki_topic(msg);
        net(&mut outputs, &mut names, tools::wikipedia(&topic, cfg));
    }
    if wants_search {
        net(&mut outputs, &mut names, tools::web_search(msg, cfg));
    }

    // User-configured MCP tools (trigger match)
    if !cfg.offline {
        for ttool in crate::mcp_store::match_tools(msg) {
            // skip if same name already ran as builtin path
            push_tool(
                &mut outputs,
                &mut names,
                crate::mcp_store::invoke(&ttool, msg, cfg),
            );
        }
    }

    // Fallback: question mark → search once (don't invent answers).
    // With AERIS linked the model answers instead.
    if outputs.is_empty() && msg.contains('?') && !llm_linked(cfg) {
        if cfg.offline {
            return AgentResponse {
                text: "Offline mode: calculator + units + /commands only. Turn off ZEROZ_OFFLINE for search/weather.".into(),
                refused: false,
                tool_names: vec![],
            };
        }
        net(&mut outputs, &mut names, tools::web_search(msg, cfg));
    }

    // No tool matched → the AERIS-64 model answers (when linked).
    if outputs.is_empty() {
        if llm_linked(cfg) {
            return ask_aeris(msg, cfg, chat);
        }
        let lang = locale::detect(msg);
        return AgentResponse {
            text: locale::empty_match_hint(lang).to_string(),
            refused: false,
            tool_names: vec![],
        };
    }

    let text = outputs
        .iter()
        .map(ToolOutput::render)
        .collect::<Vec<_>>()
        .join("\n\n");

    AgentResponse {
        text,
        refused: false,
        tool_names: names,
    }
}

/// Ask the AERIS-64 server. Streams when the UI provides a [`ChatCtx`] and
/// `aeris_stream` is on; otherwise one `POST /v1/chat`. Failures come back as
/// an `error` frame so the UI renders them as an error block.
fn ask_aeris(msg: &str, cfg: &Config, chat: Option<&mut ChatCtx>) -> AgentResponse {
    let fresh = crate::llm::new_session_id();
    let never = AtomicBool::new(false);
    let result = match chat {
        Some(ctx) if cfg.aeris_stream => {
            crate::llm::stream(cfg, ctx.session, msg, ctx.cancel, &mut *ctx.on_token)
        }
        Some(ctx) => crate::llm::chat(cfg, ctx.session, msg, ctx.cancel),
        None => crate::llm::chat(cfg, &fresh, msg, &never),
    };
    match result {
        Ok(text) if !text.trim().is_empty() => AgentResponse {
            text,
            refused: false,
            tool_names: vec!["aeris"],
        },
        Ok(_) => AgentResponse {
            text: "AERIS returned empty (checkpoint may need more training).".into(),
            refused: false,
            tool_names: vec![],
        },
        Err(e) => {
            // Short UI line — no multi-line uvicorn paste on every keystroke
            let s = e.to_string();
            let short = if s.contains("not reachable") {
                "AERIS offline — tools only (time · calc · /help). Start: scripts/zeroz/run_aeris.sh".to_string()
            } else if s.contains("model not loaded") || s.contains("not loaded") {
                "AERIS: model not loaded — set NEUROFIELD_CHECKPOINT".to_string()
            } else if s.len() > 140 {
                format!("{}…", s.chars().take(140).collect::<String>())
            } else {
                s
            };
            AgentResponse {
                text: short,
                refused: false,
                tool_names: vec![],
            }
        }
    }
}


fn push_tool(
    outputs: &mut Vec<ToolOutput>,
    names: &mut Vec<&'static str>,
    r: crate::error::Result<ToolOutput>,
) {
    match r {
        Ok(o) => {
            names.push(o.name);
            outputs.push(o);
        }
        Err(e) => outputs.push(ToolOutput {
            name: "error",
            body: format!("│ {e}\n│ (honest miss — try another phrasing or check network)"),
        }),
    }
}

fn has_word(hay: &str, word: &str) -> bool {
    hay.split(|c: char| !c.is_alphanumeric())
        .any(|w| w == word)
}

fn is_greeting(lower: &str) -> bool {
    let t = lower.trim().trim_end_matches(['!', '.', ',']);
    matches!(
        t,
        "hi" | "hello" | "hey" | "hiya" | "yo" | "sup"
            | "hi sir" | "hello sir" | "hey sir"
            | "good morning" | "good evening" | "good afternoon"
            | "namaste" | "namaskar" | "hola"
    ) || t.starts_with("hi ") && t.len() < 16
}

fn greeting_reply(lower: &str) -> String {
    if lower.contains("namaste") || lower.contains("namaskar") {
        "Namaste — main 0zcode hoon. Weather, time, calc, search, notes, MCP tools. /help".into()
    } else {
        "Hi — 0zcode here. I answer with tools, not guesses.\nTry: time · weather in Mumbai · 938+29 · /help".into()
    }
}

/// Pull first arithmetic expression like 938+29 or 25*4 from free text.
fn extract_math_expr(msg: &str) -> Option<String> {
    let chars: Vec<char> = msg.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut has_op = false;
            while i < chars.len() {
                let c = chars[i];
                if c.is_ascii_digit() || c == '.' {
                    i += 1;
                } else if "+-*/".contains(c) {
                    has_op = true;
                    i += 1;
                } else if c == '(' || c == ')' || c == ' ' {
                    i += 1;
                } else {
                    break;
                }
            }
            if has_op {
                let expr: String = chars[start..i].iter().collect();
                let expr = expr.trim().to_string();
                if looks_like_math(&expr) || expr.chars().any(|c| "+-*/".contains(c)) {
                    return Some(expr);
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

fn extract_crypto_sym(msg: &str) -> String {
    let l = msg.to_lowercase();
    for s in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "dogecoin", "doge"] {
        if l.contains(s) {
            return s.to_string();
        }
    }
    "bitcoin".into()
}

fn looks_like_math(msg: &str) -> bool {
    let t = msg.trim();
    if t.is_empty() {
        return false;
    }
    let has_op = t.chars().any(|c| "+-*/".contains(c));
    let mostly_math = t
        .chars()
        .all(|c| c.is_ascii_digit() || "+-*/().% ".contains(c));
    has_op && mostly_math
}

fn help_text() -> String {
    format!(
        "{PRODUCT_NAME} — direct answers from tools, not guesses.\n\n\
Commands\n\
  /help /tools /status /model /version /history /clear /quit\n\n\
Examples\n\
  weather in Mumbai\n\
  time\n\
  10 USD to INR\n\
  calculate 25*4\n\
  who is Ada Lovelace\n\
  search latest AI news\n\n\
Policy: no source-code write/edit/execute."
    )
}

fn extract_place(msg: &str) -> Option<String> {
    let lower = msg.to_lowercase();
    for key in [" in ", " at ", " for "] {
        if let Some(i) = lower.find(key) {
            let rest = msg[i + key.len()..].trim().trim_matches('?');
            if rest.is_empty() {
                continue;
            }
            // stop at "and" / comma for multi-intent
            let cut = rest
                .split(|c| c == ',' || c == '?')
                .next()
                .unwrap_or(rest);
            let cut = cut
                .split_whitespace()
                .take_while(|w| {
                    let l = w.to_lowercase();
                    l != "and" && l != "aur" && l != "time" && l != "date"
                })
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            if !cut.is_empty() {
                return Some(cut);
            }
        }
    }
    None
}

fn strip_wiki_topic(msg: &str) -> String {
    let mut t = msg.to_lowercase();
    for p in [
        "wikipedia",
        "who is ",
        "who was ",
        "what is ",
        "tell me about ",
        "ke bare mein ",
        "kaun hai ",
    ] {
        if let Some(s) = t.strip_prefix(p) {
            t = s.to_string();
        }
        t = t.replace(p, "");
    }
    t.trim().trim_matches('?').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn math_detect() {
        assert!(looks_like_math("2+2"));
        assert!(looks_like_math("15 * 3"));
        assert!(!looks_like_math("weather in Delhi"));
    }

    #[test]
    fn extract_math_from_sentence() {
        assert_eq!(extract_math_expr("now find to 938+29").as_deref(), Some("938+29"));
    }

    #[test]
    fn greeting_hi() {
        let r = handle("hi sir", &Config::default());
        assert!(r.text.to_lowercase().contains("hi") || r.text.contains("0zcode"));
        assert!(!r.refused);
    }

    #[test]
    fn how_time_matches() {
        let r = handle("how time", &Config::default());
        assert!(r.tool_names.iter().any(|n| *n == "time") || r.text.contains("UTC") || r.text.contains("device"));
    }

    #[test]
    fn google_sentence_not_units() {
        let r = handle("go google to finds code to hello worlds", &Config::default());
        assert!(!r.tool_names.iter().any(|n| *n == "units"));
    }

    #[test]
    fn place_stops_at_and() {
        let p = extract_place("weather in Delhi and what time");
        assert_eq!(p.as_deref(), Some("Delhi"));
    }

    #[test]
    fn help_slash() {
        let r = handle("/help", &Config::default());
        assert!(r.text.contains("Commands"));
        assert!(!r.refused);
    }

    #[test]
    fn refuses_code() {
        let r = handle("write a python function", &Config::default());
        assert!(r.refused);
    }

    #[test]
    fn offline_calc_works() {
        let mut c = Config::default();
        c.offline = true;
        let r = handle("2+2", &c);
        assert!(!r.refused);
        assert!(r.text.contains('4') || r.tool_names.contains(&"calculator"));
    }
}
