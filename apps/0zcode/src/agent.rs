//! Request orchestration: policy → route → tools → direct answer.
//! Voice: honest, short, no corporate filler (Grok-style product behavior).

use crate::config::Config;
use crate::policy::{self, PRODUCT_NAME, PRODUCT_TAGLINE};
use crate::locale;
use crate::tools::{self, ToolOutput};

#[derive(Debug)]
pub struct AgentResponse {
    pub text: String,
    pub refused: bool,
    pub tool_names: Vec<&'static str>,
}

pub fn handle(user: &str, cfg: &Config) -> AgentResponse {
    let msg = user.trim();
    if msg.is_empty() {
        return AgentResponse {
            text: "Empty input — type a question or /help.".into(),
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

    let lower = msg.to_lowercase();

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
    let wants_time = lower.contains("what time")
        || lower.contains("time now")
        || lower.contains("current time")
        || lower.contains("date today")
        || lower.contains("tarikh")
        || lower == "time"
        || lower.starts_with("time ")
        || lower.ends_with(" time?");
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
    let wants_calc = lower.starts_with("calculate")
        || lower.starts_with("compute")
        || lower.starts_with("kitna")
        || looks_like_math(msg);

    if wants_calc {
        push_tool(&mut outputs, &mut names, tools::calculator(msg));
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
        net(&mut outputs, &mut names, tools::time_now(cfg));
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

    // Fallback: question mark → search once (don't invent answers)
    if outputs.is_empty() && msg.contains('?') {
        if cfg.offline {
            return AgentResponse {
                text: "Offline mode: only calculator and /commands work. Turn off ZEROZ_OFFLINE for search/weather.".into(),
                refused: false,
                tool_names: vec![],
            };
        }
        net(&mut outputs, &mut names, tools::web_search(msg, cfg));
    }

    if outputs.is_empty() {
        return AgentResponse {
            text: format!(
                "I don't invent answers without a tool match.\n\
Try: weather in Mumbai · time · 10 USD to INR · calculate 25*4 · who is Ada Lovelace\n\
/help lists commands. Coding requests are refused by policy."
            ),
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
