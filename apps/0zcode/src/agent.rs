//! Request orchestration: policy → route → tools → response.

use crate::config::Config;
use crate::policy::{self, PRODUCT_NAME, PRODUCT_TAGLINE};
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
            text: "Empty input.".into(),
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
    if lower == "/help" || lower == "help" || lower == "?" {
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
                "product: {PRODUCT_NAME}\nversion: {}\ntagline: {PRODUCT_TAGLINE}\nmode: research tools only (no code)",
                env!("CARGO_PKG_VERSION")
            ),
            refused: false,
            tool_names: vec![],
        };
    }
    if lower.starts_with("/version") {
        return AgentResponse {
            text: policy::banner_line(),
            refused: false,
            tool_names: vec![],
        };
    }

    let mut outputs: Vec<ToolOutput> = Vec::new();
    let mut names: Vec<&'static str> = Vec::new();

    let push = |outputs: &mut Vec<ToolOutput>, names: &mut Vec<&'static str>, r: crate::error::Result<ToolOutput>| {
        match r {
            Ok(o) => {
                names.push(o.name);
                outputs.push(o);
            }
            Err(e) => outputs.push(ToolOutput {
                name: "error",
                body: format!("│ {e}"),
            }),
        }
    };

    if lower.contains("weather") || lower.contains("mausam") {
        let place = extract_place(msg).unwrap_or_else(|| "Delhi".into());
        push(&mut outputs, &mut names, tools::weather(&place, cfg));
    }
    if lower.contains("time") || lower.contains("date") || lower.contains("tarikh") {
        push(&mut outputs, &mut names, tools::time_now(cfg));
    }
    if (lower.contains("usd") || lower.contains("eur") || lower.contains("inr") || lower.contains("gbp"))
        && (lower.contains("to") || lower.contains(" in "))
    {
        push(&mut outputs, &mut names, tools::currency(msg, cfg));
    }
    if lower.contains("wikipedia")
        || lower.starts_with("who is")
        || lower.starts_with("what is")
        || lower.contains("tell me about")
    {
        let topic = strip_prefixes(
            &lower,
            &["wikipedia", "who is", "what is", "tell me about"],
        );
        push(&mut outputs, &mut names, tools::wikipedia(&topic, cfg));
    }
    if (lower.contains("search") || lower.contains("news") || lower.contains("look up"))
        && outputs.is_empty()
    {
        push(&mut outputs, &mut names, tools::web_search(msg, cfg));
    }
    if lower.starts_with("calculate")
        || lower.starts_with("compute")
        || msg
            .chars()
            .all(|c| c.is_ascii_digit() || " +-*/.()".contains(c))
    {
        push(&mut outputs, &mut names, tools::calculator(msg));
    }

    if outputs.is_empty() {
        if msg.contains('?') {
            push(&mut outputs, &mut names, tools::web_search(msg, cfg));
        } else {
            return AgentResponse {
                text: format!(
                    "{PRODUCT_NAME} — {PRODUCT_TAGLINE}\n\
Try: weather in Mumbai · time · 10 USD to INR · calculate 25*4\n\
/tools  /version  /quit"
                ),
                refused: false,
                tool_names: vec![],
            };
        }
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

fn help_text() -> String {
    format!(
        "{PRODUCT_NAME} — commands\n\n  /help      this message\n  /tools     list tools\n  /status    product status\n  /version   version banner\n  /clear     clear screen (REPL)\n  /quit      exit REPL\n\nExamples:\n  weather in Mumbai\n  time\n  10 USD to INR\n  calculate 25*4\n  who is Ada Lovelace"
    )
}

fn extract_place(msg: &str) -> Option<String> {
    let lower = msg.to_lowercase();
    for key in [" in ", " at ", " for "] {
        if let Some(i) = lower.find(key) {
            let rest = msg[i + key.len()..].trim().trim_matches('?');
            if !rest.is_empty() {
                return Some(
                    rest.split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }
        }
    }
    None
}

fn strip_prefixes(s: &str, prefixes: &[&str]) -> String {
    let mut t = s.to_string();
    for p in prefixes {
        if let Some(stripped) = t.strip_prefix(p) {
            t = stripped.to_string();
        }
        t = t.replace(p, "");
    }
    t.trim().trim_matches('?').to_string()
}
