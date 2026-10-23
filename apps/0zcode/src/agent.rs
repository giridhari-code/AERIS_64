//! Request orchestration: policy → route → tools → direct answer.
//! Voice: honest, short, no corporate filler (Grok-style product behavior).

use crate::config::Config;
use crate::policy::{self, PRODUCT_NAME, PRODUCT_TAGLINE};
use crate::locale;
use crate::tools::{self, ToolOutput};
use std::sync::atomic::{AtomicBool, Ordering};

/// One full offline hint per process; further offline replies stay one short line.
static OFFLINE_HINT_SHOWN: AtomicBool = AtomicBool::new(false);

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

    // —— Task / Team / Workflow / Web meta tools ————————————————
    if lower.starts_with("taskcreate ") || lower.starts_with("task create ") {
        let rest = if lower.starts_with("taskcreate ") { &msg[11..] } else { &msg[12..] };
        let (title, detail) = rest.split_once('|').map(|(a,b)| (a.trim(), b.trim())).unwrap_or((rest.trim(), ""));
        return map_tool(crate::tasks::create(title, detail));
    }
    if lower == "tasklist" || lower == "task list" || lower == "/tasks" {
        return map_tool(crate::tasks::list());
    }
    if lower.starts_with("taskget ") || lower.starts_with("task get ") {
        let id = msg.split_whitespace().last().unwrap_or("");
        return map_tool(crate::tasks::get(id));
    }
    if lower.starts_with("taskupdate ") || lower.starts_with("task update ") {
        let parts: Vec<&str> = msg.split_whitespace().collect();
        // TaskUpdate id status [detail...]
        let id = parts.get(1).copied().unwrap_or("");
        let status = parts.get(2).copied();
        let detail = if parts.len() > 3 { Some(parts[3..].join(" ")) } else { None };
        return map_tool(crate::tasks::update(id, status, detail.as_deref()));
    }
    if lower.starts_with("taskstop ") || lower.starts_with("task stop ") {
        let id = msg.split_whitespace().last().unwrap_or("");
        return map_tool(crate::tasks::stop(id));
    }
    if lower.starts_with("teamcreate ") || lower.starts_with("team create ") {
        let rest = if lower.starts_with("teamcreate ") { &msg[11..] } else { &msg[12..] };
        let (name, members) = rest.split_once('|').map(|(a,b)| (a.trim(), b.trim())).unwrap_or((rest.trim(), "explore,test"));
        return map_tool(crate::teams::create(name, members));
    }
    if lower.starts_with("teamdelete ") || lower.starts_with("team delete ") {
        let id = msg.split_whitespace().last().unwrap_or("");
        return map_tool(crate::teams::delete(id));
    }
    if lower == "listagents" || lower == "list agents" || lower == "/agents" {
        // /agents already handled above as catalog — also full list
        return AgentResponse {
            text: crate::teams::list_agents().render(),
            refused: false,
            tool_names: vec!["ListAgents"],
        };
    }
    if lower.starts_with("sendmessage ") || lower.starts_with("send message ") {
        let rest = msg.split_whitespace().skip(1).collect::<Vec<_>>();
        if rest.len() < 2 {
            return AgentResponse { text: "SendMessage <target> <message>".into(), refused: false, tool_names: vec![] };
        }
        let target = rest[0];
        let message = rest[1..].join(" ");
        return map_tool(crate::teams::send_message(target, &message));
    }
    if lower.starts_with("workflow ") {
        return map_tool(crate::teams::workflow(msg[9..].trim()));
    }
    if lower.starts_with("webfetch ") || lower.starts_with("web fetch ") || lower.starts_with("fetch ") {
        let url = msg.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
        return map_tool(tools::web_fetch(&url, cfg));
    }
    if lower.starts_with("websearch ") || lower.starts_with("web search ") {
        let q = msg.split_whitespace().skip(2).collect::<Vec<_>>().join(" ");
        let q = if q.is_empty() { msg.split_whitespace().skip(1).collect::<Vec<_>>().join(" ") } else { q };
        return map_tool(tools::web_search(&q, cfg));
    }


    if lower == "/subagents" || lower == "/agents" {
        return AgentResponse {
            text: crate::subagents::catalog().into(),
            refused: false,
            tool_names: vec![],
        };
    }
    if let Some((kind, goal)) = crate::subagents::parse_kind_goal(msg) {
        let goal = if goal.is_empty() {
            format!("{} task", kind.as_str())
        } else {
            goal
        };
        // pin kind by prefixing so infer stays stable — harness uses infer; pass kind via goal tag
        let tagged = format!("[{}] {}", kind.as_str(), goal);
        return match crate::harness::run(&tagged, cfg) {
            Ok(o) => AgentResponse {
                text: o.render(),
                refused: false,
                tool_names: vec!["agent"],
            },
            Err(e) => AgentResponse {
                text: format!("agent: {e}"),
                refused: false,
                tool_names: vec![],
            },
        };
    }
    if let Some(goal) = crate::harness::parse_goal(msg) {
        return match crate::harness::run(&goal, cfg) {
            Ok(o) => AgentResponse {
                text: o.render(),
                refused: false,
                tool_names: vec!["agent"],
            },
            Err(e) => AgentResponse {
                text: format!("agent: {e}"),
                refused: false,
                tool_names: vec![],
            },
        };
    }

    // direct write/mkdir (single-step) when gated
    if lower.starts_with("write file ") || lower.starts_with("ws write ") {
        let rest = if lower.starts_with("write file ") {
            msg["write file ".len()..].trim()
        } else {
            msg["ws write ".len()..].trim()
        };
        let (path, body) = match rest.split_once('|') {
            Some((p, b)) => (p.trim(), b.trim()),
            None => {
                let mut sp = rest.splitn(2, char::is_whitespace);
                (sp.next().unwrap_or(""), sp.next().unwrap_or(""))
            }
        };
        return match crate::workspace::write_file(path, body) {
            Ok(o) => AgentResponse {
                text: o.render(),
                refused: false,
                tool_names: vec!["ws.write"],
            },
            Err(e) => AgentResponse {
                text: format!("{e}"),
                refused: false,
                tool_names: vec![],
            },
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
    if matches!(lower.as_str(), "/llm" | "llm" | "/connect" | "/model-status") {
        return AgentResponse {
            text: crate::llm::diagnose(cfg),
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
    let wants_ws_list = lower == "list" || lower == "ls" || lower == "list files"
        || lower.starts_with("list files")
        || lower.starts_with("list dir")
        || lower.starts_with("show files")
        || lower.starts_with("project structure")
        || lower == "/ws" || lower == "/workspace" || lower.starts_with("/ws ");
    let wants_ws_read = lower.starts_with("read ")
        || lower.starts_with("show file ")
        || lower.starts_with("open file ")
        || lower.starts_with("cat ");
    let wants_ws_grep = lower.starts_with("grep ")
        || lower.starts_with("search in files ")
        || lower.starts_with("find in code ");
    let wants_shell = lower.starts_with("shell ")
        || lower.starts_with("$ ")
        || lower.starts_with("run allowlisted ");
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

    if wants_ws_list {
        let path = if lower.starts_with("/ws ") {
            msg[4..].trim()
        } else {
            "."
        };
        push_tool(&mut outputs, &mut names, crate::workspace::list(path));
    }
    if lower == "/ws" || lower == "/workspace" {
        push_tool(&mut outputs, &mut names, Ok(crate::workspace::status()));
    }
    if wants_ws_read {
        let path = msg.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
        push_tool(&mut outputs, &mut names, crate::workspace::read(&path));
    }
    if wants_ws_grep {
        let rest = msg.split_whitespace().skip(1).collect::<Vec<_>>();
        let pat = rest.first().copied().unwrap_or("");
        let path = if rest.len() > 1 { rest[1..].join(" ") } else { ".".into() };
        push_tool(&mut outputs, &mut names, crate::workspace::grep(pat, &path));
    }
    if wants_shell {
        let cmd_owned: String = if lower.starts_with("$ ") {
            msg[2..].trim().to_string()
        } else if lower.starts_with("shell ") {
            msg[6..].trim().to_string()
        } else {
            msg.split_whitespace().skip(2).collect::<Vec<_>>().join(" ")
        };
        push_tool(
            &mut outputs,
            &mut names,
            crate::workspace::shell_allowlisted(cmd_owned.trim()),
        );
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
    // Company-standard: soft reconnect before chat (Claude Code / OpenCode style).
    if let Some(url) = cfg.aeris_url.as_deref() {
        if matches!(
            crate::aeris_boot::probe_status(url),
            crate::aeris_boot::AerisStatus::Down
        ) {
            // Mutate via env-backed re-read is hard; just surface diagnose — boot is owned by main.
            // One-shot soft message on first fail still from below.
        }
    }
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
        Ok(text) if !text.trim().is_empty() => {
            OFFLINE_HINT_SHOWN.store(false, Ordering::Relaxed);
            AgentResponse {
                text,
                refused: false,
                tool_names: vec!["aeris"],
            }
        }
        Ok(_) => AgentResponse {
            text: "AERIS returned empty (checkpoint may need more training).".into(),
            refused: false,
            tool_names: vec![],
        },
        Err(e) => {
            let s = e.to_string();
            let short = if s.contains("not reachable") || s.contains("Connection refused") {
                if OFFLINE_HINT_SHOWN.swap(true, Ordering::Relaxed) {
                    String::new()
                } else {
                    format!(
                        "LLM offline (not reachable).\n{}",
                        crate::llm::diagnose(cfg)
                    )
                }
            } else if s.contains("model not loaded") || s.contains("not loaded") {
                if OFFLINE_HINT_SHOWN.swap(true, Ordering::Relaxed) {
                    String::new()
                } else {
                    format!(
                        "Model not loaded.\n{}",
                        crate::llm::diagnose(cfg)
                    )
                }
            } else if s.len() > 120 {
                format!("{}…", s.chars().take(120).collect::<String>())
            } else {
                s
            };
            if short.is_empty() {
                return AgentResponse {
                    text: String::new(),
                    refused: false,
                    tool_names: vec!["_silent"],
                };
            }
            AgentResponse {
                text: short,
                refused: false,
                tool_names: vec![],
            }
        }
    }
}


fn map_tool(r: crate::error::Result<crate::tools::ToolOutput>) -> AgentResponse {
    match r {
        Ok(o) => AgentResponse {
            text: o.render(),
            refused: false,
            tool_names: vec![o.name],
        },
        Err(e) => AgentResponse {
            text: format!("{e}"),
            refused: false,
            tool_names: vec![],
        },
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
