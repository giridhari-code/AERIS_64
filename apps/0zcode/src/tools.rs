//! Core tools: web search / fetch, calculator, catalog. File + shell tools live in `workspace`.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::http::{get_text, url_encode};
use crate::json_util::string_field;

#[derive(Debug)]
pub struct ToolOutput {
    pub name: &'static str,
    pub body: String,
}

impl ToolOutput {
    /// Plain body lines for TUI / one-shot (no box-drawing).
    pub fn render(&self) -> String {
        self.body
            .lines()
            .map(|l| {
                let s = l.trim().trim_start_matches('│').trim();
                s.to_string()
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn catalog() -> &'static str {
    "┌ tools ────────────────────────────────
│ read <path> [start] [n]   read a file (line range)
│ list [path]               list a folder
│ glob <pattern> [path]     find files by name (*.rs, src/**/*.rs)
│ grep <pattern> [path]     search file contents
│ edit / write              change files (asks approval)
│ shell <cmd>               allowlisted cargo / git (asks approval)
│ fetch <url> · search <q>  web page / web search
│ calc <expr>               + - * / only
│ note save/list/read       sandbox notes only
└────────────────────────────────────────
agent loop: /code <goal>  (needs ZEROZ_CODING=1 ZEROZ_AGENT=1)
slash: /help /tools /mcp /status /history /copy /quit"
}

pub fn web_search(query: &str, cfg: &Config) -> Result<ToolOutput> {
    let q = query.trim();
    if q.is_empty() {
        return Err(Error::Tool("empty query".into()));
    }
    let body = get_text(
        "api.duckduckgo.com",
        &format!("/?q={}&format=json&no_html=1&skip_disambig=1", url_encode(q)),
        cfg,
    )?;
    if let Some(abs) = string_field(&body, "AbstractText") {
        if !abs.is_empty() {
            let heading = string_field(&body, "Heading").unwrap_or_default();
            return Ok(ToolOutput {
                name: "search",
                body: format!("│ {heading}\n│ {abs}"),
            });
        }
    }
    Ok(ToolOutput {
        name: "search",
        body: format!("│ No instant answer for: {q}"),
    })
}

pub fn web_fetch(url: &str, cfg: &Config) -> Result<ToolOutput> {
    let url = url.trim();
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| Error::Tool("WebFetch: only http:// or https:// URLs".into()))?;
    let (host, path) = match rest.split_once('/') {
        Some((h, p)) => (h.to_string(), format!("/{p}")),
        None => (rest.to_string(), "/".into()),
    };
    // std client is HTTP-only; https needs curl (same as MCP)
    let body = if url.starts_with("https://") {
        fetch_https_curl(url, cfg.http_timeout)?
    } else {
        let host_only = host.split(':').next().unwrap_or(&host);
        get_text(host_only, &path, cfg)?
    };
    // strip tags lightly
    let text = strip_tags(&body);
    let short: String = text.chars().take(cfg.max_output_chars.min(3000)).collect();
    Ok(ToolOutput {
        name: "WebFetch",
        body: format!("│ {url}\n│ {}\n│ source: WebFetch", short.replace('\n', "\n│ ")),
    })
}

fn strip_tags(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn fetch_https_curl(url: &str, timeout: std::time::Duration) -> Result<String> {
    use std::process::Command;
    let secs = timeout.as_secs().clamp(1, 60).to_string();
    let out = Command::new("curl")
        .args(["-sS", "-L", "--max-time", &secs, "-H", "Accept: text/html,text/plain,*/*", url])
        .output()
        .map_err(|e| Error::Tool(format!("WebFetch https needs curl ({e})")))?;
    if !out.status.success() {
        return Err(Error::Tool(format!(
            "WebFetch curl failed: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(200).collect::<String>()
        )));
    }
    String::from_utf8(out.stdout).map_err(|e| Error::Parse(e.to_string()))
}


pub fn calculator(expr: &str) -> Result<ToolOutput> {
    let e = expr
        .to_lowercase()
        .replace("calculate", "")
        .replace("compute", "")
        .replace("what is", "")
        .replace('?', "")
        .trim()
        .to_string();
    let v = eval_expr(&e).map_err(Error::Tool)?;
    Ok(ToolOutput {
        name: "calc",
        body: format!("│ {e} = {v}"),
    })
}



fn eval_expr(s: &str) -> std::result::Result<f64, String> {
    if !s.chars().all(|c| c.is_ascii_digit() || " +-*/.()".contains(c)) {
        return Err("only + - * / and digits allowed".into());
    }
    let cs: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut i = 0;
    let v = parse_expr(&cs, &mut i)?;
    if i != cs.len() {
        return Err("trailing input".into());
    }
    Ok(v)
}

fn parse_expr(cs: &[char], i: &mut usize) -> std::result::Result<f64, String> {
    let mut v = parse_term(cs, i)?;
    while *i < cs.len() && (cs[*i] == '+' || cs[*i] == '-') {
        let op = cs[*i];
        *i += 1;
        let r = parse_term(cs, i)?;
        v = if op == '+' { v + r } else { v - r };
    }
    Ok(v)
}

fn parse_term(cs: &[char], i: &mut usize) -> std::result::Result<f64, String> {
    let mut v = parse_factor(cs, i)?;
    while *i < cs.len() && (cs[*i] == '*' || cs[*i] == '/') {
        let op = cs[*i];
        *i += 1;
        let r = parse_factor(cs, i)?;
        v = if op == '*' {
            v * r
        } else if r == 0.0 {
            return Err("division by zero".into());
        } else {
            v / r
        };
    }
    Ok(v)
}

fn parse_factor(cs: &[char], i: &mut usize) -> std::result::Result<f64, String> {
    if *i < cs.len() && cs[*i] == '(' {
        *i += 1;
        let v = parse_expr(cs, i)?;
        if *i >= cs.len() || cs[*i] != ')' {
            return Err("mismatched parentheses".into());
        }
        *i += 1;
        return Ok(v);
    }
    let start = *i;
    if *i < cs.len() && (cs[*i] == '+' || cs[*i] == '-') {
        *i += 1;
    }
    while *i < cs.len() && (cs[*i].is_ascii_digit() || cs[*i] == '.') {
        *i += 1;
    }
    if start == *i {
        return Err("expected number".into());
    }
    cs[start..*i]
        .iter()
        .collect::<String>()
        .parse()
        .map_err(|_| "invalid number".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_basic() {
        let o = calculator("25*4").unwrap();
        assert!(o.body.contains("100"));
    }
}
