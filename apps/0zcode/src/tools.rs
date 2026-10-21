//! Allow-listed tools. No shell, no code execution.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::http::{get_text, url_encode};
use crate::json_util::{number_field, string_field};

#[derive(Debug)]
pub struct ToolOutput {
    pub name: &'static str,
    pub body: String,
}

impl ToolOutput {
    pub fn render(&self) -> String {
        let bar = "─".repeat(30);
        format!("┌ {} {bar}\n{}\n└{bar}", self.name, self.body)
    }
}

pub fn catalog() -> &'static str {
    "┌ tools ───────────────────────────\n\
│ weather <place>     live weather\n\
│ time                UTC time\n\
│ calc <expr>         + - * / only\n\
│ wiki <topic>        encyclopedia\n\
│ search <query>      web instant answers\n\
│ 10 USD to INR       FX convert\n\
└──────────────────────────────────\n\
slash: /tools /help /version /clear /quit\n\
policy: coding disabled"
}

pub fn weather(place: &str, cfg: &Config) -> Result<ToolOutput> {
    let q = if place.trim().is_empty() {
        "Mumbai"
    } else {
        place.trim()
    };
    let geo = get_text(
        "geocoding-api.open-meteo.com",
        &format!("/v1/search?name={}&count=1", url_encode(q)),
        cfg,
    )?;
    let lat = number_field(&geo, "latitude")
        .ok_or_else(|| Error::Tool(format!("place not found: {q}")))?;
    let lon = number_field(&geo, "longitude")
        .ok_or_else(|| Error::Tool(format!("place not found: {q}")))?;
    let name = string_field(&geo, "name").unwrap_or_else(|| q.to_string());

    let body = get_text(
        "api.open-meteo.com",
        &format!(
            "/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        ),
        cfg,
    )?;
    let temp = number_field(&body, "temperature_2m").unwrap_or(0.0);
    let hum = number_field(&body, "relative_humidity_2m").unwrap_or(0.0);
    let wind = number_field(&body, "wind_speed_10m").unwrap_or(0.0);

    Ok(ToolOutput {
        name: "weather",
        body: format!(
            "│ {name}\n│ {temp:.1}°C · humidity {hum:.0}% · wind {wind:.1} km/h\n│ source: Open-Meteo"
        ),
    })
}

pub fn time_now(cfg: &Config) -> Result<ToolOutput> {
    match get_text("worldtimeapi.org", "/api/timezone/Etc/UTC", cfg) {
        Ok(body) => {
            let dt = string_field(&body, "datetime").unwrap_or(body);
            Ok(ToolOutput {
                name: "time",
                body: format!("│ UTC {dt}"),
            })
        }
        Err(e) => Ok(ToolOutput {
            name: "time",
            body: format!("│ unavailable ({e})"),
        }),
    }
}

pub fn wikipedia(topic: &str, cfg: &Config) -> Result<ToolOutput> {
    let t = topic.trim();
    if t.is_empty() {
        return Err(Error::Tool("empty topic".into()));
    }
    let body = get_text(
        "en.wikipedia.org",
        &format!("/api/rest_v1/page/summary/{}", url_encode(t)),
        cfg,
    )?;
    let title = string_field(&body, "title").unwrap_or_else(|| t.to_string());
    let extract = string_field(&body, "extract").unwrap_or_else(|| "No summary.".into());
    let short: String = extract.chars().take(cfg.max_output_chars.min(500)).collect();
    Ok(ToolOutput {
        name: "wikipedia",
        body: format!("│ {title}\n│ {short}"),
    })
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

pub fn currency(arg: &str, cfg: &Config) -> Result<ToolOutput> {
    let upper = arg.to_uppercase();
    let amount = arg
        .split_whitespace()
        .find_map(|p| p.parse::<f64>().ok())
        .unwrap_or(1.0);
    let codes: Vec<&str> = ["USD", "INR", "EUR", "GBP"]
        .into_iter()
        .filter(|c| upper.contains(c))
        .collect();
    if codes.len() < 2 {
        return Err(Error::Tool("usage: 10 USD to INR".into()));
    }
    let (from, to) = (codes[0], codes[1]);
    let body = get_text(
        "api.frankfurter.app",
        &format!("/latest?amount={amount}&from={from}&to={to}"),
        cfg,
    )?;
    let rate = number_field(&body, to).unwrap_or(0.0);
    Ok(ToolOutput {
        name: "currency",
        body: format!("│ {amount} {from} = {rate:.4} {to}\n│ source: Frankfurter/ECB"),
    })
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
