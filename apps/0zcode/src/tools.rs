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
        let bar = "─".repeat(36);
        format!("┌ {} {bar}\n{}\n└{bar}", self.name, self.body)
    }
}

pub fn catalog() -> &'static str {
    "┌ tools ────────────────────────────────
│ weather <place>     live conditions (Open-Meteo)
│ time [in City]      UTC or city timezone
│ calc <expr>         + - * / only
│ units 10 km to mi   length / temp convert
│ wiki <topic>        short encyclopedia
│ search <query>      web instant answers
│ 10 USD to INR       FX convert
│ btc price           crypto (CoinGecko)
│ github owner/repo   public repo info
│ headlines           BBC RSS top stories
└────────────────────────────────────────
slash: /help /tools /status /history /copy /quit
policy: coding disabled — research tools only"
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
            let dt = string_field(&body, "datetime").unwrap_or_else(|| body.clone());
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


pub fn units(expr: &str) -> Result<ToolOutput> {
    // Offline: "10 km to mi", "32 c to f", "100 f to c"
    let e = expr.to_lowercase().replace('?', "");
    let e = e.replace("convert", " ").replace("units", " ");
    let parts: Vec<&str> = e.split_whitespace().collect();
    // find number
    let mut num: Option<f64> = None;
    let mut from = "";
    let mut to = "";
    for (i, w) in parts.iter().enumerate() {
        if let Ok(n) = w.parse::<f64>() {
            num = Some(n);
            if i + 1 < parts.len() {
                from = parts[i + 1];
            }
            // look for "to" X
            if let Some(ti) = parts.iter().position(|x| *x == "to" || *x == "in") {
                if ti + 1 < parts.len() {
                    to = parts[ti + 1];
                }
            }
            break;
        }
    }
    let Some(n) = num else {
        return Err(Error::Tool("usage: 10 km to mi · 32 c to f".into()));
    };
    let from = from.trim_matches(|c: char| !c.is_alphanumeric());
    let to = to.trim_matches(|c: char| !c.is_alphanumeric());
    let (out, note) = convert_units(n, from, to)?;
    Ok(ToolOutput {
        name: "units",
        body: format!("│ {n} {from} = {out:.6} {to}\n│ {note}"),
    })
}

fn convert_units(n: f64, from: &str, to: &str) -> Result<(f64, &'static str)> {
    let f = from.to_lowercase();
    let t = to.to_lowercase();
    let pair = (f.as_str(), t.as_str());
    let (v, note) = match pair {
        ("km", "mi") | ("km", "mile") | ("km", "miles") => (n * 0.621371, "length"),
        ("mi", "km") | ("mile", "km") | ("miles", "km") => (n / 0.621371, "length"),
        ("m", "ft") | ("meter", "ft") | ("metres", "ft") => (n * 3.28084, "length"),
        ("ft", "m") | ("feet", "m") => (n / 3.28084, "length"),
        ("kg", "lb") | ("kg", "lbs") => (n * 2.20462, "mass"),
        ("lb", "kg") | ("lbs", "kg") => (n / 2.20462, "mass"),
        ("c", "f") | ("celsius", "f") | ("°c", "f") => (n * 9.0 / 5.0 + 32.0, "temperature"),
        ("f", "c") | ("fahrenheit", "c") | ("°f", "c") => ((n - 32.0) * 5.0 / 9.0, "temperature"),
        ("c", "k") | ("celsius", "k") => (n + 273.15, "temperature"),
        ("k", "c") => (n - 273.15, "temperature"),
        _ => {
            return Err(Error::Tool(format!(
                "unsupported {from} → {to}; try km/mi, m/ft, kg/lb, c/f"
            )))
        }
    };
    Ok((v, note))
}

pub fn crypto_price(symbol: &str, cfg: &Config) -> Result<ToolOutput> {
    let s = symbol.trim().to_lowercase();
    let id = match s.as_str() {
        "btc" | "bitcoin" => "bitcoin",
        "eth" | "ethereum" => "ethereum",
        "sol" | "solana" => "solana",
        "doge" | "dogecoin" => "dogecoin",
        other if !other.is_empty() => other,
        _ => "bitcoin",
    };
    let body = get_text(
        "api.coingecko.com",
        &format!("/api/v3/simple/price?ids={id}&vs_currencies=usd,inr"),
        cfg,
    )?;
    // nested: "bitcoin":{"usd":...}
    let usd = number_field(&body, "usd").unwrap_or(0.0);
    let inr = number_field(&body, "inr").unwrap_or(0.0);
    if usd == 0.0 && inr == 0.0 {
        return Err(Error::Tool(format!("price not found for {id}")));
    }
    Ok(ToolOutput {
        name: "crypto",
        body: format!("│ {id}\n│ USD {usd:.2} · INR {inr:.2}\n│ source: CoinGecko"),
    })
}

pub fn github_repo(spec: &str, cfg: &Config) -> Result<ToolOutput> {
    let s = spec
        .replace("github.com/", "")
        .replace("https://", "")
        .replace("http://", "")
        .replace("github ", "")
        .trim()
        .trim_matches('/')
        .to_string();
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return Err(Error::Tool("usage: github owner/repo".into()));
    }
    let owner = parts[0];
    let repo = parts[1].trim_end_matches(".git");
    let body = get_text("api.github.com", &format!("/repos/{owner}/{repo}"), cfg)?;
    let full = string_field(&body, "full_name").unwrap_or_else(|| format!("{owner}/{repo}"));
    let desc = string_field(&body, "description").unwrap_or_else(|| "—".into());
    let stars = number_field(&body, "stargazers_count").unwrap_or(0.0);
    let lang = string_field(&body, "language").unwrap_or_else(|| "—".into());
    let desc_short: String = desc.chars().take(160).collect();
    Ok(ToolOutput {
        name: "github",
        body: format!(
            "│ {full}\n│ ★ {stars:.0} · {lang}\n│ {desc_short}\n│ source: GitHub API"
        ),
    })
}

pub fn headlines(cfg: &Config) -> Result<ToolOutput> {
    let body = get_text("feeds.bbci.co.uk", "/news/rss.xml", cfg)?;
    let mut titles: Vec<String> = Vec::new();
    let mut rest = body.as_str();
    while titles.len() < 5 {
        if let Some(i) = rest.find("<title>") {
            rest = &rest[i + 7..];
            if let Some(j) = rest.find("</title>") {
                let t = rest[..j].trim();
                rest = &rest[j + 8..];
                // skip channel title
                if t.to_lowercase().contains("bbc") && titles.is_empty() {
                    continue;
                }
                if !t.is_empty() {
                    titles.push(t.chars().take(100).collect());
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if titles.is_empty() {
        return Err(Error::Tool("no headlines parsed".into()));
    }
    let mut body_out = String::from("│ BBC News (RSS)\n");
    for (i, t) in titles.iter().enumerate() {
        body_out.push_str(&format!("│ {}. {}\n", i + 1, t));
    }
    body_out.push_str("│ source: feeds.bbci.co.uk");
    Ok(ToolOutput {
        name: "headlines",
        body: body_out,
    })
}

pub fn time_in_place(place: &str, cfg: &Config) -> Result<ToolOutput> {
    let q = place.trim();
    if q.is_empty() {
        return time_now(cfg);
    }
    // worldtimeapi timezone search is limited; use area heuristic + Etc/UTC fallback
    let tz = guess_timezone(q);
    match get_text("worldtimeapi.org", &format!("/api/timezone/{tz}"), cfg) {
        Ok(body) => {
            let dt = string_field(&body, "datetime").unwrap_or_else(|| "unknown".into());
            let abbr = string_field(&body, "abbreviation").unwrap_or_default();
            Ok(ToolOutput {
                name: "time",
                body: format!("│ {q} ({tz})\n│ {dt} {abbr}"),
            })
        }
        Err(_) => time_now(cfg),
    }
}

fn guess_timezone(place: &str) -> String {
    let p = place.to_lowercase();
    let map = [
        ("tokyo", "Asia/Tokyo"),
        ("delhi", "Asia/Kolkata"),
        ("mumbai", "Asia/Kolkata"),
        ("kolkata", "Asia/Kolkata"),
        ("london", "Europe/London"),
        ("paris", "Europe/Paris"),
        ("new york", "America/New_York"),
        ("nyc", "America/New_York"),
        ("sf", "America/Los_Angeles"),
        ("san francisco", "America/Los_Angeles"),
        ("dubai", "Asia/Dubai"),
        ("singapore", "Asia/Singapore"),
        ("sydney", "Australia/Sydney"),
        ("utc", "Etc/UTC"),
    ];
    for (k, v) in map {
        if p.contains(k) {
            return v.to_string();
        }
    }
    "Etc/UTC".into()
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
    fn units_km_mi() {
        let o = units("10 km to mi").unwrap();
        assert!(o.body.contains("mi"));
    }

    #[test]
    fn calc_basic() {
        let o = calculator("25*4").unwrap();
        assert!(o.body.contains("100"));
    }
}
