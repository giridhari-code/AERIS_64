//! Which user lines need network permission.

pub fn needs_network(line: &str) -> Option<&'static str> {
    let l = line.to_lowercase();
    if l.starts_with('/') {
        return None;
    }
    if l.contains("weather") {
        return Some("weather");
    }
    if l.contains("wiki") || l.contains("wikipedia") {
        return Some("wikipedia");
    }
    if l.contains("search") || l.contains("headline") || l.contains("news") {
        return Some("search");
    }
    if l.contains("btc") || l.contains("crypto") || l.contains("price") {
        return Some("crypto");
    }
    if l.contains("github.com") || l.starts_with("github ") {
        return Some("github");
    }
    if l.contains(" usd") || l.contains(" inr") || l.contains(" to inr") || l.contains(" to usd") {
        return Some("currency");
    }
    if l.contains("http://") || l.contains("https://") {
        return Some("fetch");
    }
    // time may use network optionally — treat local-first; no gate
    None
}
