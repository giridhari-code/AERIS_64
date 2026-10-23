//! Tiny JSON field extractors (no serde — offline builds).

/// Extract a JSON string value for `"key": "..."`.
pub fn string_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let i = body.find(&pat)?;
    let mut rest = body[i + pat.len()..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    rest = &rest[1..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                match n {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hi = read_hex4(&mut chars);
                        match hi {
                            // UTF-16 surrogate pair (emoji etc.): \ud83d\ude00
                            Some(h) if (0xD800..0xDC00).contains(&h) => {
                                let mut ahead = chars.clone();
                                let lo = (ahead.next() == Some('\\') && ahead.next() == Some('u'))
                                    .then(|| read_hex4(&mut ahead))
                                    .flatten()
                                    .filter(|l| (0xDC00..0xE000).contains(l));
                                if let Some(l) = lo {
                                    chars = ahead;
                                    let cp = 0x10000 + ((h - 0xD800) << 10) + (l - 0xDC00);
                                    out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                                } else {
                                    out.push('\u{fffd}');
                                }
                            }
                            Some(h) => out.push(char::from_u32(h).unwrap_or('\u{fffd}')),
                            None => out.push('\u{fffd}'),
                        }
                    }
                    other => out.push(other), // \" \\ \/
                }
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn read_hex4(chars: &mut std::str::Chars<'_>) -> Option<u32> {
    let hex: String = chars.by_ref().take(4).collect();
    if hex.len() == 4 {
        u32::from_str_radix(&hex, 16).ok()
    } else {
        None
    }
}

/// Extract a JSON number for `"key": 1.23`.
pub fn number_field(body: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    // prefer last occurrence for nested "current" blocks when key is unique enough
    let i = body.rfind(&pat)?;
    let rest = body[i + pat.len()..].trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    num.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string() {
        let j = r#"{"name":"Mumbai","x":1}"#;
        assert_eq!(string_field(j, "name").as_deref(), Some("Mumbai"));
    }

    #[test]
    fn decodes_escapes_and_surrogates() {
        let j = r#"{"t":"a\r\nb \u0928\u092e \ud83d\ude00 \"q\" \/"}"#;
        assert_eq!(string_field(j, "t").as_deref(), Some("a\r\nb नम 😀 \"q\" /"));
        // lone surrogate must not panic
        assert!(string_field(r#"{"t":"\ud83d x"}"#, "t").is_some());
    }

    #[test]
    fn parses_number() {
        let j = r#"{"temperature_2m":28.5}"#;
        assert!((number_field(j, "temperature_2m").unwrap() - 28.5).abs() < 1e-6);
    }
}
