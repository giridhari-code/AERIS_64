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
                out.push(match n {
                    'n' => '\n',
                    't' => '\t',
                    '"' => '"',
                    '\\' => '\\',
                    'u' => {
                        // skip simple \uXXXX
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                            char::from_u32(cp).unwrap_or('?')
                        } else {
                            '?'
                        }
                    }
                    other => other,
                });
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
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
    fn parses_number() {
        let j = r#"{"temperature_2m":28.5}"#;
        assert!((number_field(j, "temperature_2m").unwrap() - 28.5).abs() < 1e-6);
    }
}
