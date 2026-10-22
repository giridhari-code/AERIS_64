//! Lightweight language detection for reply framing (no ML).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Hi,
    Hinglish,
}

/// Heuristic: Devanagari → Hi; common Hinglish tokens → Hinglish; else En.
pub fn detect(text: &str) -> Lang {
    if text.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c)) {
        return Lang::Hi;
    }
    let l = text.to_lowercase();
    const TOKENS: &[&str] = &[
        "kya", "hai", "nahi", "nahin", "mausam", "batao", "karo", "kitna", "kaise", "acha",
        "theek", "bhai", "yaar", "tarikh", "samay", "kripya",
    ];
    if TOKENS.iter().any(|t| l.split_whitespace().any(|w| w == *t) || l.contains(t)) {
        // avoid false positive on pure English containing "hai" as substring in other words — token split helps
        if TOKENS.iter().any(|t| l.split(|c: char| !c.is_alphabetic()).any(|w| w == *t)) {
            return Lang::Hinglish;
        }
    }
    Lang::En
}

pub fn progress_label(lang: Lang, tool_hint: &str) -> String {
    match lang {
        Lang::Hi | Lang::Hinglish => format!("… {tool_hint} laa raha hoon"),
        Lang::En => format!("… fetching {tool_hint}"),
    }
}

pub fn empty_match_hint(lang: Lang) -> &'static str {
    match lang {
        Lang::Hi | Lang::Hinglish => {
            "Tool match nahi mila — invent nahi karunga. Try: weather in Mumbai · time · calculate 2+2 · /help"
        }
        Lang::En => {
            "I don't invent answers without a tool match. Try: weather in Mumbai · time · calculate 2+2 · /help"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_hinglish() {
        assert_eq!(detect("aaj mausam kaisa hai"), Lang::Hinglish);
    }

    #[test]
    fn detects_en() {
        assert_eq!(detect("weather in Delhi"), Lang::En);
    }
}
