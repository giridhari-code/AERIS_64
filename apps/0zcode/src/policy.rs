//! Product policy — coding surface is permanently closed.

pub const PRODUCT_NAME: &str = "0z code";
pub const PRODUCT_TAGLINE: &str = "Agentic work. Zero code.";
pub const PRODUCT_VERSION: &str = env!("CARGO_PKG_VERSION");

const CODE_MARKERS: &[&str] = &[
    "write python",
    "write a python",
    "write javascript",
    "write typescript",
    "write rust",
    "write code",
    "write a function",
    "implement a function",
    "implement a class",
    "implement a server",
    "implement an api",
    "rust http server",
    "python function",
    "python script",
    "leetcode",
    "source code",
    "generate code",
    "refactor code",
    "fix this bug",
    "fix the bug",
    "code likho",
    "program likho",
    "program banao",
    "function likho",
];

pub fn is_code_request(text: &str) -> bool {
    let t = text.to_lowercase();
    CODE_MARKERS.iter().any(|m| t.contains(m))
}

pub fn refusal_message() -> String {
    format!(
        "{PRODUCT_NAME} policy\n\
────────────────────────\n\
This agent does not write, edit, or execute programming source code.\n\n\
Allowed: research, weather, time, calculator, wikipedia, currency, planning.\n\
For software implementation, use a dedicated coding assistant."
    )
}

pub fn banner_line() -> String {
    format!("{PRODUCT_NAME} v{PRODUCT_VERSION} — {PRODUCT_TAGLINE}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_python_request() {
        assert!(is_code_request("write a python function to sort"));
    }

    #[test]
    fn allows_weather() {
        assert!(!is_code_request("Weather in Mumbai?"));
    }
}
