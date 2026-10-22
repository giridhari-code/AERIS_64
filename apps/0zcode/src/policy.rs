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
    "create a react",
    "build me an app",
    "run shell",
    "bash -c",
    "execute command",
    "run this command",
    "rm -rf",
    "sudo ",
];

pub fn is_code_request(text: &str) -> bool {
    let t = text.to_lowercase();
    CODE_MARKERS.iter().any(|m| t.contains(m))
}

pub fn refusal_message() -> String {
    format!(
        "{PRODUCT_NAME} — straight answer: no.\n\
This agent does not write, edit, or run programming source code.\n\n\
Use it for: weather, time, calc, wikipedia, FX, web search.\n\
Need code or shell? Use a dedicated coding agent — not 0zcode.\n\nEmail: set ZEROZ_EMAIL_STATUS_URL (private summary API).\nNotes: note save/list/read (sandbox only).\nTool schema: /mcp or `0zcode tools --json`"
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

    #[test]
    fn blocks_hinglish_code() {
        assert!(is_code_request("python function likho"));
    }

    #[test]
    fn banner_nonempty() {
        assert!(banner_line().contains("0z"));
    }
}
