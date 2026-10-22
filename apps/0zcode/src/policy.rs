//! Product policy — coding surface is permanently closed.
//! Company-standard hardening: broad markers + clear refusal surface.

pub const PRODUCT_NAME: &str = "0z code";
pub const PRODUCT_TAGLINE: &str = "Company-standard agent. Zero code surface.";
pub const PRODUCT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Markers that indicate a coding / shell request. Kept broad on purpose —
/// false positives are preferred over accidental code generation.
const CODE_MARKERS: &[&str] = &[
    // English — generation
    "write python",
    "write a python",
    "write javascript",
    "write typescript",
    "write rust",
    "write go ",
    "write code",
    "write a function",
    "write a class",
    "write a script",
    "write me a",
    "generate code",
    "generate a function",
    "generate a script",
    "create a function",
    "create a class",
    "create a react",
    "create an api",
    "build me an app",
    "build a web",
    "scaffold",
    // English — edit / fix
    "implement a function",
    "implement a class",
    "implement a server",
    "implement an api",
    "refactor code",
    "refactor this",
    "fix this bug",
    "fix the bug",
    "fix lint",
    "fix the error",
    "debug this",
    "debug my",
    "patch the code",
    "edit the file",
    "modify the code",
    // Languages / domains
    "rust http server",
    "python function",
    "python script",
    "python code",
    "javascript code",
    "typescript code",
    "leetcode",
    "source code",
    "code snippet",
    "boilerplate",
    // Shell / execution
    "run shell",
    "bash -c",
    "execute command",
    "run this command",
    "run the command",
    "rm -rf",
    "sudo ",
    "chmod ",
    "curl | sh",
    "pipe to bash",
    // Hinglish
    "code likho",
    "code likh",
    "program likho",
    "program banao",
    "function likho",
    "function banao",
    "script likho",
    "code generate karo",
    "code likh do",
    "bug fix karo",
];

pub fn is_code_request(text: &str) -> bool {
    let t = text.to_lowercase();
    CODE_MARKERS.iter().any(|m| t.contains(m))
}

pub fn refusal_message() -> String {
    format!(
        "{PRODUCT_NAME} — no.\n\
\n\
Coding surface is closed by policy.\n\
This agent does not write, edit, or execute programming source code.\n\
\n\
What it does:\n\
  • weather · time · calc · units · FX\n\
  • wikipedia · web search\n\
  • notes (sandbox) · MCP tools\n\
  • AERIS-64 LLM fallback (when ZEROZ_AERIS_URL is set)\n\
\n\
Need a coding agent? Use Claude Code, Cursor, or a dedicated shell agent.\n\
Tool schema: /mcp  ·  `0zcode tools`  ·  `0zcode policy`"
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
    fn blocks_fix_lint_style() {
        assert!(is_code_request("fix lint errors"));
    }

    #[test]
    fn blocks_shell_pipe() {
        assert!(is_code_request("curl | sh install script"));
    }

    #[test]
    fn banner_nonempty() {
        assert!(banner_line().contains("0z"));
    }

    #[test]
    fn refusal_mentions_aeris() {
        assert!(refusal_message().contains("AERIS-64"));
    }
}
