//! MCP-style tool registry — descriptors for clients / future MCP bridge.
//! 0zcode stays std-only; this is the allow-list contract.

use crate::config::Config;

#[derive(Debug, Clone)]
pub struct ToolDesc {
    pub name: &'static str,
    pub description: &'static str,
    pub input_hint: &'static str,
    pub network: bool,
    pub offline_ok: bool,
}

pub fn all_tools() -> &'static [ToolDesc] {
    &[
        ToolDesc {
            name: "read",
            description: "Read a file (optional start line + count)",
            input_hint: "read src/main.rs 1 80",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "list",
            description: "List a folder in the workspace",
            input_hint: "list src",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "glob",
            description: "Find files by name pattern",
            input_hint: "glob src/**/*.rs",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "grep",
            description: "Search file contents",
            input_hint: "grep TODO src",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "edit",
            description: "Replace exact unique text in a file (asks approval)",
            input_hint: "/code fix the typo in README.md",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "write",
            description: "Create or overwrite a file (asks approval)",
            input_hint: "/code add tests/smoke.rs",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "shell",
            description: "Allowlisted cargo / git commands (asks approval)",
            input_hint: "shell cargo test",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "web_fetch",
            description: "Fetch a web page as text",
            input_hint: "fetch https://example.com",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "web_search",
            description: "Web instant answers",
            input_hint: "search rust std fs read_dir",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "calculator",
            description: "Arithmetic + - * /",
            input_hint: "2+2",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "notes",
            description: "Read/write short notes in 0zcode sandbox only",
            input_hint: "note save buy milk · note list · note read buy",
            network: false,
            offline_ok: true,
        },
    ]
}

/// JSON-ish list for MCP / IDE bridges (hand-rolled, no serde).
pub fn tools_json(_cfg: &Config) -> String {
    let mut out = String::from("{\n  \"tools\": [\n");
    for (i, t) in all_tools().iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"description\": \"{}\", \"input_hint\": \"{}\", \"network\": {}, \"offline_ok\": {}}}",
            t.name,
            t.description.replace('"', "'"),
            t.input_hint.replace('"', "'"),
            t.network,
            t.offline_ok
        ));
    }
    out.push_str("\n  ],\n");
    out.push_str("  \"policy\": {\n");
    out.push_str("    \"shell\": \"allowlist + approval (ZEROZ_SHELL=1)\",\n");
    out.push_str("    \"code_generation\": \"opt-in (ZEROZ_CODING=1)\",\n");
    out.push_str("    \"file_write\": \"workspace only + approval (ZEROZ_WRITE=1)\"\n");
    out.push_str("  }\n}\n");
    out
}

pub fn tools_markdown() -> String {
    let mut out = String::from("Allow-listed tools (MCP-style registry)\n\n");
    for t in all_tools() {
        out.push_str(&format!(
            "- **{}** — {} _(e.g. `{}`)_ net={} offline_ok={}\n",
            t.name, t.description, t.input_hint, t.network, t.offline_ok
        ));
    }
    out.push_str("\nGated: shell (allowlist + approval), file write (workspace + approval), coding loop (ZEROZ_CODING=1).\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_nonempty() {
        assert!(all_tools().len() >= 8);
        assert!(tools_json(&Config::default()).contains("\"glob\""));
    }
}
