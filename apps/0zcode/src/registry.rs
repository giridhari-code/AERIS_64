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
            name: "weather",
            description: "Live weather for a place",
            input_hint: "weather in Mumbai",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "time",
            description: "UTC or city local time",
            input_hint: "time in Tokyo",
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
            name: "units",
            description: "Length / mass / temperature convert",
            input_hint: "10 km to mi",
            network: false,
            offline_ok: true,
        },
        ToolDesc {
            name: "wikipedia",
            description: "Short encyclopedia summary",
            input_hint: "who is Ada Lovelace",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "web_search",
            description: "Web instant answers",
            input_hint: "search latest AI news",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "currency",
            description: "FX conversion",
            input_hint: "10 USD to INR",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "crypto",
            description: "Crypto price USD/INR",
            input_hint: "btc price",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "github",
            description: "Public GitHub repo info",
            input_hint: "github owner/repo",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "headlines",
            description: "BBC RSS top headlines",
            input_hint: "headlines",
            network: true,
            offline_ok: false,
        },
        ToolDesc {
            name: "email_status",
            description: "Today's email summary via your configured endpoint (not Gmail password login)",
            input_hint: "today email / my email",
            network: true,
            offline_ok: false,
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
    out.push_str("    \"shell\": false,\n");
    out.push_str("    \"code_generation\": false,\n");
    out.push_str("    \"arbitrary_file_write\": false,\n");
    out.push_str("    \"notes_sandbox_only\": true\n");
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
    out.push_str("\nDenied by policy: arbitrary shell, source-code gen/refactor, unrestricted file write.\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_nonempty() {
        assert!(all_tools().len() >= 10);
        assert!(tools_json(&Config::default()).contains("email_status"));
    }
}
