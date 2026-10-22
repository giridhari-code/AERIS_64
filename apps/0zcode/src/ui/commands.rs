//! Slash command registry + palette filter.

#[derive(Debug, Clone, Copy)]
pub struct CommandSpec {
    pub name: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/help",
        description: "show commands",
    },
    CommandSpec {
        name: "/clear",
        description: "clear transcript",
    },
    CommandSpec {
        name: "/model",
        description: "set display model label",
    },
    CommandSpec {
        name: "/tools",
        description: "list built-in tools",
    },
    CommandSpec {
        name: "/mcp",
        description: "user MCP tools",
    },
    CommandSpec {
        name: "/status",
        description: "session status",
    },
    CommandSpec {
        name: "/compact",
        description: "trim older transcript lines",
    },
    CommandSpec {
        name: "/resume",
        description: "load last session",
    },
    CommandSpec {
        name: "/config",
        description: "show config path",
    },
    CommandSpec {
        name: "/quit",
        description: "exit",
    },
    CommandSpec {
        name: "/exit",
        description: "exit",
    },
];

/// Prefix match for palette (input starts with `/`).
pub fn filter_commands(prefix: &str) -> Vec<&'static CommandSpec> {
    let p = prefix.trim().to_lowercase();
    if p.is_empty() || p == "/" {
        return COMMANDS.iter().collect();
    }
    COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(&p) || c.name[1..].starts_with(p.trim_start_matches('/')))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_help() {
        let v = filter_commands("/he");
        assert!(v.iter().any(|c| c.name == "/help"));
    }
}
