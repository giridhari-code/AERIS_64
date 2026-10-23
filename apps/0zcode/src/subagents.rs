//! Subagents — specialized task workers (OpenCode-style), company-standard.
//!
//! Dispatch via `/agent <kind>: <goal>` or harness auto-picks from goal keywords.
//! Each subagent returns a fixed TOOL plan; the harness executes under the same
//! workspace / write / shell gates.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Explore,
    Edit,
    Test,
    Search,
    Research,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Explore => "explore",
            Kind::Edit => "edit",
            Kind::Test => "test",
            Kind::Search => "search",
            Kind::Research => "research",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "explore" | "scout" => Some(Kind::Explore),
            "edit" | "writer" | "patch" => Some(Kind::Edit),
            "test" | "qa" | "ci" => Some(Kind::Test),
            "search" | "grep" | "find" => Some(Kind::Search),
            "research" | "web" | "docs" => Some(Kind::Research),
            _ => None,
        }
    }
}

/// Infer subagent from free-form goal when user did not pin a kind.
pub fn infer(goal: &str) -> Kind {
    let g = goal.to_lowercase();
    if let Some(rest) = g.strip_prefix('[') {
        if let Some((k, _)) = rest.split_once(']') {
            if let Some(kind) = Kind::parse(k.trim()) {
                return kind;
            }
        }
    }
    if g.contains("test") || g.contains("cargo test") || g.contains("ci") {
        return Kind::Test;
    }
    if g.contains("edit")
        || g.contains("replace")
        || g.contains("write ")
        || g.contains("create file")
        || g.contains("patch")
    {
        return Kind::Edit;
    }
    if g.contains("grep") || g.contains("search for") || g.contains("find ") {
        return Kind::Search;
    }
    if g.contains("weather") || g.contains("news") || g.contains("wiki") || g.contains("research")
    {
        return Kind::Research;
    }
    Kind::Explore
}

/// Plan tool steps for this subagent + goal.
pub fn plan(kind: Kind, goal: &str) -> Vec<String> {
    let g = goal.to_lowercase();
    match kind {
        Kind::Explore => vec![
            "TOOL list .".into(),
            "TOOL read README.md".into(),
            "TOOL read Cargo.toml".into(),
            "TOOL grep TODO".into(),
            "DONE explore complete".into(),
        ],
        Kind::Search => {
            let pat = extract_pat(goal).unwrap_or_else(|| "TODO".into());
            vec![
                "TOOL list .".into(),
                format!("TOOL grep {pat}"),
                format!("DONE search `{pat}`"),
            ]
        }
        Kind::Test => vec![
            "TOOL list .".into(),
            "TOOL shell cargo --version".into(),
            "TOOL shell cargo test".into(),
            "DONE tests finished".into(),
        ],
        Kind::Edit => {
            if let Some((path, old, newv)) = extract_replace(goal) {
                vec![
                    format!("TOOL read {path}"),
                    format!("TOOL edit {path} | {old} | {newv}"),
                    format!("TOOL read {path}"),
                    format!("DONE edited {path}"),
                ]
            } else if let Some(name) = extract_write_path(goal) {
                let body = extract_body(goal)
                    .unwrap_or_else(|| format!("# created by edit subagent\n# goal: {goal}\n"));
                vec![
                    "TOOL list .".into(),
                    format!("TOOL write {name} | {}", body.replace('\n', "\\n")),
                    format!("TOOL read {name}"),
                    format!("DONE wrote {name}"),
                ]
            } else {
                vec![
                    "TOOL list .".into(),
                    "DONE edit: specify write path | body  or  replace path old -> new".into(),
                ]
            }
        }
        Kind::Research => {
            vec![
                "TOOL list .".into(),
                format!(
                    "DONE research goal logged: {}",
                    g.chars().take(80).collect::<String>()
                ),
            ]
        }
    }
}

fn extract_pat(goal: &str) -> Option<String> {
    let lower = goal.to_lowercase();
    for key in ["grep ", "find ", "search for ", "search "] {
        if let Some(i) = lower.find(key) {
            let rest = goal[i + key.len()..].trim();
            let tok = rest.split_whitespace().next()?;
            return Some(tok.trim_matches('"').to_string());
        }
    }
    None
}

fn extract_write_path(goal: &str) -> Option<String> {
    let lower = goal.to_lowercase();
    for key in ["write ", "create file ", "add file "] {
        if let Some(i) = lower.find(key) {
            let rest = goal[i + key.len()..].trim();
            let name = rest.split_whitespace().next()?;
            if name.contains('.') || name.contains('/') {
                return Some(name.trim_matches('"').to_string());
            }
        }
    }
    None
}

fn extract_body(goal: &str) -> Option<String> {
    if let Some(i) = goal.find('|') {
        return Some(goal[i + 1..].trim().to_string());
    }
    if let Some(i) = goal.find(':') {
        let b = goal[i + 1..].trim();
        if !b.is_empty() {
            return Some(b.to_string());
        }
    }
    None
}

fn extract_replace(goal: &str) -> Option<(String, String, String)> {
    if !goal.contains("->") {
        return None;
    }
    let path = goal
        .split_whitespace()
        .find(|t| t.contains('.') || t.contains('/'))?
        .trim_matches('"')
        .to_string();
    let (left, rhs) = goal.split_once("->")?;
    let old = left
        .split_whitespace()
        .rev()
        .find(|t| *t != path.as_str() && !t.eq_ignore_ascii_case("replace"))?
        .to_string();
    let newv = rhs.trim().to_string();
    if old.is_empty() || newv.is_empty() {
        return None;
    }
    Some((path, old, newv))
}

pub fn catalog() -> &'static str {
    "subagents: explore · edit · test · search · research\n\
     usage: /agent explore: map the repo\n\
            /agent test: run cargo test\n\
            /agent search: grep TODO\n\
            /agent edit: write notes/x.md | hello\n\
     gates: ZEROZ_AGENT=1 · ZEROZ_WRITE=1 · ZEROZ_SHELL=1"
}

pub fn parse_kind_goal(msg: &str) -> Option<(Kind, String)> {
    let t = msg.trim();
    let lower = t.to_lowercase();
    let rest = if lower.starts_with("/agent ") {
        t[7..].trim()
    } else if lower.starts_with("agent: ") {
        t[7..].trim()
    } else {
        return None;
    };
    if rest.is_empty() {
        return Some((Kind::Explore, String::new()));
    }
    if let Some((k, g)) = rest.split_once(':') {
        if let Some(kind) = Kind::parse(k) {
            return Some((kind, g.trim().to_string()));
        }
    }
    let mut sp = rest.splitn(2, char::is_whitespace);
    if let Some(first) = sp.next() {
        if let Some(kind) = Kind::parse(first) {
            return Some((kind, sp.next().unwrap_or("").trim().to_string()));
        }
    }
    Some((infer(rest), rest.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_test() {
        assert_eq!(infer("run cargo test please"), Kind::Test);
    }

    #[test]
    fn parse_kind() {
        let (k, g) = parse_kind_goal("/agent test: run suite").unwrap();
        assert_eq!(k, Kind::Test);
        assert!(g.contains("suite"));
    }
}
