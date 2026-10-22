//! `@` fuzzy file path completion (cwd-relative, no external crate).

use std::fs;
use std::path::{Path, PathBuf};

/// Extract the path token after the last `@` in the input.
pub fn at_token(input: &str) -> Option<&str> {
    let i = input.rfind('@')?;
    let rest = &input[i + 1..];
    if rest.contains(' ') {
        return None;
    }
    Some(rest)
}

/// Ranked path matches under cwd (max 8).
pub fn fuzzy_paths(cwd: &Path, partial: &str) -> Vec<String> {
    let partial = partial.trim();
    let (dir, file_prefix) = if partial.contains('/') {
        let p = Path::new(partial);
        let parent = p.parent().unwrap_or(Path::new("."));
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        (cwd.join(parent), name)
    } else {
        (cwd.to_path_buf(), partial.to_string())
    };
    let dir = if dir.is_dir() { dir } else { cwd.to_path_buf() };
    let mut out = Vec::new();
    let rd = match fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return out,
    };
    let prefix_l = file_prefix.to_lowercase();
    for ent in rd.flatten() {
        let name = ent.file_name().to_string_lossy().to_string();
        if name.starts_with('.') && !prefix_l.starts_with('.') {
            continue;
        }
        let name_l = name.to_lowercase();
        if prefix_l.is_empty() || name_l.starts_with(&prefix_l) || name_l.contains(&prefix_l) {
            let is_dir = ent.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let rel = if partial.contains('/') {
                let parent = Path::new(partial).parent().unwrap_or(Path::new(""));
                PathBuf::from(parent).join(&name)
            } else {
                PathBuf::from(&name)
            };
            let mut s = rel.to_string_lossy().replace('\\', "/");
            if is_dir && !s.ends_with('/') {
                s.push('/');
            }
            out.push(s);
        }
        if out.len() >= 24 {
            break;
        }
    }
    out.sort();
    out.truncate(8);
    out
}

/// Replace `@token` with `@chosen` (or just path without @ for insert).
pub fn apply_path_completion(input: &str, chosen: &str) -> String {
    if let Some(i) = input.rfind('@') {
        let mut s = input[..i].to_string();
        s.push_str(chosen);
        s
    } else {
        chosen.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_parse() {
        assert_eq!(at_token("see @src/"), Some("src/"));
        assert_eq!(at_token("nope"), None);
    }
}
