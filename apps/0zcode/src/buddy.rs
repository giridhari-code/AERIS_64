//! Buddy — local ASCII companion for 0zcode (no coding help).
//!
//! Parity with AERIS `sentinel.buddy` spirit: deterministic hatch, egg
//! incubation on chat steps, pet, status. State is local to the terminal
//! user — not a remote demo.
//!
//! Commands:
//!   /buddy          show companion (or egg progress)
//!   /buddy egg      incubation only
//!   /buddy pet      affection + frame
//!   /buddy status   short line for status bars
//!
//! Env:
//!   ZEROZ_BUDDY_ID     stable hatch seed (default: username@hostname)
//!   ZEROZ_BUDDY_FILE   state path (default ~/.local/state/0zcode/buddy.json)
//!   ZEROZ_BUDDY=0      disable

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct Species {
    id: &'static str,
    name: &'static str,
    lines: &'static [&'static str],
}

const SPECIES: &[Species] = &[
    Species {
        id: "cat",
        name: "Cat",
        lines: &["  /\\_/\\  ", " ( o.o ) ", "  > ^ <  "],
    },
    Species {
        id: "owl",
        name: "Owl",
        lines: &["  /\\_/\\  ", " ( o o ) ", "  > ^ <  "],
    },
    Species {
        id: "capybara",
        name: "Capybara",
        lines: &["  (---)  ", " ( o o ) ", "  \\___/  "],
    },
    Species {
        id: "duck",
        name: "Duck",
        lines: &["  __(o)> ", " (  =  ) ", "  ^^ ^^  "],
    },
    Species {
        id: "dragon",
        name: "Dragon",
        lines: &["  /^_^\\  ", " ( o o ) ", " /|   |\\ "],
    },
    Species {
        id: "blobbit",
        name: "Blobbit",
        lines: &["  .---.  ", " ( o o ) ", "  '---'  "],
    },
    Species {
        id: "fox",
        name: "Fox",
        lines: &["  /\\_/\\  ", " ( ^.^ ) ", "  \" \" \"  "],
    },
    Species {
        id: "panda",
        name: "Panda",
        lines: &["  (\\\"_\"/) ", "  ( -.- ) ", "  o_(\")(\") "],
    },
];

const NAMES_A: &[&str] = &[
    "Pip", "Mochi", "Nix", "Luma", "Zig", "Bop", "Kori", "Sol", "Pebble", "Mist", "Ember",
    "Nova", "Tiko", "Wisp", "Juno", "Ora",
];
const NAMES_B: &[&str] = &["bean", "puff", "spark", "dot", "glow", "hop", "soft", "blink"];
const PERSONALITY: &[&str] = &[
    "quiet watcher through long sessions",
    "curious; tilts head at hard questions",
    "sleepy but loyal on night shifts",
    "cheerful when a tool returns ok",
    "stoic; never judges a rough draft",
    "playful bounce on every /pet",
    "calm presence during tough ops",
    "dreamy; stares at the cursor like stars",
];

fn buddy_enabled() -> bool {
    !env::var("ZEROZ_BUDDY")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

fn seed_id() -> String {
    if let Ok(s) = env::var("ZEROZ_BUDDY_ID") {
        if !s.trim().is_empty() {
            return s.trim().to_string();
        }
    }
    let user = env::var("USER").unwrap_or_else(|_| "operator".into());
    let host = env::var("HOSTNAME")
        .or_else(|_| env::var("HOST"))
        .unwrap_or_else(|_| "local".into());
    format!("{user}@{host}")
}

fn state_path() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_BUDDY_FILE") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/state/0zcode/buddy.json")
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn digest(seed: &str) -> u64 {
    fnv1a64(format!("0zcode-buddy-v1:{seed}").as_bytes())
}

#[derive(Clone)]
struct State {
    seed: String,
    steps: u32,
    pets: u32,
    required: u32,
    last_step_ms: u64,
}

impl State {
    fn default_for(seed: &str) -> Self {
        let d = digest(seed);
        // 6–10 steps, deterministic
        let required = 6 + ((d >> 8) % 5) as u32;
        Self {
            seed: seed.to_string(),
            steps: 0,
            pets: 0,
            required,
            last_step_ms: 0,
        }
    }
}

fn load_state() -> State {
    let seed = seed_id();
    let path = state_path();
    let Ok(text) = fs::read_to_string(&path) else {
        return State::default_for(&seed);
    };
    let mut st = State::default_for(&seed);
    if let Some(v) = json_u32(&text, "steps") {
        st.steps = v;
    }
    if let Some(v) = json_u32(&text, "pets") {
        st.pets = v;
    }
    if let Some(v) = json_u32(&text, "required") {
        st.required = v.clamp(6, 12);
    }
    if let Some(v) = json_u64(&text, "last_step_ms") {
        st.last_step_ms = v;
    }
    // if seed changed, keep progress but re-bind seed
    st.seed = seed;
    st
}

fn save_state(st: &State) {
    let path = state_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let body = format!(
        "{{\"seed\":{},\"steps\":{},\"pets\":{},\"required\":{},\"last_step_ms\":{}}}\n",
        crate::http::json_string(&st.seed),
        st.steps,
        st.pets,
        st.required,
        st.last_step_ms
    );
    let _ = fs::write(path, body);
}

fn json_u32(s: &str, key: &str) -> Option<u32> {
    json_u64(s, key).map(|v| v as u32)
}

fn json_u64(s: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\"");
    let i = s.find(&pat)?;
    let rest = s[i + pat.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let num: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().ok()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct Hatch {
    species: &'static Species,
    name: String,
    rarity: &'static str,
    shiny: bool,
    personality: &'static str,
}

fn hatch_from(seed: &str) -> Hatch {
    let d = digest(seed);
    let sp = &SPECIES[(d as usize) % SPECIES.len()];
    let rarity = match (d >> 16) % 100 {
        0..=59 => "common",
        60..=84 => "uncommon",
        85..=94 => "rare",
        95..=98 => "epic",
        _ => "legendary",
    };
    let shiny = ((d >> 24) & 0xff) < 8;
    let na = NAMES_A[(d as usize >> 4) % NAMES_A.len()];
    let nb = NAMES_B[(d as usize >> 12) % NAMES_B.len()];
    let personality = PERSONALITY[(d as usize >> 20) % PERSONALITY.len()];
    Hatch {
        species: sp,
        name: format!("{na}-{nb}"),
        rarity,
        shiny,
        personality,
    }
}

fn render_sprite(h: &Hatch, hearts: bool) -> String {
    let mut out = String::new();
    if hearts {
        out.push_str("  ♡  ♡  ♡\n");
    }
    if h.shiny {
        out.push_str("  ✦ shiny\n");
    }
    for ln in h.species.lines {
        out.push_str(ln);
        out.push('\n');
    }
    out
}

fn egg_art(progress: f32) -> String {
    let blink = (now_ms() / 800) % 2 == 0;
    let eyes = if blink { "-   -" } else { "o   o" };
    format!(
        "   .-~~~-. \n  / {eyes} \\\n :  ~~~~~ :\n  `-.___.-'\n  [{:>3}%]\n",
        (progress * 100.0).round() as u32
    )
}

/// Count one incubation step (chat turn). Call from agent path.
pub fn on_user_message() {
    if !buddy_enabled() {
        return;
    }
    let mut st = load_state();
    if st.steps >= st.required {
        return;
    }
    st.steps = st.steps.saturating_add(1);
    st.last_step_ms = now_ms();
    save_state(&st);
}

fn status_block() -> String {
    let st = load_state();
    let h = hatch_from(&st.seed);
    if st.steps < st.required {
        let p = st.steps as f32 / st.required as f32;
        return format!(
            "Buddy egg — hatching soon\n{}\nsteps {}/{} · seed {}\n\
When the egg hatches you will meet a companion (never writes code).\n\
/buddy pet after hatch · /buddy egg for progress only",
            egg_art(p),
            st.steps,
            st.required,
            st.seed
        );
    }
    format!(
        "Buddy · {} the {} ({}){}\n{}\n{}\n\
pets: {} · /buddy pet\n\
(decorative companion — tools & policy still rule)",
        h.name,
        h.species.name,
        h.rarity,
        if h.shiny { " · shiny" } else { "" },
        render_sprite(&h, false),
        h.personality,
        st.pets
    )
}

fn pet() -> String {
    let mut st = load_state();
    if st.steps < st.required {
        return format!(
            "Egg is still incubating ({}/{}).\nKeep chatting — then /buddy pet.",
            st.steps, st.required
        );
    }
    st.pets = st.pets.saturating_add(1);
    save_state(&st);
    let h = hatch_from(&st.seed);
    format!(
        "{} says thanks\n{}\n♥ pet #{} · still no code, only company",
        h.name,
        render_sprite(&h, true),
        st.pets
    )
}

fn egg_only() -> String {
    let st = load_state();
    let p = (st.steps as f32 / st.required.max(1) as f32).min(1.0);
    if st.steps >= st.required {
        return "Egg already hatched — /buddy to meet them.".into();
    }
    format!(
        "Incubation\n{}\n{}/{} message-steps (deterministic goal)",
        egg_art(p),
        st.steps,
        st.required
    )
}

fn one_liner() -> String {
    let st = load_state();
    if st.steps < st.required {
        return format!("egg {}/{}", st.steps, st.required);
    }
    let h = hatch_from(&st.seed);
    format!("{} the {}", h.name, h.species.name)
}

/// Slash / natural buddy commands.
pub fn handle_command(line: &str) -> Option<String> {
    if !buddy_enabled() {
        let lower = line.trim().to_lowercase();
        if lower.starts_with("/buddy") || lower == "buddy" {
            return Some("Buddy is disabled (ZEROZ_BUDDY=0).".into());
        }
        return None;
    }
    let lower = line.trim().to_lowercase();
    if lower == "/buddy"
        || lower == "buddy"
        || lower == "/buddy show"
        || lower == "/buddy status"
    {
        return Some(status_block());
    }
    if lower == "/buddy egg" || lower == "buddy egg" {
        return Some(egg_only());
    }
    if lower == "/buddy pet" || lower == "buddy pet" || lower == "/pet" {
        return Some(pet());
    }
    if lower == "/buddy help" || lower.starts_with("/buddy ") {
        return Some(
            "Buddy — local companion (0zcode)\n\
  /buddy         show egg or companion\n\
  /buddy egg     incubation progress\n\
  /buddy pet     affection\n\
  /buddy status  same as /buddy\n\
Env: ZEROZ_BUDDY_ID · ZEROZ_BUDDY_FILE · ZEROZ_BUDDY=0\n\
Does not write or run code."
                .into(),
        );
    }
    None
}

/// For status footer / help lines.
pub fn status_chip() -> String {
    if !buddy_enabled() {
        return String::new();
    }
    one_liner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn hatch_is_stable() {
        let a = hatch_from("operator@lab");
        let b = hatch_from("operator@lab");
        assert_eq!(a.name, b.name);
        assert_eq!(a.species.id, b.species.id);
    }

    #[test]
    fn egg_then_hatch_flow() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join("0zcode_buddy_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        env::set_var("ZEROZ_BUDDY_FILE", dir.join("b.json"));
        env::set_var("ZEROZ_BUDDY_ID", "test-user-1");
        env::remove_var("ZEROZ_BUDDY");
        let mut st = load_state();
        assert!(st.steps < st.required);
        let need = st.required;
        for _ in 0..need {
            on_user_message();
        }
        st = load_state();
        assert!(st.steps >= st.required);
        let text = status_block();
        assert!(text.contains("Buddy ·") || text.contains("the"));
        let p = pet();
        assert!(p.contains("pet #"));
        env::remove_var("ZEROZ_BUDDY_FILE");
        env::remove_var("ZEROZ_BUDDY_ID");
        let _ = fs::remove_dir_all(&dir);
    }
}
