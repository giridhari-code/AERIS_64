//! 0zcode — company-standard terminal agent binary.

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use zeroz::agent::handle;
use zeroz::config::Config;
use zeroz::history::History;
use zeroz::policy::refusal_message;
use zeroz::tools;
use zeroz::tui::Theme;
use zeroz::VERSION;

fn main() -> ExitCode {
    let cfg = Config::from_env();
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("0zcode {VERSION}");
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("tools")
        || args.iter().any(|a| a == "--tools")
    {
        println!("{}", tools::catalog());
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("policy")
        || args.iter().any(|a| a == "--policy")
    {
        println!("{}", refusal_message());
        return ExitCode::SUCCESS;
    }

    if !args.is_empty() {
        let msg = args.join(" ");
        let r = handle(&msg, &cfg);
        println!("{}", r.text);
        return if r.refused {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }

    run_repl(&cfg)
}

fn print_help() {
    println!("0zcode {VERSION} — Agentic work. Zero code.");
    println!(
        "\nUsage:\n  0zcode                 interactive TUI\n  0zcode <message>       one-shot\n  0zcode tools|policy    meta\n  0zcode --version\n\nEnv / config:\n  ~/.config/0zcode/config.toml  (or ZEROZ_CONFIG)\n  ZEROZ_OFFLINE=1               calc-only\n  ZEROZ_HTTP_TIMEOUT_SECS\n  ZEROZ_MODEL_LABEL\n  ZEROZ_NO_COLOR / NO_COLOR\n"
    );
}

fn run_repl(cfg: &Config) -> ExitCode {
    let theme = Theme::from_config(cfg);
    theme.clear_screen();
    theme.print_header();
    if cfg.offline {
        theme.print_assistant("Offline mode: calculator + commands only.");
        println!();
    }

    let model_label = cfg.model_label.clone();
    let mut history = History::new(cfg.history_size);

    let stdin = io::stdin();
    loop {
        print!("{}", theme.prompt());
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let lower = line.to_lowercase();
        match lower.as_str() {
            "/help" | "?" | "help" => {
                theme.print_help_panel();
                continue;
            }
            "/history" | "history" => {
                theme.print_assistant(&history.render());
                println!();
                continue;
            }
            "/status" | "status" => {
                theme.print_status(&model_label, history.len() as u32, if cfg.offline { "offline" } else { "agent (no code)" });
                theme.print_assistant(&format!("config: {}", Config::config_path_display()));
                println!();
                continue;
            }
            "/model" | "model" => {
                theme.print_assistant(&format!(
                    "Active label: {model_label}\nSet ZEROZ_MODEL_LABEL or model_label in config.toml."
                ));
                println!();
                continue;
            }
            _ => {}
        }

        theme.print_user(line);
        history.push("user", line);
        let r = handle(line, cfg);
        if r.text == "__QUIT__" {
            println!("bye.");
            break;
        }
        if r.text == "__CLEAR__" {
            history.clear();
            theme.clear_screen();
            theme.print_header();
            continue;
        }
        history.push("assistant", &r.text);
        if r.refused {
            theme.print_err(&r.text);
        } else if !r.tool_names.is_empty() {
            let name = r.tool_names.join("+");
            theme.print_tool_box(&name, &r.text);
        } else {
            theme.print_assistant(&r.text);
        }
        println!();
    }
    ExitCode::SUCCESS
}
