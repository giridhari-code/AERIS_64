//! 0zcode — company-standard terminal agent binary.

use std::env;
use std::process::ExitCode;
use zeroz::agent::handle;
use zeroz::config::Config;
use zeroz::policy::refusal_message;
use zeroz::tools;
use zeroz::VERSION;

fn main() -> ExitCode {
    let mut cfg = Config::from_env();
    let args: Vec<String> = env::args().skip(1).collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("0zcode {VERSION}");
        return ExitCode::SUCCESS;
    }
    // Global help only when not a subcommand that has its own --help
    if args.first().map(String::as_str) != Some("serve")
        && args.iter().any(|a| a == "--help" || a == "-h")
    {
        print_help();
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("tools") {
        println!("{}", tools::catalog());
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("policy") {
        println!("{}", refusal_message());
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("llm") {
        println!("{}", zeroz::llm::diagnose(&cfg));
        return ExitCode::SUCCESS;
    }

    // Rust backend: links 0zcode edge → SENTINEL core engine
    if args.first().map(String::as_str) == Some("serve") {
        let rest: Vec<String> = args.into_iter().skip(1).collect();
        return match zeroz::backend::run_serve(&rest) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("0zcode serve error: {e}");
                ExitCode::from(1)
            }
        };
    }

    // Interactive launch: open the UI at once; AERIS-64 boots in the background.
    let plain = env::var("ZEROZ_PLAIN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if args.is_empty() && !plain && zeroz::term::is_interactive() {
        zeroz::aeris_boot::ensure_aeris_background(&mut cfg);
        return zeroz::repl::run(&cfg);
    }

    // LLM auto-on: start local AERIS-64 if down (ZEROZ_AERIS_AUTO=0 to disable)
    if let Some(msg) = zeroz::aeris_boot::ensure_aeris(&mut cfg) {
        if env::var("ZEROZ_QUIET_BOOT").ok().as_deref() != Some("1") {
            eprintln!("• {msg}");
        }
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

    zeroz::repl::run(&cfg)
}

fn print_help() {
    println!("0zcode {VERSION} — Agentic work. Zero code.");
    println!(
        "\nUsage:\n  0zcode                      interactive TUI (auto-starts AERIS-64 LLM if local)\n  0zcode <message>            one-shot\n  0zcode serve                Rust backend → SENTINEL core (default :8100)\n  0zcode serve --port 8100 --core http://127.0.0.1:8000\n  0zcode tools|policy|llm     meta\n  0zcode --version\n\nEnv / config:\n  ~/.config/0zcode/config.toml  (or ZEROZ_CONFIG)\n  ZEROZ_AERIS_URL               core or backend URL (default http://127.0.0.1:8000)\n  ZEROZ_BACKEND_PORT            serve listen port (default 8100)\n  ZEROZ_AERIS_AUTO=0            disable auto-start LLM\n  ZEROZ_REPO=~/AERIS_64         repo root for auto-start\n  NEUROFIELD_CHECKPOINT         model.safetensors folder\n  ZEROZ_OFFLINE=1               calc-only\n  ZEROZ_PLAIN=1                 force plain line mode\n"
    );
}
