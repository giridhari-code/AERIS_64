//! 0zcode — company-standard terminal agent binary.

use std::env;
use std::process::ExitCode;
use zeroz::agent::handle;
use zeroz::config::Config;
use zeroz::policy::refusal_message;
use zeroz::tools;
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
    if args.first().map(String::as_str) == Some("tools") {
        println!("{}", tools::catalog());
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("policy") {
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

    zeroz::repl::run(&cfg)
}

fn print_help() {
    println!("0zcode {VERSION} — Agentic work. Zero code.");
    println!(
        "\nUsage:\n  0zcode                 interactive TUI\n  0zcode <message>       one-shot\n  0zcode tools|policy    meta\n  0zcode --version\n\nEnv / config:\n  ~/.config/0zcode/config.toml  (or ZEROZ_CONFIG)\n  ZEROZ_OFFLINE=1               calc-only\n  ZEROZ_HTTP_TIMEOUT_SECS\n  ZEROZ_MODEL_LABEL\n  ZEROZ_NO_COLOR / NO_COLOR\n  ZEROZ_PLAIN=1                 force plain line mode\n  ZEROZ_HISTORY_FILE\n  ZEROZ_AERIS_URL               AERIS-64 LLM fallback\n"
    );
}
