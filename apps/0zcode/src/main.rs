//! 0zcode — company-standard terminal agent binary.

use std::env;
use std::io::{self, Write};
use std::process::ExitCode;
use zeroz::agent::handle;
use zeroz::config::Config;
use zeroz::policy::{banner_line, refusal_message};
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
    println!("{}", banner_line());
    println!(
        "\nUsage:\n  0zcode                 interactive terminal\n  0zcode <message>       one-shot\n  0zcode tools|policy    meta\n  0zcode --version\n\nEnv:\n  ZEROZ_HTTP_TIMEOUT_SECS   default 12\n  ZEROZ_NO_COLOR            disable ANSI\n  NO_COLOR                  disable ANSI\n"
    );
}

fn run_repl(cfg: &Config) -> ExitCode {
    let theme = Theme::from_config(cfg);
    theme.clear_screen();
    theme.print_header();

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
        let r = handle(line, cfg);
        if r.text == "__QUIT__" {
            println!("bye.");
            break;
        }
        if r.text == "__CLEAR__" {
            theme.clear_screen();
            theme.print_header();
            continue;
        }
        println!("\n{}\n", r.text);
    }
    ExitCode::SUCCESS
}
