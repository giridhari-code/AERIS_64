//! 0zcode binary.

use std::env;
use std::process::ExitCode;
use zeroz::agent::handle;
use zeroz::config::Config;
use zeroz::policy::{banner_line, refusal_message};
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
        println!("{}", banner_line());
        println!(
            "\nUsage:\n  0zcode                 interactive TUI\n  0zcode <message>       one-shot\n  0zcode tools|policy\n"
        );
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

    match zeroz::ui::run_app(&cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("0zcode TUI error: {e}");
            ExitCode::FAILURE
        }
    }
}
