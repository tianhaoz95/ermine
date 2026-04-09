use burn_ggml::model::qwen35::{Qwen35Runner, Qwen35RunnerOptions};
use std::path::{Path, PathBuf};

struct CliArgs {
    prompt: String,
    model_dir: PathBuf,
    max_new_tokens: usize,
    max_layers_in_ram: usize,
}

impl CliArgs {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let mut model_dir = workspace_root().join("model/qwen-3.5-0.8b");
        let mut max_new_tokens = 8usize;
        let mut max_layers_in_ram = 24usize;
        let mut prompt_parts = Vec::new();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    model_dir = PathBuf::from(args.next().ok_or("--model-dir requires a path")?);
                }
                "--max-new-tokens" => {
                    let value = args.next().ok_or("--max-new-tokens requires a number")?;
                    max_new_tokens = value
                        .parse()
                        .map_err(|_| format!("invalid --max-new-tokens value: {value}"))?;
                }
                "--max-layers-in-ram" => {
                    let value = args.next().ok_or("--max-layers-in-ram requires a number")?;
                    max_layers_in_ram = value
                        .parse()
                        .map_err(|_| format!("invalid --max-layers-in-ram value: {value}"))?;
                }
                "--help" | "-h" => return Err(usage()),
                _ => prompt_parts.push(arg),
            }
        }

        if prompt_parts.is_empty() {
            return Err(usage());
        }

        Ok(Self {
            prompt: prompt_parts.join(" "),
            model_dir,
            max_new_tokens,
            max_layers_in_ram,
        })
    }
}

fn main() {
    let args = match CliArgs::parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    if let Err(err) = run(args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(args: CliArgs) -> Result<(), String> {
    if !args.model_dir.exists() {
        return Err(format!("model directory not found: {}", args.model_dir.display()));
    }

    let runner = Qwen35Runner::load(
        &args.model_dir,
        &Qwen35RunnerOptions {
            max_new_tokens: args.max_new_tokens,
            max_layers_in_ram: args.max_layers_in_ram,
        },
    )?;
    let response = runner.generate(&args.prompt, args.max_new_tokens)?;
    println!("{}", response.trim());
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("burn-ggml should live under the workspace root")
        .to_path_buf()
}

fn usage() -> String {
    "Usage: cargo run -p burn-ggml -- [--model-dir PATH] [--max-new-tokens N] [--max-layers-in-ram N] <prompt>".to_string()
}
