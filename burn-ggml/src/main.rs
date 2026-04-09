use burn_ggml::gguf::GgufIndex;
use burn_ggml::memory::LayerKey;
use burn_ggml::{GgmlContext, GgmlDevice};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

struct CliArgs {
    prompt: String,
    model_path: PathBuf,
    llama_cli_path: PathBuf,
    max_layers_in_ram: usize,
    warm_layers: usize,
    n_predict: usize,
    gpu_layers: String,
}

impl CliArgs {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args().skip(1);
        let root_dir = workspace_root();
        let mut model_path = root_dir.join("Qwen3.5-2B-Q4_K_M.gguf");
        let mut llama_cli_path = root_dir.join("llama.cpp/build/bin/llama-cli");
        let mut max_layers_in_ram = 4usize;
        let mut warm_layers = 1usize;
        let mut n_predict = 64usize;
        let mut gpu_layers = "99".to_string();
        let mut prompt_parts = Vec::new();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    let value = args.next().ok_or("--model requires a path")?;
                    model_path = PathBuf::from(value);
                }
                "--llama-cli" => {
                    let value = args.next().ok_or("--llama-cli requires a path")?;
                    llama_cli_path = PathBuf::from(value);
                }
                "--max-layers-in-ram" => {
                    let value = args.next().ok_or("--max-layers-in-ram requires a number")?;
                    max_layers_in_ram = value
                        .parse()
                        .map_err(|_| format!("invalid --max-layers-in-ram value: {value}"))?;
                }
                "--warm-layers" => {
                    let value = args.next().ok_or("--warm-layers requires a number")?;
                    warm_layers = value
                        .parse()
                        .map_err(|_| format!("invalid --warm-layers value: {value}"))?;
                }
                "--n-predict" => {
                    let value = args.next().ok_or("--n-predict requires a number")?;
                    n_predict = value
                        .parse()
                        .map_err(|_| format!("invalid --n-predict value: {value}"))?;
                }
                "--gpu-layers" => {
                    gpu_layers = args.next().ok_or("--gpu-layers requires a value")?;
                }
                "--help" | "-h" => {
                    return Err(usage());
                }
                _ => prompt_parts.push(arg),
            }
        }

        if prompt_parts.is_empty() {
            return Err(usage());
        }

        Ok(Self {
            prompt: prompt_parts.join(" "),
            model_path,
            llama_cli_path,
            max_layers_in_ram,
            warm_layers,
            n_predict,
            gpu_layers,
        })
    }
}

#[tokio::main]
async fn main() {
    let args = match CliArgs::parse() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    if let Err(err) = run(args).await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

async fn run(args: CliArgs) -> Result<(), String> {
    if !args.model_path.exists() {
        return Err(format!(
            "model file not found: {}",
            args.model_path.display()
        ));
    }
    if !args.llama_cli_path.exists() {
        return Err(format!(
            "llama-cli not found: {}",
            args.llama_cli_path.display()
        ));
    }

    warm_offload_cache(&args).await?;

    let output = Command::new(&args.llama_cli_path)
        .arg("-m")
        .arg(&args.model_path)
        .arg("-ngl")
        .arg(&args.gpu_layers)
        .arg("-n")
        .arg(args.n_predict.to_string())
        .arg("-cnv")
        .arg("-st")
        .arg("--reasoning")
        .arg("off")
        .arg("--no-display-prompt")
        .arg("--simple-io")
        .arg("--no-warmup")
        .arg("-p")
        .arg(&args.prompt)
        .output()
        .map_err(|err| format!("failed to run llama-cli: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("llama-cli failed: {detail}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let response = extract_response(&stdout, &args.prompt)
        .ok_or_else(|| format!("failed to parse llama-cli output\n{stdout}"))?;

    println!("{response}");
    Ok(())
}

async fn warm_offload_cache(args: &CliArgs) -> Result<(), String> {
    let device = GgmlDevice::MetalWithOffload {
        kv_cache_dir: workspace_root().join("kv_cache"),
        max_layers_in_ram: args.max_layers_in_ram,
    };
    let index = Arc::new(
        GgufIndex::open(&args.model_path)
            .map_err(|err| format!("failed to open GGUF index: {err}"))?,
    );
    let ctx = GgmlContext::get(&device);
    ctx.init_cache(index.clone()).await;

    let Some(cache) = ctx.layer_cache.get() else {
        return Ok(());
    };

    let layer_count = index
        .tensors
        .keys()
        .filter_map(|name| {
            let suffix = name.strip_prefix("blk.")?;
            let layer = suffix.split('.').next()?;
            layer.parse::<usize>().ok()
        })
        .max()
        .map(|max_layer| max_layer + 1)
        .unwrap_or(0);

    let warm_count = args.warm_layers.min(layer_count);
    if warm_count == 0 {
        return Ok(());
    }

    let keys: Vec<LayerKey> = (0..warm_count).map(|layer| LayerKey { layer }).collect();
    cache.prefetch(&keys);

    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("burn-ggml should live under the workspace root")
        .to_path_buf()
}

fn usage() -> String {
    "Usage: cargo run -p burn-ggml -- [--model PATH] [--llama-cli PATH] [--max-layers-in-ram N] [--warm-layers N] [--n-predict N] [--gpu-layers N] <prompt>".to_string()
}

fn extract_response(output: &str, prompt: &str) -> Option<String> {
    let prompt_marker = format!("> {prompt}");
    let tail = output.rsplit_once(&prompt_marker)?.1;
    let mut lines = Vec::new();
    let mut started = false;

    for line in tail.lines() {
        let trimmed = line.trim();

        if !started {
            if trimmed.is_empty() {
                continue;
            }
            started = true;
        }

        if trimmed.is_empty() || trimmed.starts_with("[ Prompt:") || trimmed == "Exiting..." {
            break;
        }

        lines.push(trimmed);
    }

    let response = lines.join("\n").trim().to_string();
    (!response.is_empty()).then_some(response)
}

#[cfg(test)]
mod tests {
    use super::extract_response;

    #[test]
    fn extracts_single_line_response() {
        let output = "\
header
> what is 1+1? only answer with number

2

[ Prompt: 10.0 t/s | Generation: 20.0 t/s ]
Exiting...
";

        let response = extract_response(output, "what is 1+1? only answer with number");
        assert_eq!(response.as_deref(), Some("2"));
    }
}
