use std::process::{Command, Stdio};
use std::path::Path;
use std::time::{Duration, Instant};
use std::thread;

#[test]
fn test_qwen_capital_of_china() {
    // Model from: https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
    let model_file = "../Qwen3.5-2B-Q4_K_M.gguf";
    let llama_cli = "../llama.cpp/build/bin/llama-cli";

    // Skip if environment is not ready
    if !Path::new(model_file).exists() {
        println!("Skipping: Qwen model not found at {}", model_file);
        return;
    }
    if !Path::new(llama_cli).exists() {
        println!("Skipping: llama-cli not found at {}. Run 'cargo test -p burn-ggml' first to build it.", llama_cli);
        return;
    }

    // Allow overriding prompt via environment variable
    let mut custom_prompt = std::env::var("PROMPT").ok();
    
    // Check for --prompt <value> in args as a fallback, but environment variable is preferred
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--prompt" && i + 1 < args.len() {
            custom_prompt = Some(args[i+1].clone());
            break;
        }
    }

    let default_prompt = "What is the capital of China? Answer only with the name.";
    let is_custom = custom_prompt.is_some();
    let prompt = custom_prompt.unwrap_or_else(|| default_prompt.to_string());
    
    println!("Starting Qwen inference with prompt: '{}'", prompt);
    
    let mut child = Command::new(llama_cli)
        .stdin(Stdio::null()) // Critical: Prevents infinite loop/waiting for input
        .args(&[
            "-m", model_file,
            "-p", &prompt,
            "-n", "32",           // Limit tokens
            "-c", "128",          // Small context for speed
            "-ngl", "99",         // Offload all layers to GPU
            "-t", "4",            // Use 4 threads
            "--temp", "0.0",      // Deterministic
            "--no-display-prompt",// Clean output
            "--simple-io",        // Disable fancy terminal logic
            "--reasoning", "off", // Disable thinking for faster response
            "-st",                // Single turn only
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // Show progress/logs in real-time
        .spawn()
        .expect("Failed to spawn llama-cli");

    // 60-second timeout as requested
    let timeout = Duration::from_secs(60);
    let start = Instant::now();
    let mut status = None;
    
    while start.elapsed() < timeout {
        if let Some(s) = child.try_wait().expect("Wait failed") {
            status = Some(s);
            break;
        }
        thread::sleep(Duration::from_millis(500));
        println!("... generating ({:.1}s elapsed) ...", start.elapsed().as_secs_f32());
    }

    if status.is_none() {
        println!("TIMEOUT: Inference took too long. Killing process.");
        child.kill().ok();
        panic!("Inference timed out after 60s");
    }

    let output = child.wait_with_output().expect("Failed to read output");
    let response = String::from_utf8_lossy(&output.stdout);
    let response_lower = response.to_lowercase();
    
    println!("\n--- GENERATED RESPONSE ---\n{}\n--------------------------\n", response);
    
    if !is_custom {
        assert!(response_lower.contains("beijing"), "Response did not contain 'beijing'. Actual response: {}", response);
        println!("Assertion passed: Capital of China correctly identified.");
    } else {
        println!("Custom prompt used, skipping 'beijing' assertion.");
    }
}
