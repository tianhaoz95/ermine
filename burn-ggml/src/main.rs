use burn_ggml::{GgmlBackend, GgmlDevice, gguf::GgufIndex, GgmlContext};
use burn_ggml::model::qwen::QwenModel;
use burn_ggml::model::ModelConfig;
use burn_ggml::ops::tokenizer::SimpleTokenizer;
use burn::tensor::{Tensor, Int, Float, TensorData};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run -- <prompt>");
        return;
    }
    let prompt = &args[1];

    let model_file = "Qwen3.5-2B-Q4_K_M.gguf";
    if !std::path::Path::new(model_file).exists() {
        println!("Model file not found: {}", model_file);
        return;
    }

    println!("Loading GGUF index...");
    let index = Arc::new(GgufIndex::open(model_file).expect("Failed to open GGUF index"));
    let mut config = ModelConfig::from_index(&index);
    config.n_layers = 1; // Use only 1 layer for fast PoC
    let tokenizer = SimpleTokenizer::from_index(&index);

    let device = GgmlDevice::MetalWithOffload {
        kv_cache_dir: PathBuf::from("kv_cache"),
        max_layers_in_ram: 32,
    };
    let ctx = GgmlContext::get(&device);
    ctx.init_cache(index.clone()).await;

    println!("Loading global weights...");
    let mut weights = HashMap::new();
    let global_tensors = vec!["token_embd.weight", "output_norm.weight"];
    for name in global_tensors {
        let t = unsafe { index.load_tensor(name, &ctx).expect(&format!("Failed to load {}", name)) };
        weights.insert(name.to_string(), t);
    }

    let output_weight = match unsafe { index.load_tensor("output.weight", &ctx) } {
        Ok(t) => t,
        Err(_) => weights.get("token_embd.weight").expect("Missing token_embd.weight").clone(),
    };
    weights.insert("output.weight".to_string(), output_weight);

    let model = QwenModel::new(config, weights, ctx.layer_cache.get().cloned());

    println!("Encoding prompt...");
    let mut tokens = tokenizer.encode(prompt);
    println!("Prompt tokens: {:?}", tokens);

    println!("Starting generation (layers: {})...", model.config.n_layers);
    for _ in 0..20 {
        let input_ids_data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let input_ids = Tensor::<GgmlBackend, 1, Int>::from_data(
            TensorData::new(input_ids_data, [tokens.len()]), 
            &device
        );
        let cur_pos = tokens.len() - 1;
        
        print!("[");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let logits: Tensor<GgmlBackend, 1, Float> = model.forward(input_ids, cur_pos).await;
        print!("] ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        // Greedy sampling
        let next_token = logits.argmax(0);
        let next_token_id = next_token.into_data().as_slice::<i32>().unwrap()[0] as u32;
        
        tokens.push(next_token_id);
        
        let new_text = tokenizer.decode(&[next_token_id]);
        print!("{}", new_text);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        if next_token_id == 151643 { // Assuming common EOS token ID
            break;
        }
    }
    println!("\nDone.");
}
