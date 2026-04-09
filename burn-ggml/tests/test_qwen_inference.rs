use burn_ggml::{GgmlBackend, GgmlDevice, gguf::GgufIndex, GgmlContext, GgmlOps};
use burn::tensor::{Tensor, Float, Int, TensorPrimitive};
use burn::tensor::ops::ModuleOps;
use std::path::Path;

#[test]
fn test_qwen_inference_burn_ggml() {
    // Model from: https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
    let model_file = "../Qwen3.5-2B-Q4_K_M.gguf";

    // Skip if environment is not ready
    if !Path::new(model_file).exists() {
        println!("Skipping: Qwen model not found at {}", model_file);
        return;
    }

    println!("Loading Qwen model from: {}", model_file);
    let index = GgufIndex::open(model_file).expect("Failed to open GGUF index");
    let device = GgmlDevice::Cpu;
    let ctx = GgmlContext::get(&device);

    // 1. Load Embedding Weights
    println!("Loading embedding weights...");
    let embd_primitive = unsafe { 
        index.load_tensor("token_embd.weight", &ctx).expect("Failed to load token_embd.weight") 
    };
    let embd_weight: Tensor<GgmlBackend, 2, Float> = Tensor::from_primitive(TensorPrimitive::Float(embd_primitive));

    // 2. Load Layer 0 Attention Norm Weights
    println!("Loading layer 0 attn_norm weights...");
    let norm_primitive = unsafe {
        index.load_tensor("blk.0.attn_norm.weight", &ctx).expect("Failed to load blk.0.attn_norm.weight")
    };
    let norm_weight: Tensor<GgmlBackend, 1, Float> = Tensor::from_primitive(TensorPrimitive::Float(norm_primitive));

    // 3. Prepare Input (dummy token IDs)
    // "What" in Qwen might be token 1234 (just dummy for test)
    let input_ids = Tensor::<GgmlBackend, 1, Int>::from_data([1234i32, 5678i32], &device);

    // 4. Run Inference (Embedding + RMSNorm)
    println!("Running Embedding...");
    // Use associated function syntax for ModuleOps::embedding
    let x_primitive = GgmlBackend::embedding(embd_weight.into_primitive().tensor(), input_ids.into_primitive());
    let x: Tensor<GgmlBackend, 2, Float> = Tensor::from_primitive(TensorPrimitive::Float(x_primitive));
    
    println!("Running RMSNorm...");
    // GgmlOps extension
    let x_primitive = match x.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => panic!("Expected Float primitive"),
    };
    let norm_weight_primitive = match norm_weight.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => panic!("Expected Float primitive"),
    };
    
    let out_primitive = GgmlBackend::rms_norm(x_primitive, norm_weight_primitive, 1e-6);
    let out: Tensor<GgmlBackend, 2, Float> = Tensor::from_primitive(TensorPrimitive::Float(out_primitive));

    // 5. Verify Output
    let data = out.into_data();
    let values = data.as_slice::<f32>().unwrap();
    println!("Inference output (first 10 values): {:?}", &values[..std::cmp::min(values.len(), 10)]);
    
    let sum: f32 = values.iter().sum();
    assert!(sum.abs() > 1e-5, "Output should not be zero");
    println!("Success: Inference pass through Embedding + RMSNorm completed.");
}
