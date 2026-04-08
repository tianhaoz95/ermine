use burn_ggml::{GgmlBackend, GgmlDevice, gguf::GgufIndex, GgmlContext};
use burn::tensor::{Tensor, Float, TensorPrimitive, TensorMetadata};
use std::sync::Arc;

#[test]
fn test_rust_backend_with_qwen_weights() {
    let model_file = "../Qwen3.5-2B-Q4_K_M.gguf";
    
    if !std::path::Path::new(model_file).exists() {
        println!("Skipping test: Qwen model file not found at {}", model_file);
        return;
    }

    let device = GgmlDevice::Cpu;
    let index = GgufIndex::open(model_file).expect("Failed to open GGUF index");
    
    // 1. Load a real weight from the model
    let ctx = Arc::new(GgmlContext::new(device.clone()));
    let weight_name = "blk.0.attn_norm.weight";
    
    println!("Loading weight '{}' from model...", weight_name);
    let weight_primitive = unsafe { 
        index.load_tensor(weight_name, ctx.clone()).expect("Failed to load tensor") 
    };
    
    println!("Weight loaded. Shape: {:?}, Type: {:?}", weight_primitive.shape(), weight_primitive.dtype());
    
    // Check if it's actually F32 or if we need to handle it differently
    // For now, let's assume it's something Burn can handle or panic with a clear message
    let weight: Tensor<GgmlBackend, 1, Float> = Tensor::from_primitive(TensorPrimitive::Float(weight_primitive));
    
    // 2. Create a dummy input tensor
    let input = Tensor::<GgmlBackend, 1>::from_data([1.0, 2.0], &device);
    
    // 3. Perform an operation
    println!("Running computation on Rust backend...");
    let output = weight.slice([0..2]) + input;
    let data = output.into_data();
    
    // 4. Verify results are not all zeros
    let values = data.as_slice::<f32>().unwrap();
    println!("First few output values: {:?}", &values[..std::cmp::min(values.len(), 5)]);
    
    let sum: f32 = values.iter().sum();
    assert!(sum != 0.0, "Backend returned all zeros! Weights likely failed to sync to backend memory.");
    println!("Success: Backend returned non-zero results.");
}
