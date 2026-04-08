use burn::tensor::Tensor;
use burn_ggml::{GgmlBackend, GgmlDevice, get_prefetch_count, TensorPrefetch};

#[test]
fn test_tensor_prefetch_updates_count() {
    let device = GgmlDevice::Cpu;
    let tensor: Tensor<GgmlBackend, 2> = Tensor::zeros([3, 3], &device);
    
    let initial_count = get_prefetch_count();
    
    // Call prefetch (via extension trait)
    let tensor = tensor.prefetch(&device);
    
    let final_count = get_prefetch_count();
    
    assert_eq!(final_count, initial_count + 1, "Prefetch count should have increased by 1");
    
    // Test chaining
    let _ = tensor.prefetch(&device).prefetch(&device);
    assert_eq!(get_prefetch_count(), initial_count + 3, "Prefetch count should have increased by 3 total");
}
