use burn::tensor::Tensor;
use burn_wgpu::{Wgpu, WgpuDevice};
use burn_ggml::{OffloadBackend, OFFLOAD_PREFETCH_COUNT, TensorPrefetch};
use std::sync::atomic::Ordering;

#[test]
fn test_wgpu_prefetch_simulated_improvement() {
    type WgpuOffload = OffloadBackend<Wgpu>;
    let device = WgpuDevice::DefaultDevice;
    
    // Reset counter
    OFFLOAD_PREFETCH_COUNT.store(0, Ordering::SeqCst);
    
    // Create some tensors on the offload backend
    let t1: Tensor<WgpuOffload, 1> = Tensor::from_data([1.0], &device);
    let t2: Tensor<WgpuOffload, 1> = Tensor::from_data([2.0], &device);
    
    assert_eq!(OFFLOAD_PREFETCH_COUNT.load(Ordering::SeqCst), 0, "Initially 0 prefetch calls");

    // Verify that prefetch increments the count
    let _t1 = t1.prefetch(&device);
    assert_eq!(OFFLOAD_PREFETCH_COUNT.load(Ordering::SeqCst), 1, "Prefetch should increment count");
    
    let _t2 = t2.prefetch(&device);
    assert_eq!(OFFLOAD_PREFETCH_COUNT.load(Ordering::SeqCst), 2, "Second prefetch should increment count again");
}
