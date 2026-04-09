use burn::tensor::{Float, Tensor, TensorData, TensorPrimitive};
use burn_ggml::memory::{LayerKey, WeightCache};
use burn_ggml::{gguf::GgufIndex, GgmlBackend, GgmlContext, GgmlDevice};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::Write;
use std::sync::Arc;

fn create_test_model(path: &str) {
    let mut buf = Vec::new();

    // Magic and Version
    buf.write_u32::<LittleEndian>(0x46554747).unwrap(); // 'GGUF'
    buf.write_u32::<LittleEndian>(3).unwrap();

    // Header: 2 tensors, 1 KV pair
    buf.write_u64::<LittleEndian>(2).unwrap();
    buf.write_u64::<LittleEndian>(1).unwrap();

    // Metadata: general.alignment = 32
    write_string(&mut buf, "general.alignment");
    buf.write_u32::<LittleEndian>(4).unwrap(); // Uint32
    buf.write_u32::<LittleEndian>(32).unwrap();

    // Tensor Info 0: "blk.0.weight", [2], F32, offset 0
    write_string(&mut buf, "blk.0.weight");
    buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(2).unwrap(); // d1
    buf.write_u32::<LittleEndian>(0).unwrap(); // GGML_TYPE_F32
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    // Tensor Info 1: "blk.1.weight", [2], F32, offset 8
    write_string(&mut buf, "blk.1.weight");
    buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(2).unwrap(); // d1
    buf.write_u32::<LittleEndian>(0).unwrap(); // GGML_TYPE_F32
    buf.write_u64::<LittleEndian>(8).unwrap(); // offset

    // Padding to 32 bytes
    let pos = buf.len();
    let pad = (32 - (pos % 32)) % 32;
    for _ in 0..pad {
        buf.write_u8(0).unwrap();
    }

    // Tensor Data 0: [1.0, 2.0]
    buf.write_f32::<LittleEndian>(1.0).unwrap();
    buf.write_f32::<LittleEndian>(2.0).unwrap();

    // Tensor Data 1: [3.0, 4.0]
    buf.write_f32::<LittleEndian>(3.0).unwrap();
    buf.write_f32::<LittleEndian>(4.0).unwrap();

    std::fs::write(path, buf).unwrap();
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

#[tokio::test]
async fn test_weight_cache_loading() {
    let model_path = "cache_test.gguf";
    create_test_model(model_path);

    let index = Arc::new(GgufIndex::open(model_path).unwrap());
    let device = GgmlDevice::Cpu;
    let ctx = GgmlContext::get(&device);

    let cache = WeightCache::new(index.clone(), ctx.clone(), 1); // Only 1 slot

    // Load layer 0
    println!("Loading layer 0...");
    let slot0 = cache.get(LayerKey { layer: 0 }).await;
    let tensors0 = slot0.tensors.get().unwrap();
    assert!(tensors0.contains_key("blk.0.weight"));

    let t0 = tensors0.get("blk.0.weight").unwrap();
    let tensor0: Tensor<GgmlBackend, 1, Float> =
        Tensor::from_primitive(TensorPrimitive::Float(t0.clone()));
    let data0: TensorData = tensor0.into_data();
    assert_eq!(data0.as_slice::<f32>().unwrap(), &[1.0, 2.0]);

    // Load layer 1 (should evict layer 0 from LRU, but slot0 still held by Arc)
    println!("Loading layer 1...");
    let slot1 = cache.get(LayerKey { layer: 1 }).await;
    let tensors1 = slot1.tensors.get().unwrap();
    assert!(tensors1.contains_key("blk.1.weight"));

    let t1 = tensors1.get("blk.1.weight").unwrap();
    let tensor1: Tensor<GgmlBackend, 1, Float> =
        Tensor::from_primitive(TensorPrimitive::Float(t1.clone()));
    let data1: TensorData = tensor1.into_data();
    assert_eq!(data1.as_slice::<f32>().unwrap(), &[3.0, 4.0]);

    // Verify prefetch
    println!("Testing prefetch...");
    cache.prefetch(&[LayerKey { layer: 0 }]);
    // Wait a bit for prefetch task
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let slot0_new = cache.get(LayerKey { layer: 0 }).await;
    let tensors0_new = slot0_new.tensors.get().unwrap();
    assert!(tensors0_new.contains_key("blk.0.weight"));

    std::fs::remove_file(model_path).ok();
}
