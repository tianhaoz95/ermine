use burn_ggml::{GgmlBackend, GgmlDevice, gguf::GgufIndex};
use burn::tensor::{Tensor, TensorData, Tolerance, Float, TensorPrimitive};
use std::process::Command;
use std::io::Write;

#[test]
fn test_burn_ggml_backend_full() {
    let python_script = r#"
import struct

def create_gguf_model(path):
    # GGUF Magic
    magic = b"GGUF"
    version = 3
    n_tensors = 1
    n_kv = 1
    
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))
        
        # KV: general.alignment
        key = b"general.alignment"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 4)) # Type: Uint32
        f.write(struct.pack("<I", 32))
        
        # Tensor Info
        name = b"test.weight"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        f.write(struct.pack("<I", 2)) # 2 dims
        f.write(struct.pack("<Q", 2)) # 2
        f.write(struct.pack("<Q", 2)) # 2
        f.write(struct.pack("<I", 0)) # Type: F32
        f.write(struct.pack("<Q", 0)) # offset
        
        # Alignment padding
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        f.write(b"\0" * pad)
        
        # Tensor Data
        for val in [1.0, 2.0, 3.0, 4.0]:
            f.write(struct.pack("<f", val))

if __name__ == "__main__":
    create_gguf_model("test_model.gguf")
"#;

    // 1. Create a test GGUF model using the embedded python script
    let mut script_file = std::fs::File::create("create_test_model_tmp.py").expect("Failed to create script file");
    script_file.write_all(python_script.as_bytes()).expect("Failed to write script");
    drop(script_file);

    let status = Command::new("python3")
        .arg("create_test_model_tmp.py")
        .status()
        .expect("Failed to create test model");
    assert!(status.success());

    let device = GgmlDevice::Cpu;
    let index = GgufIndex::open("test_model.gguf").expect("Failed to open GGUF index");
    
    // 2. Load tensor from GGUF
    let ctx = std::sync::Arc::new(burn_ggml::GgmlContext::new(device.clone()));
    let weight_primitive = unsafe { index.load_tensor("test.weight", ctx.clone()).expect("Failed to load tensor") };
    let weight: Tensor<GgmlBackend, 2, Float> = Tensor::from_primitive(TensorPrimitive::Float(weight_primitive));
    
    // 3. Run some operations to verify synchronization and backend persistence
    let input = Tensor::<GgmlBackend, 2>::from_data([[10.0, 20.0], [30.0, 40.0]], &device);
    
    // Test Add
    let sum = weight.clone() + input.clone();
    let sum_data = sum.into_data();
    sum_data.assert_approx_eq(&TensorData::from([[11.0, 22.0], [33.0, 44.0]]), Tolerance::<f32>::absolute(1e-5));
    
    // Test Matmul
    let res = weight.matmul(input);
    let res_data = res.into_data();
    res_data.assert_approx_eq(&TensorData::from([[70.0, 100.0], [150.0, 220.0]]), Tolerance::<f32>::absolute(1e-5));

    // Cleanup
    let _ = std::fs::remove_file("test_model.gguf");
    let _ = std::fs::remove_file("create_test_model_tmp.py");
}
