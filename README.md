# Ermine: burn-ggml Backend

`burn-ggml` is a [Burn](https://burn.dev) backend that delegates tensor operations to [ggml](https://github.com/ggerganov/llama.cpp), providing access to high-performance quantization kernels and hardware acceleration (Metal/Vulkan/CUDA).

## Prerequisites

To build this project, you need:
- **Rust** (Stable)
- **CMake** (to build ggml)
- **Clang/LLVM** (for bindgen FFI bindings)
- **OpenMP** (optional but recommended for CPU performance)

## Building

```bash
cargo build -p burn-ggml
```

## Testing

### Run Unit Tests
This runs the basic operator tests (add, mul, matmul) and the GGUF loader test.
```bash
cargo test -p burn-ggml
```

### Run GGUF Loading Test specifically
```bash
cargo test -p burn-ggml --lib gguf::tests::test_gguf_loading
```

### Integration Testing with Real Models
These tests require an existing GGUF model file (default: `Qwen3.5-2B-Q4_K_M.gguf` in the parent directory).

#### 1. Full Model Inference (GPU Accelerated)
Uses the `llama-cli` reference runner to verify that the model generates correct text. This test identifies "Beijing" as the capital of China using the **[Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)** model.

The test is configured with a **60-second timeout** and asserts that the response contains "beijing". It offloads all layers to the GPU (`-ngl 99`) if available.

```bash
# Default prompt
cargo test -p burn-ggml --test test_qwen_inference -- --nocapture

# Custom prompt (assertion for "beijing" will be skipped)
PROMPT="What is the capital of France?" cargo test -p burn-ggml --test test_qwen_inference -- --nocapture
```

#### 2. Rust-Native Backend Math
Verifies our actual Rust `burn-ggml` bridge by loading real weights and performing individual tensor operations directly via our backend.
```bash
cargo test -p burn-ggml --test test_backend_rust -- --nocapture
```

### Full Backend Verification (Tiny Model)
To verify the `burn-ggml` backend logic (including GGUF loading and tensor operations) using a dynamically generated tiny model (no download required):
```bash
cargo test -p burn-ggml --test test_backend_full
```

## How to use with a Model

To test with a GGUF model, follow this pattern:

### 1. Load the GGUF Index
```rust
use burn_ggml::gguf::GgufIndex;
use std::path::Path;

let index = GgufIndex::open(Path::new("path/to/model.gguf")).unwrap();
```

### 2. Initialize the Backend Context
```rust
use burn_ggml::{GgmlBackend, GgmlDevice, GgmlContext};
use std::sync::Arc;

let device = GgmlDevice::Cpu;
let ctx = Arc::new(GgmlContext::new(device));
```

### 3. Load Tensors into Burn
```rust
use burn::tensor::Tensor;

// Load a specific weight from GGUF
let weight_primitive = unsafe { 
    index.load_tensor("blk.0.attn_q.weight", ctx.clone()).unwrap() 
};

// Wrap it in a Burn Tensor
let weight: Tensor<GgmlBackend, 2> = Tensor::from_primitive(weight_primitive);
```

### 4. Run Operations
```rust
let input = Tensor::<GgmlBackend, 2>::random([1, 5120], Distribution::Default, &device);
let output = weight.matmul(input);
```

## Project Structure

- `ggml-sys`: Low-level FFI bindings to `llama.cpp/ggml`.
- `burn-ggml`: The Burn backend implementation.
  - `src/gguf.rs`: GGUF file parser and tensor loader.
  - `src/ops/`: Implementation of Burn tensor operations.
  - `src/memory/`: Weight streaming and KV cache offload logic (WIP).

## Current Implementation Status

- [x] GGUF Metadata Parsing & Lazy Loading.
- [x] Basic Float Operations (Add, Sub, Mul, Div).
- [x] Matmul Support.
- [x] Linear Layer Support.
- [ ] Quantization Support (Q4_K, Q3_K).
- [ ] SSD KV Cache Offloading.
- [x] GPU Acceleration (via ggml-backend).
