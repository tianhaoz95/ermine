# Ermine: GEMINI Mandates

This document defines the foundational mandates, architectural standards, and operational workflows for the Ermine project. These instructions take absolute precedence over general defaults.

## 1. Architectural Vision
Ermine is a **Burn backend** that leverages **GGML** (via `llama.cpp`) for high-performance tensor operations and hardware acceleration.
- **Goal**: Seamlessly bridge Burn's high-level ergonomic API with GGML's low-level performance and quantization kernels.
- **Constraint**: Maintain compatibility with the `burn-tensor` API while exposing GGML-specific capabilities (like GGUF lazy loading and backend offloading) where appropriate.

## 2. Coding Standards

### 2.1 Rust & Safety
- **FFI Boundary**: All interaction with `ggml-sys` must be encapsulated within safe Rust abstractions in `burn-ggml`.
- **Unsafe blocks**: Must be accompanied by a `// SAFETY:` comment explaining why the invariants are upheld.
- **Resource Management**: Strictly use `Arc` or `Box` with custom `Drop` implementations to manage GGML contexts and tensors, ensuring no memory leaks from the C side.

### 2.2 GGML Integration
- **Context Management**: GGML contexts (`ggml_context`) and backends should be managed via the `GgmlContext` struct.
- **Tensor Alignment**: Always respect GGML's alignment requirements (default 32 bytes) when loading or creating tensors.
- **Backend Offloading**: Prioritize using `ggml-backend` APIs for hardware-agnostic acceleration (Metal, CUDA, Vulkan).

## 3. Testing Protocols

### 3.1 Unit Tests
- Must be fast and runnable without external model files.
- Use the tiny model generation utility (`create_test_model.rs`) for testing GGUF parsing and basic ops.
- **Run all unit tests**: `cargo test -p burn-ggml`
- **Run GGUF loader test specifically**: `cargo test -p burn-ggml --lib gguf::tests::test_gguf_loading`
- **Verify backend with tiny model**: `cargo test -p burn-ggml --test test_backend_full`

### 3.2 Integration Tests (Model-Based)
- Integration tests involving real models (e.g., Qwen) should gracefully skip if the required `.gguf` file is missing.
- **Verification**: Always assert behavioral correctness (e.g., specific text output or non-zero math results) rather than just successful execution.
- **Performance**: Monitor execution time; inference tests should ideally complete within 60 seconds on a modern CPU.

#### 3.2.1 Full Model Inference (GPU Accelerated)
Verifies text generation using the `llama-cli` reference runner and the **Qwen3.5-2B-GGUF** model.
- **Default test**: `cargo test -p burn-ggml --test test_qwen_inference -- --nocapture`
- **Custom prompt**: `PROMPT="What is the capital of France?" cargo test -p burn-ggml --test test_qwen_inference -- --nocapture` (Note: skip "beijing" assertion when using custom prompts).

#### 3.2.2 Rust-Native Backend Math
Verifies the Rust `burn-ggml` bridge by loading real weights and performing individual tensor operations.
- **Run command**: `cargo test -p burn-ggml --test test_backend_rust -- --nocapture`

## 4. GGUF Standards
- Ermine must support **GGUF v3** and handle metadata parsing lazily.
- Tensors should be memory-mapped (`memmap2`) whenever possible to reduce RAM pressure.

## 5. Dependency Management
- **llama.cpp**: This is a git submodule. Do not modify files inside `llama.cpp/` directly. If changes are needed, they should be proposed as upstream-compatible patches or handled via the `ggml-sys` wrapper.
- **bindgen**: Keep FFI bindings minimal and scoped only to necessary GGML/backend functions.
