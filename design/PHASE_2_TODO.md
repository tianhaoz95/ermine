# Phase 2: ggml + Metal on macOS Implementation Tasks

**Goal:** Gemma 4 31B dense at Q3_K_M running on macOS Apple Silicon via `burn-ggml`, with layer streaming and KV offload, targeting >=2 tok/s at 256K context.

**Duration:** Weeks 11-20 (estimated)

**Status:** 🔴 Not Started

---

## Overview

Phase 2 ports the memory management architecture validated in Phase 1 (WeightCache, KvOffloadManager, GgufIndex, PrefetchOps) to the full `burn-ggml` backend running on Apple Metal. All existing Burn tests must continue to pass, and the CLI inference command must work correctly.

### Reused from Phase 1
- ✅ `PrefetchOps` trait in Burn (unchanged)
- ✅ `WeightCache<T>` unified expert + layer cache (unchanged)
- ✅ `KvOffloadManager` ping-pong SSD offload (unchanged)
- ✅ `GgufIndex` GGUF model loader (unchanged)

### New in Phase 2
- ❌ `ggml-sys` crate with FFI bindings
- ❌ Core `burn-ggml` backend implementation
- ❌ Quantized matmul (Q4_K_M, Q3_K_M)
- ❌ Hybrid attention (local + global) on Metal
- ❌ Gemma 4 31B dense model definition
- ❌ Layer streaming with cache

---

## Phase 2 Tasks

### P2-T1: `ggml-sys` Crate with FFI Bindings

**Status:** 🔴 Not Started

**Objective:** Create low-level FFI bindings to ggml (llama.cpp) with Metal support.

**Subtasks:**

- **P2-T1.1:** Set up git submodule for llama.cpp
  - Add llama.cpp as submodule pinned to specific commit SHA
  - Document submodule location and update procedure
  - **Test:** Submodule clones successfully with `git clone --recursive`

- **P2-T1.2:** Create `ggml-sys/build.rs` build script
  - CMake configuration with `LLAMA_METAL=ON` for Metal support
  - Static library linking
  - Link Metal, MetalKit, Foundation, Accelerate frameworks
  - **Test:** Build succeeds on macOS with Metal framework present
  - **Test:** Fallback gracefully on non-macOS platforms (CPU-only mode)

- **P2-T1.3:** Generate FFI bindings with bindgen
  - Bindgen configuration for `ggml.h`, `ggml-backend.h`, `ggml-alloc.h`
  - Minimal surface area: only expose actively used symbols
  - Document which symbols are intentionally excluded
  - **Test:** Bindings compile without warnings

- **P2-T1.4:** Create C wrapper for API stability
  - Write `ggml_wrapper.c` exposing stable C API
  - Wrapper absorbs ggml header churn
  - Include safety helpers: context lifecycle functions, tensor validation
  - **Test:** Wrapper compiles and links correctly

- **P2-T1.5:** Smoke test FFI bindings
  - Create `ggml-sys/tests/smoke_test.rs`
  - Test: Call `ggml_init()` and `ggml_free()` successfully
  - Test: Call `ggml_backend_metal_init()` on macOS
  - Test: Call `ggml_new_tensor` and verify return value
  - **Documentation:** Record Metal backend availability

**Acceptance Criteria:**
- ✅ `cargo build -p ggml-sys` succeeds on macOS with Metal
- ✅ All smoke tests pass
- ✅ No unsafe warnings in generated bindings
- ✅ Build fails gracefully if Metal framework unavailable

---

### P2-T2: `burn-ggml` Backend Skeleton

**Status:** 🔴 Not Started

**Objective:** Implement core GGML backend infrastructure for Burn.

**Subtasks:**

- **P2-T2.1:** Define `GgmlDevice` enum and context management
  - Device variants: `Cpu`, `Metal`, `MetalWithOffload { kv_cache_dir, max_layers_in_ram }`
  - `GgmlContext` struct with:
    - `ggml_context` pointer
    - Backend pointers (Metal primary + CPU fallback)
    - Optional `LayerWeightCache` and `KvOffloadManager`
    - Executor for graph computation
  - **Test:** Context initialization succeeds with all device variants
  - **Test:** Context cleanup (Drop impl) prevents memory leaks

- **P2-T2.2:** Implement `GgmlTensor` wrapper
  - Struct: `{ ptr: *mut ggml_tensor, ctx: Arc<GgmlContext>, shape: Shape }`
  - Enforce: tensor lifetime ≤ context lifetime via Arc
  - Implement Clone, Debug
  - **Test:** Tensor outliving context is impossible (type system prevents)

- **P2-T2.3:** Implement `PrefetchOps` for GgmlBackend
  - Variant `Cpu` / `Metal`: no-op (return early)
  - Variant `MetalWithOffload`: call `WeightCache::prefetch()`
  - Prefetch returns immediately (async, fire-and-forget)
  - **Test:** Prefetch on CPU/Metal returns immediately
  - **Test:** Prefetch on MetalWithOffload triggers cache load

- **P2-T2.4:** Implement core float ops
  - `float_matmul`: leverage `ggml_mul_mat` (must fix row-major/column-major mismatch first)
  - `float_add`, `float_mul`, `float_sub`: standard element-wise ops
  - `float_softmax`: use `ggml_soft_max`
  - `embedding_forward`: gather embeddings from weight matrix
  - `rms_norm_forward`: implement or use `ggml_rms_norm`
  - `linear_forward`: matmul + optional bias
  - **Test:** Each op matches PyTorch reference numerically
  - **Test:** Shapes propagate correctly through ops

- **P2-T2.5:** Run Burn tensor test suite
  - Execute `cargo test -p burn-ggml` against CPU backend
  - Fix any failures (tensor shape mismatches, numerical precision issues, etc.)
  - Document known limitations (e.g., matmul semantics)
  - **Acceptance:** No panics or assertion failures in Burn tests

**Acceptance Criteria:**
- ✅ All device variants initialize successfully
- ✅ Core float ops compile and run
- ✅ Burn tensor test suite passes on CPU backend
- ✅ PrefetchOps integrated and callable

---

### P2-T3: Quantized Matmul (Q4_K_M, Q3_K_M)

**Status:** 🔴 Not Started

**Objective:** Support quantized matrix multiplication for efficient inference.

**Subtasks:**

- **P2-T3.1:** Implement quantization operations
  - `QTensorOps::quantize`: F32 → Q4_K_M or Q3_K_M via `ggml_quantize_*`
  - `QTensorOps::dequantize`: Quantized → F32 via `ggml_dequantize_*`
  - Support tensor size inference (quantized size ~25-33% of float32)
  - **Test:** Quantize then dequantize, recover original values within tolerance

- **P2-T3.2:** Implement `GgmlQuantizedTensor` wrapper
  - Struct: `{ ptr: *mut ggml_tensor, ctx: Arc<GgmlContext>, dtype: QuantDType }`
  - Track quantization type (Q4_K_M, Q3_K_M, IQ3_S)
  - Lifetime safety identical to `GgmlTensor`
  - **Test:** Quantized tensor persists across operations

- **P2-T3.3:** Integrate quantized matmul with Metal
  - Call `ggml_mul_mat` on Metal backend with quantized weights
  - Verify Metal path is used (not CPU fallback)
  - **Test:** Q4_K matmul(weights, input) with Metal produces correct output
  - **Test:** Benchmark Q4_K vs F32 matmul performance ratio

- **P2-T3.4:** Implement quantized ops in ModuleOps
  - `quantized_matmul`: quantized weights × float activations
  - `quantized_embedding`: quantized weight lookup
  - **Test:** Quantized embedding matches float embedding reference

**Acceptance Criteria:**
- ✅ Quantize → dequantize recovers original (5% max error allowed)
- ✅ Q4_K matmul on Metal produces numerically correct output
- ✅ Q4_K achieves >2x throughput vs F32 on M-series GPU

---

### P2-T4: Hybrid Attention (Local + Global) on Metal

**Status:** 🔴 Not Started

**Objective:** Implement both sliding-window and full-context attention for Gemma 4 interleaved architecture.

**Subtasks:**

- **P2-T4.1:** Implement local (sliding window) attention
  - Build ggml graph with window size = 1024 tokens
  - Standard RoPE encoding
  - Use `ggml_flash_attn_ext` or fallback to manual `ggml_mul_mat` for Q @ K^T, softmax, @ V
  - **Test:** Output matches PyTorch sliding window attention (torch.nn.functional.scaled_dot_product_attention with causal mask)

- **P2-T4.2:** Implement global (full context) attention
  - Build ggml graph with full sequence length
  - Proportional RoPE: base=1M, scale=0.125 (per Gemma 4 spec)
  - Use `ggml_flash_attn_ext` or fallback to manual ops
  - **Test:** Output matches PyTorch full attention

- **P2-T4.3:** Integrate KV cache offload for global layers
  - Global layers load KV from ping buffer (SSD)
  - Write updated KV to pong buffer after attention
  - Swap buffers, prefetch next layer's KV
  - Sync with `KvOffloadManager` lifecycle
  - **Test:** KV values correct before/after swap
  - **Test:** No data corruption from concurrent prefetch

- **P2-T4.4:** Implement shared KV cache aliasing
  - Gemma 4 interleaves local and global layers
  - Later global layers reuse KV from earlier ones (beam search friendly)
  - Implement tensor aliasing (view with same underlying buffer)
  - **Test:** Aliased KV produces same results as fresh compute

- **P2-T4.5:** Dispatch hybrid attention routing
  - Per-layer config: local vs global (from GGUF metadata or config)
  - Router selects appropriate graph builder at inference time
  - **Test:** 26-layer model with interleaved attention layers routes correctly

**Acceptance Criteria:**
- ✅ Local attention output numerically matches PyTorch reference (1e-5 tolerance)
- ✅ Global attention output numerically matches PyTorch reference (1e-5 tolerance)
- ✅ KV offload/swap produces identical results as resident KV
- ✅ Hybrid routing selected correct attention function for each layer

---

### P2-T5: Gemma 4 31B Dense Model Definition + Layer Streaming

**Status:** 🔴 Not Started

**Objective:** Define Gemma 4 31B dense model with layer-by-layer streaming.

**Subtasks:**

- **P2-T5.1:** Define Gemma4DenseConfig
  - Extract from GGUF metadata: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`
  - Config keys for hybrid attention: layer_type per layer, sliding_window size, rope_theta, etc.
  - Extend config for streaming: `max_layers_in_ram` (default 4)
  - **Test:** Config parses correctly from Gemma 4 9B GGUF
  - **Test:** Config extracts all required parameters

- **P2-T5.2:** Implement Gemma4DenseModel Burn module
  - Embedding layer + 46 transformer layers + RMSNorm + lm_head
  - Each layer contains: attention (hybrid) + MLP
  - Integrate `LayerWeightCache` for streaming (N=4 slots)
  - **Test:** Model instantiation succeeds
  - **Test:** Single forward pass completes without OOM

- **P2-T5.3:** Implement GemmaRunner with decode_step
  - Load model weights lazily via `LayerWeightCache`
  - Decode loop:
    - Run layer i forward
    - **Before** layer i+1: call `layer_cache.prefetch([layer i+2, layer i+3])`
    - KV prefetch: fire `kv_offload.prefetch_next()` before global attention layers
  - Explicit prefetch schedule documented in code
  - **Test:** Prefetch fires at expected points during decode

- **P2-T5.4:** Verify layer streaming correctness
  - Compare output: N=4 layer slots vs N=46 (all resident)
  - Both should produce identical logits (within numerical precision)
  - **Test:** 1000-token generation with N=4 matches N=46 (same random seed)

- **P2-T5.5:** Benchmark single-layer inference
  - Measure latency: layer compute + cache load time
  - Target: <50ms per layer at typical hidden_size
  - **Test:** Throughput matches profiling targets

**Acceptance Criteria:**
- ✅ Model loads from Gemma 4 9B GGUF without errors
- ✅ Single forward pass produces correct shapes
- ✅ N=4 streaming produces identical output to N=46 resident
- ✅ Layer prefetch timing verified via instrumentation
- ✅ No OOM with 16 GB RAM + 256K context

---

### P2-T6: End-to-End Benchmark on macOS Apple Silicon

**Status:** 🔴 Not Started

**Objective:** Validate full system on target hardware with performance targets.

**Subtasks:**

- **P2-T6.1:** Set up macOS test environment
  - Target: MacBook Air M-series (M1/M2/M3/M4), 16 GB unified memory
  - Download Gemma 4 31B Q3_K_M GGUF (~12-14 GB)
  - Prepare NVMe SSD scratch directory for KV/layer cache (~100 GB available)
  - **Test:** All files present and accessible

- **P2-T6.2:** Benchmark decode throughput
  - Generate 256 tokens with context length = 4K, 32K, 128K, 256K
  - Measure tokens/second (excluding initial prefill)
  - Record: avg latency, std dev, peak memory
  - Target: ≥2 tok/s at 256K context
  - **Test:** No OOM errors at any context length

- **P2-T6.3:** Benchmark prefill (time-to-first-token)
  - Input: 4K, 32K tokens (static)
  - Measure: time from start until first generated token
  - Breakdown: embed time, first N=4 layers, first attention
  - **Test:** TTFT <2 seconds at 4K context

- **P2-T6.4:** Profile compute/IO overlap
  - Instruments trace: GPU compute vs SSD read activity
  - Verify prefetch completes before compute stalls
  - Measure: overlap efficiency (target >70%)
  - **Test:** Compute overlaps with layer/KV prefetch

- **P2-T6.5:** Compare against llama.cpp ceiling
  - Run identical benchmark with `llama.cpp` binary
  - Measure llama.cpp throughput at same context lengths
  - Compute: `burn-ggml` / `llama.cpp` ratio (target >85%)
  - **Test:** Ratio within 85-95% on M-series GPU

- **P2-T6.6:** Correctness validation
  - Verify greedy decoding determinism (same seed → same output)
  - Test on few-shot prompts from MMLU or similar
  - Spot-check numerical accuracy vs HuggingFace models
  - **Test:** Model follows instructions correctly

- **P2-T6.7:** Generate benchmark report
  - Document: hardware config, model, quantization, context lengths
  - Report: throughput table, profiling results, comparison to llama.cpp
  - Include: memory breakdown (resident vs streamed)
  - **Test:** Report complete and reproducible

**Acceptance Criteria:**
- ✅ Throughput ≥2 tok/s at 256K context on MacBook Air M-series
- ✅ No OOM across all tested context lengths
- ✅ Performance within 85% of llama.cpp direct
- ✅ Compute/IO overlap >70% via profiling
- ✅ Greedy sampling deterministic
- ✅ Benchmark report published with full methodology

---

### P2-T7: Regression Testing & All-Tests Pass

**Status:** 🔴 Not Started

**Objective:** Ensure no regressions in existing tests, and all new tests pass.

**Subtasks:**

- **P2-T7.1:** Run Phase 1 unit tests
  - `cargo test -p burn-ggml --lib -- --test-threads=1`
  - Expected: test_add, test_mul pass; test_matmul remains #[ignore]
  - **Test:** 2 passed, 0 failed, 1 ignored

- **P2-T7.2:** Run Phase 1 integration tests
  - `cargo test -p burn-ggml --test test_backend_full`
  - `cargo test -p burn-ggml --test test_backend_rust`
  - `cargo test -p burn-ggml --test test_prefetch`
  - `cargo test -p burn-ggml --test test_weight_cache`
  - `cargo test -p burn-ggml --test test_wgpu_memory`
  - Expected: all 5 pass
  - **Test:** 5 passed, 0 failed

- **P2-T7.3:** Run CLI inference test
  - Execute: `cargo run -p burn-ggml -- --max-new-tokens 16 "what is 1+1? answer with only the number"`
  - Expected output: `2`
  - **Test:** Output is exactly "2"

- **P2-T7.4:** Add Phase 2 unit tests
  - `test_ggml_quantize`: quantize F32 → Q4_K, dequantize back, compare
  - `test_ggml_matmul`: matmul with quantized weights
  - `test_ggml_attention_local`: sliding window attention vs PyTorch
  - `test_ggml_attention_global`: full attention vs PyTorch
  - `test_gemma_layer_streaming`: N=4 vs N=46 layers identical output
  - **Test:** All Phase 2 tests pass

- **P2-T7.5:** Add Phase 2 integration tests
  - `test_gemma31b_qwen_inference`: load Gemma 4 9B, run forward pass
  - `test_gemma31b_kv_offload`: verify KV swap produces same results
  - `test_gemma31b_decode_determinism`: greedy sampling with seed
  - **Test:** All integration tests pass

- **P2-T7.6:** Memory leak check
  - Run under `valgrind --leak-check=full` on Linux
  - Run under `AddressSanitizer` via `RUSTFLAGS`
  - **Test:** No memory leaks detected

**Acceptance Criteria:**
- ✅ All Phase 1 tests still pass
- ✅ All Phase 2 unit tests pass
- ✅ All Phase 2 integration tests pass
- ✅ CLI command outputs "2" correctly
- ✅ No memory leaks under sanitizers

---

### P2-T8: Documentation and Deliverables

**Status:** 🔴 Not Started

**Objective:** Document implementation for maintainability and reproducibility.

**Subtasks:**

- **P2-T8.1:** Document ggml FFI layer
  - Comment all unsafe blocks with SAFETY notes
  - Explain pointer lifetime requirements
  - List assumptions (context outlives tensors)
  - **Test:** Safety invariants clear to reviewer

- **P2-T8.2:** Document quantization strategy
  - Explain Q4_K_M vs Q3_K_M tradeoff (size vs accuracy)
  - Quantization error tolerance rationale
  - Reference GGML kernel implementations
  - **Test:** Technical documentation complete

- **P2-T8.3:** Document layer streaming cache
  - Explain LRU eviction strategy
  - Profile cache hit rates on typical inputs
  - Prefetch schedule in Gemma4 runner
  - **Test:** Cache behavior transparent to users

- **P2-T8.4:** Document Metal backend
  - Which ops use Metal vs CPU fallback
  - Profiling: Metal overhead vs compute savings
  - Known limitations (e.g., batch size constraints)
  - **Test:** Metal integration clear

- **P2-T8.5:** Update README.md
  - Feature list: quantized inference, layer streaming, KV offload
  - Supported models: Gemma 4 31B
  - Hardware requirements: macOS M1/M2+, 16 GB RAM, NVMe SSD
  - Quick-start example for CLI
  - **Test:** README reflects Phase 2 state

- **P2-T8.6:** Benchmark report
  - Gemma 4 31B Q3_K_M on MacBook Air
  - Throughput: 4K, 32K, 128K, 256K context
  - Comparison: vs llama.cpp direct
  - Profile traces included
  - **Test:** Report reproducible by future developers

- **P2-T8.7:** Changelog
  - Phase 2 features added
  - Compatibility notes
  - Migration guide (if any from Phase 1)
  - **Test:** Changelog complete

**Acceptance Criteria:**
- ✅ All code unsafe blocks have SAFETY comments
- ✅ FFI layer documented with invariants
- ✅ README updated with Phase 2 features
- ✅ Benchmark report published
- ✅ Changelog reflects Phase 2 work

---

## Test Plan Summary

### Unit Tests (Quick, CPU-based)
```
✅ test_ggml_quantize
✅ test_ggml_matmul
✅ test_ggml_attention_local
✅ test_ggml_attention_global
✅ test_gemma_layer_streaming
✅ test_ggml_add, test_ggml_mul (existing)
```

### Integration Tests
```
✅ test_backend_full (Phase 1, continuing)
✅ test_backend_rust (Phase 1, continuing)
✅ test_prefetch (Phase 1, continuing)
✅ test_weight_cache (Phase 1, continuing)
✅ test_wgpu_memory (Phase 1, continuing)
✅ test_gemma31b_qwen_inference (new)
✅ test_gemma31b_kv_offload (new)
✅ test_gemma31b_decode_determinism (new)
```

### CLI Verification
```
✅ cargo run -p burn-ggml -- --max-new-tokens 16 "what is 1+1? answer with only the number"
   Expected: 2
```

### Performance Benchmarks
```
✅ Decode throughput at 4K, 32K, 128K, 256K context
✅ Prefill (TTFT) at 4K, 32K context
✅ Comparison to llama.cpp direct (ceiling)
✅ Profiling: compute/IO overlap >70%
✅ No OOM at 256K context with 16 GB RAM
```

### Memory & Safety
```
✅ Valgrind leak check
✅ AddressSanitizer (ASAN) clean
✅ ThreadSanitizer (TSAN) clean
✅ No unsafe dereferences
```

---

## Implementation Notes

### Known Issues to Address

1. **Matmul Row-Major/Column-Major Mismatch**
   - Phase 1 temporarily disabled test_matmul due to semantic mismatch
   - Must be resolved before Phase 2 can fully validate numerical correctness
   - Suggested fix: ensure GGML tensor dimensions align with Burn row-major layout

2. **Context Buffer Exhaustion**
   - Phase 1 increased context buffer from 100MB to 256MB
   - Phase 2 may need further tuning for 256K context inference
   - Monitor buffer usage under load; increase if needed

3. **Metal Backend Availability**
   - ggml-sys build will fail on non-macOS without graceful fallback
   - Ensure CPU-only mode works as fallback
   - CI/CD must test both Metal (macOS) and CPU (Linux) paths

### Dependencies to Manage

- llama.cpp submodule: pin to specific SHA, document update procedure
- Burn v0.21 pre-release: track changelog for breaking changes
- ggml-backend API stability: wrap in thin C layer to insulate Rust code

### Risk Mitigation

- Start with Gemma 4 9B (small, fast iteration) before 31B
- Profile early and often: identify memory/compute bottlenecks
- Benchmark against llama.cpp at each milestone
- Test on actual M-series hardware (M1 Min, M2 Air, M3/M4 if available)

---

## Success Criteria

**Phase 2 Complete When:**

1. ✅ `cargo run -p burn-ggml -- --max-new-tokens 16 "what is 1+1? answer with only the number"` outputs "2"
2. ✅ All Phase 1 tests still pass (no regressions)
3. ✅ All Phase 2 unit tests pass
4. ✅ Gemma 4 31B Q3_K_M runs on MacBook Air M-series
5. ✅ Throughput ≥2 tok/s at 256K context
6. ✅ Performance within 85% of llama.cpp direct
7. ✅ No memory leaks (valgrind/ASAN clean)
8. ✅ Benchmark report published
9. ✅ All code documented with safety invariants
10. ✅ README reflects Phase 2 state

---

## Milestone Dates (Estimated)

| Milestone | Date | Deliverable |
|-----------|------|-------------|
| P2-T1 Complete | Week 13 | ggml-sys builds, smoke tests pass |
| P2-T2 Complete | Week 14 | Backend skeleton, core ops working |
| P2-T3 Complete | Week 15 | Quantized matmul on Metal |
| P2-T4 Complete | Week 16 | Hybrid attention tested |
| P2-T5 Complete | Week 17 | Gemma 4 31B model, layer streaming |
| P2-T6 Complete | Week 19 | Benchmarks, performance targets met |
| P2-T7 Complete | Week 20 | All tests pass, no regressions |
| **Milestone 2** | **Week 20** | **Full Phase 2 complete** |

---

## Open Questions

1. How should matmul semantics be fixed to align GGML's column-major with Burn's row-major?
2. What is the acceptable numerical tolerance for quantized ops (currently 1e-5)?
3. Should layer streaming cache be configurable per model, or fixed at N=4?
4. How do we handle Gemma 4 variants with different num_hidden_layers (26 vs 28)?
5. Should KV cache quantization be included in Phase 2, or deferred to Phase 3?

---

## References

- Design Document: `./design/ggml-burn-backend.md` (Sections 10.2: Phase 2 details)
- Phase 1 TODO: `./design/PHASE_1_TODO.md`
- llama.cpp: https://github.com/ggerganov/llama.cpp (submodule)
- Burn Framework: https://github.com/tracel-ai/burn
- Gemma 4 Spec: [link to Gemma 4 model card]
