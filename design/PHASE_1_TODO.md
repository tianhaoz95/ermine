# Phase 1: Remaining Implementation Tasks

This document tracks the remaining work for **Phase 1 (Linux, wgpu/Vulkan)** as defined in the `burn-ggml` design document. Phase 1 focuses on validating the memory management architecture (Expert/Layer streaming and KV offloading) using `burn-wgpu` before porting to GGML/Metal.

---

## P1-T1 — `PrefetchOps` Trait in Burn
**Status:** 🔴 Not Started
- [ ] Create `burn-backend/src/backend/ops/prefetch.rs`:
  - [ ] Define `PrefetchPrimitive<B>` struct.
  - [ ] Define `PrefetchOps<B>` trait with default no-op.
- [ ] Modify `burn-backend/src/backend/base.rs`:
  - [ ] Add `+ PrefetchOps<Self>` to `Backend` supertrait.
- [ ] Modify `burn-tensor/src/tensor/api/base.rs`:
  - [ ] Implement `Tensor::prefetch(self, device) -> Self`.
- [ ] Verify compilation across all existing backends (`ndarray`, `wgpu`, etc.).

## P1-T2 — `WeightCache<T>` Implementation
**Status:** 🟡 Skeleton Only (See `burn-ggml/src/memory/weight_cache.rs`)
- [ ] **Disk I/O:** Replace simulated loading with real async SSD I/O.
  - [ ] Use `memmap2` for GGUF mapping.
  - [ ] Implement `tokio::fs` pread for cache misses.
- [ ] **LRU Eviction:** Ensure the `LruCache` correctly handles capacity overflows and evicts the least recently used slots.
- [ ] **Backend Integration:** Transition from `Vec<u8>` to backend-specific buffer handles.

## P1-T3 — KV Cache SSD Offload (Ping-Pong)
**Status:** 🔴 Not Started (See `burn-ggml/src/memory/kv_offload.rs`)
- [ ] Implement `KvBuffer` (wgpu buffer wrapper for ping/pong pairs).
- [ ] Implement `KvOffloadManager`:
  - [ ] `swap_and_get(global_layer_idx)` logic.
  - [ ] `prefetch_next(next_global_layer_idx)` spawned via Tokio.
  - [ ] `writeback_async(global_layer_idx, buf)` for deferred SSD persistence.
- [ ] Implement async write-back batching (flush every 16-32 steps).

## P1-T4 — GGUF Model Loader Enhancements
**Status:** 🟡 Partial (See `burn-ggml/src/gguf.rs`)
- [ ] Implement `GgufIndex::expert_offsets(layer_idx)` to return byte ranges for MoE experts.
- [ ] Implement `GgufIndex::layer_offsets()` to return byte ranges for dense layers.
- [ ] Add support for parsing hybrid attention metadata (`sliding_window` vs `global`).

## P1-T5 — WGPU Backend Core Ops & Offload Device
**Status:** 🟡 Partial (See `burn-ggml/src/ops/float_ops.rs`)
- [ ] **Device Definition:** Implement `WgpuOffloadDevice` variant in `GgmlDevice`.
- [ ] **ModuleOps:** Complete `linear_forward`, `embedding_forward`, and `rms_norm_forward`.
- [ ] **Attention Dispatch:** Implement logic to route computation to either local (resident) or global (offloaded) attention.
- [ ] **ActivationOps:** Implement `relu`, `gelu`, and `silu` (critical for Gemma 4).

## P1-T6 — MoE Routing Kernel (WGSL)
**Status:** 🔴 Not Started
- [ ] Write `shaders/moe_router.wgsl`:
  - [ ] Shared-memory parallel reduction for softmax.
  - [ ] Top-8 selection (partial insertion sort or bitonic sort).
- [ ] Implement `Features::SUBGROUP` fast path for Intel iGPUs.
- [ ] Create `MoeRouter` Burn module.

## P1-T7 — `MUL_MAT_ID` Grouped GEMM Kernel (WGSL)
**Status:** 🔴 Not Started
- [ ] Write `shaders/mul_mat_id.wgsl`:
  - [ ] Implement expert-indexed tiled GEMM.
  - [ ] Support push constants for `M, N, K, expert_stride`.
- [ ] Implement `MulMatId` op in `ModuleOps`.

## P1-T8 — Gemma 4 26B MoE Model Definition
**Status:** 🔴 Not Started
- [ ] Define `Gemma4MoeConfig` (extracted from GGUF metadata).
- [ ] Implement `Gemma4MoeLayer` (Attention + MoE FFN).
- [ ] Implement `GemmaRunner::decode_step` with the explicit prefetch schedule:
  ```rust
  // Fire prefetch immediately after routing
  expert_cache.prefetch(top_k_indices);
  // Fire KV prefetch before the next global layer
  kv_offload.prefetch_next(next_global);
  ```

## P1-T9 — End-to-End Validation
**Status:** 🔴 Not Started
- [ ] **Correctness:** Pass `factual::capital_of_china` and `instruction::arithmetic_strict` (deterministic greedy).
- [ ] **Offload Validation:** Ensure output with N=4 slots matches output with all layers resident.
- [ ] **Performance:** Benchmark on Intel iGPU (Target: >=1.5 tok/s at 32K context).
