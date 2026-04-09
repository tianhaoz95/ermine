# Study: Vulkan Backend Support for burn-ggml on Android

This document explores the feasibility, architecture, and implementation path for supporting the Vulkan backend provided by `llama.cpp` within the `burn-ggml` framework, specifically targeting Android devices.

## 1. Architectural Overview

### 1.1 llama.cpp Vulkan Backend
The Vulkan backend in `llama.cpp` is a high-performance compute backend that uses GLSL compute shaders. It is designed to be cross-platform and has minimal external dependencies beyond the Vulkan loader and a shader compiler (`glslc`).

**Key Components:**
- **Shaders**: Located in `ggml/src/ggml-vulkan/vulkan-shaders/`. These are `.comp` files.
- **Generator**: `vulkan-shaders-gen` is a C++ utility that compiles GLSL to SPIR-V (using `glslc`) and embeds them into C++ sources.
- **Backend Implementation**: `ggml-vulkan.cpp` implements the `ggml_backend` interface for Vulkan.

### 1.2 Android Vulkan Support
Vulkan is a first-class citizen on Android (since Android 6.0/7.0). High-end mobile GPUs (Qualcomm Adreno, ARM Mali) have mature Vulkan drivers.
- **Linking**: On Android, `libvulkan.so` is provided by the system. The Android NDK provides the necessary headers and stub libraries for linking.
- **Shader Compilation**: Shaders must be pre-compiled or compiled at build time. `llama.cpp` handles this via the generator.

## 2. Implementation roadmap for burn-ggml

### 2.1 Extending burn-ggml
To support Vulkan, several layers of the `burn-ggml` crate must be updated:

1.  **`GgmlDevice`**: Add `Vulkan(usize)` (device index) to the device enum in `device.rs`.
2.  **`GgmlContext`**: Update `get_backend` in `context.rs` to call `ggml_backend_vk_init(device_id)`.
3.  **Backend Registry**: Ensure Vulkan devices are properly enumerated and registered.

### 2.2 Updating ggml-sys (FFI bindings)
1.  **`wrapper.h`**: Include `ggml-vulkan.h`.
2.  **`build.rs`**: 
    - Support the `GGML_VULKAN=ON` CMake flag.
    - Handle cross-compilation specific logic:
        - When cross-compiling for Android, `llama.cpp` needs to build `vulkan-shaders-gen` for the **host** architecture.
        - Ensure `glslc` is in the path or provided via `Vulkan_GLSLC_EXECUTABLE`.
    - Link `libvulkan` (on Android, this is usually just `-lvulkan`).

## 3. Build Pipeline for Android

### 3.1 Dependencies
- **Android NDK**: Required for cross-compilation.
- **Vulkan SDK (Host)**: Required for `glslc` and host-side shader generation tools.
- **cargo-ndk**: Recommended for managing the Rust Android build.

### 3.2 Compilation Steps
1.  Set `ANDROID_NDK_HOME`.
2.  Run `cargo ndk -t aarch64-linux-android build --package burn-ggml`.
3.  The `ggml-sys/build.rs` will invoke CMake with the Android toolchain file.
4.  Pass `-DGGML_VULKAN=ON` to the CMake configuration.

## 4. Challenges and Considerations

### 4.1 Shader Generator Cross-compilation
In a cross-compilation environment (building for Android on x86_64), `llama.cpp`'s CMake build needs to compile a tool (`vulkan-shaders-gen`) for the host. This can be tricky with `cmake-rs`. We may need to pass `GGML_VULKAN_SHADERS_GEN_TOOLCHAIN` or ensure the host compiler is correctly detected by `llama.cpp`.

### 4.2 Driver Compatibility
While Vulkan is standard on Android, different GPU vendors have varying levels of support for specific extensions (e.g., `VK_KHR_memory_model`, `VK_KHR_shader_float16_int8`). `llama.cpp` includes runtime checks for these, but performance may vary significantly between Adreno and Mali GPUs.

### 4.3 Memory Management
Android devices often have limited RAM. `burn-ggml`'s use of GGUF and memory mapping (`memmap2`) is already well-suited for this, but Vulkan buffer allocations should be monitored to avoid OOM (Out Of Memory) kills by the Android LMK (Low Memory Killer).

## 5. Conclusion
Vulkan support is highly feasible and would provide a significant performance boost for Burn models on Android, matching the performance of native `llama.cpp` Android applications. The primary technical hurdle is the build system integration, specifically handling host-side tools during cross-compilation.
