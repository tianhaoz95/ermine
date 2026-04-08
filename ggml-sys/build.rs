use std::path::PathBuf;

fn main() {
    let llama_dir = PathBuf::from("../llama.cpp");

    // 1. Build ggml + CPU backend via cmake
    let mut config = cmake::Config::new(&llama_dir);
    
    config
        .define("LLAMA_STATIC", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF");

    if cfg!(target_os = "macos") {
        config.define("LLAMA_METAL", "ON");
    }

    let dst = config.build();

    // 2. Tell cargo to link the static libraries
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/build/src", dst.display());
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());
    println!("cargo:rustc-link-search=native={}/build/common", dst.display());

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=common");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // On macOS: link Metal framework
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // Link C++ standard library and OpenMP
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    }

    // 3. Generate bindings with bindgen
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llama_dir.join("ggml/include").display()))
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_.*")
        .allowlist_function("ggml_backend_metal_.*")
        .allowlist_function("ggml_backend_cpu_.*")
        .blocklist_function("ggml_internal_.*")
        .generate_comments(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Failed to generate ggml bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ggml_bindings.rs"))
        .expect("Couldn't write bindings");

    println!("cargo:rerun-if-changed=../llama.cpp/ggml/include/ggml.h");
    println!("cargo:rerun-if-changed=../llama.cpp/ggml/include/ggml-backend.h");
    println!("cargo:rerun-if-changed=../llama.cpp/ggml/include/ggml-alloc.h");
    println!("cargo:rerun-if-changed=wrapper.h");
}
