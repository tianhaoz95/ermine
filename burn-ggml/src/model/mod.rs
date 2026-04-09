pub mod qwen;

use crate::gguf::{GgufIndex, GgufMetadataValue};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_heads_kv: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn from_index(index: &GgufIndex) -> Self {
        let arch = match index.metadata.get("general.architecture").expect("Missing general.architecture") {
            GgufMetadataValue::String(s) => s.clone(),
            _ => panic!("Expected string for general.architecture"),
        };

        let get_u32 = |key: &str| -> usize {
            let full_key = format!("{}.{}", arch, key);
            index.metadata.get(&full_key).or_else(|| index.metadata.get(&format!("llama.{}", key)))
                .map(|v| match v {
                    GgufMetadataValue::Uint32(u) => *u as usize,
                    GgufMetadataValue::Uint64(u) => *u as usize,
                    _ => panic!("Expected uint for {}", full_key),
                }).unwrap_or_else(|| {
                    panic!("Missing metadata key: {} or llama.{}", full_key, key);
                })
        };

        let get_f32 = |key: &str| -> f32 {
            let full_key = format!("{}.{}", arch, key);
            index.metadata.get(&full_key).or_else(|| index.metadata.get(&format!("llama.{}", key)))
                .map(|v| match v {
                    GgufMetadataValue::Float32(f) => *f,
                    _ => panic!("Expected float for {}", full_key),
                }).unwrap_or_else(|| {
                    panic!("Missing metadata key: {} or llama.{}", full_key, key);
                })
        };

        ModelConfig {
            n_layers: get_u32("block_count"),
            d_model: get_u32("embedding_length"),
            d_ff: get_u32("feed_forward_length"),
            n_heads: get_u32("attention.head_count"),
            n_heads_kv: get_u32("attention.head_count_kv"),
            rms_norm_eps: get_f32("attention.layer_norm_rms_epsilon"),
            rope_theta: index.metadata.get(&format!("{}.rope.freq_base", arch))
                .or_else(|| index.metadata.get("llama.rope.freq_base"))
                .map(|v| match v {
                    GgufMetadataValue::Float32(f) => *f,
                    _ => 10000.0,
                }).unwrap_or(10000.0),
            vocab_size: index.metadata.get("tokenizer.ggml.tokens").map(|v| {
                if let GgufMetadataValue::Array(arr) = v { arr.len() } else { 0 }
            }).unwrap_or(0),
        }
    }
}
