use crate::gguf::{GgufIndex, GgufMetadataValue};
use std::collections::HashMap;

pub struct SimpleTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl SimpleTokenizer {
    pub fn from_index(index: &GgufIndex) -> Self {
        let tokens = index.metadata.get("tokenizer.ggml.tokens")
            .and_then(|v| {
                if let GgufMetadataValue::Array(arr) = v {
                    Some(arr.iter().map(|item| {
                        if let GgufMetadataValue::String(s) = item {
                            s.clone()
                        } else {
                            panic!("Expected string in tokenizer.ggml.tokens")
                        }
                    }).collect::<Vec<_>>())
                } else {
                    None
                }
            }).expect("Failed to find tokenizer.ggml.tokens in metadata");

        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        SimpleTokenizer {
            token_to_id,
            id_to_token: tokens,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Very basic whitespace-based tokenization for now.
        // Real Qwen uses BPE, but this is a placeholder to show the mechanism.
        // Actually, for a real response, we need a real tokenizer.
        // But let's try to match tokens greedily.
        
        let mut ids = Vec::new();
        let mut remaining = text;
        
        while !remaining.is_empty() {
            let mut found = false;
            // Try longest match
            for len in (1..=remaining.len()).rev() {
                let sub = &remaining[..len];
                if let Some(&id) = self.token_to_id.get(sub) {
                    ids.push(id);
                    remaining = &remaining[len..];
                    found = true;
                    break;
                }
            }
            
            if !found {
                // Skip one char if no match (fallback)
                remaining = &remaining[1..];
            }
        }
        
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Handle special tokens/formatting
                let t = token.replace(" ", " "); // Replace GGUF space char
                text.push_str(&t);
            }
        }
        text
    }
}
