use crate::{GgmlBackend, GgmlTensor, GgmlOps};
use crate::model::ModelConfig;
use crate::memory::{LayerWeightCache, LayerKey};
use burn::tensor::{Tensor, Float, Int, TensorPrimitive};
use burn::tensor::ops::{FloatTensorOps, ModuleOps};
use std::sync::Arc;
use std::collections::HashMap;

pub struct QwenModel {
    pub config: ModelConfig,
    pub weights: HashMap<String, GgmlTensor>,
    pub cache: Option<Arc<LayerWeightCache>>,
}

impl QwenModel {
    pub fn new(config: ModelConfig, weights: HashMap<String, GgmlTensor>, cache: Option<Arc<LayerWeightCache>>) -> Self {
        QwenModel { config, weights, cache }
    }

    pub async fn forward(&self, input_ids: Tensor<GgmlBackend, 1, Int>, cur_pos: usize) -> Tensor<GgmlBackend, 1, Float> {
        // 1. Embedding
        let embd_weight = self.weights.get("token_embd.weight").expect("Missing token_embd.weight");
        let mut x = GgmlBackend::embedding(embd_weight.clone(), input_ids.into_primitive());
        
        // 2. Decoder Layers
        for i in 0..self.config.n_layers {
            if i % 4 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }

            // Prefetch next layer
            if let Some(cache) = &self.cache {
                if i + 1 < self.config.n_layers {
                    cache.prefetch(&[LayerKey { layer: i + 1 }]);
                }
            }

            // Get current layer weights
            let layer_tensors = if let Some(cache) = &self.cache {
                let slot = cache.get(LayerKey { layer: i }).await;
                slot.tensors.get().expect("Failed to load layer weights").clone()
            } else {
                // Fallback to pre-loaded weights if no cache
                self.get_layer_weights(i)
            };

            x = self.layer_forward(i, x, &layer_tensors).await;
        }

        // 3. Final Norm
        let final_norm_weight = self.weights.get("output_norm.weight").expect("Missing output_norm.weight");
        x = GgmlBackend::rms_norm(x, final_norm_weight.clone(), self.config.rms_norm_eps);

        // 4. Output Head
        let output_weight = self.weights.get("output.weight").expect("Missing output.weight");
        let logits = GgmlBackend::float_matmul(x, output_weight.clone());
        
        // Return logits for the last token
        let logits_tensor: Tensor<GgmlBackend, 2, Float> = Tensor::from_primitive(TensorPrimitive::Float(logits));
        let last_logits = logits_tensor.slice([cur_pos..cur_pos+1]);
        last_logits.reshape([self.config.vocab_size])
    }

    async fn layer_forward(&self, _layer_idx: usize, x: GgmlTensor, weights: &HashMap<String, GgmlTensor>) -> GgmlTensor {
        // RMSNorm
        let attn_norm_weight = weights.get(&format!("blk.{}.attn_norm.weight", _layer_idx)).expect("Missing attn_norm");
        let norm_x = GgmlBackend::rms_norm(x.clone(), attn_norm_weight.clone(), self.config.rms_norm_eps);

        // Attention (Simplified: just Linear for now to show the loop works)
        let w_q = weights.get(&format!("blk.{}.attn_q.weight", _layer_idx)).expect("Missing q weight");
        let q = GgmlBackend::float_matmul(norm_x, w_q.clone());
        
        // MLP
        let ffn_norm_weight = weights.get(&format!("blk.{}.ffn_norm.weight", _layer_idx)).expect("Missing ffn_norm");
        let norm_x_ffn = GgmlBackend::rms_norm(q, ffn_norm_weight.clone(), self.config.rms_norm_eps);
        
        let w_gate = weights.get(&format!("blk.{}.ffn_gate.weight", _layer_idx)).expect("Missing gate weight");
        let gate = GgmlBackend::float_matmul(norm_x_ffn, w_gate.clone());
        let gate_silu = GgmlBackend::silu(gate);
        
        // Residual
        GgmlBackend::float_add(x, gate_silu)
    }

    fn get_layer_weights(&self, layer_idx: usize) -> HashMap<String, GgmlTensor> {
        let prefix = format!("blk.{}.", layer_idx);
        self.weights.iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}
