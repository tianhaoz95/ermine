use crate::context::GgmlContext;
use crate::device::GgmlDevice;
use crate::tensor::GgmlTensor;
use crate::GgmlBackend;
use burn::tensor::ops::{FloatTensorOps, IntTensorOps, ModuleOps};
use burn::tensor::{Shape, TensorData};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Qwen35RunnerOptions {
    pub max_new_tokens: usize,
    pub max_layers_in_ram: usize,
}

impl Default for Qwen35RunnerOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 8,
            max_layers_in_ram: 24,
        }
    }
}

pub struct Qwen35Runner {
    device: GgmlDevice,
    config: Qwen35TextConfig,
    tokenizer: PythonTokenizer,
    global: GlobalWeights,
    layers: Vec<LayerWeights>,
}

impl Qwen35Runner {
    pub fn load(model_dir: &Path, options: &Qwen35RunnerOptions) -> Result<Self, String> {
        let config = Qwen35TextConfig::from_dir(model_dir)?;
        let tokenizer = PythonTokenizer::new(model_dir.join("tokenizer.json"));
        let device = GgmlDevice::MetalWithOffload {
            kv_cache_dir: model_dir.join("kv_cache"),
            max_layers_in_ram: options.max_layers_in_ram,
        };
        let store = Arc::new(SafetensorStore::open(model_dir.join("model.safetensors"))?);

        let global = GlobalWeights::load(store.clone(), &device)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(LayerWeights::load(store.clone(), &device, &config, layer_idx)?);
        }

        Ok(Self {
            device,
            config,
            tokenizer,
            global,
            layers,
        })
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String, String> {
        let formatted_prompt = format_generation_prompt(prompt);
        let mut token_ids = self.tokenizer.encode(&formatted_prompt)?;
        let prompt_len = token_ids.len();
        if token_ids.is_empty() {
            return Err("prompt tokenized to an empty sequence".to_string());
        }

        for _ in 0..max_new_tokens {
            let logits = self.forward(&token_ids)?;
            let next_token = argmax(&logits).ok_or("empty logits".to_string())? as u32;
            token_ids.push(next_token);
            if next_token == self.config.eos_token_id {
                break;
            }
            let cleaned = strip_think_tags(&self.tokenizer.decode(&token_ids[prompt_len..])?)
                .trim()
                .to_string();
            if should_stop_early(prompt, &cleaned) {
                break;
            }
        }

        let generated = token_ids[prompt_len..].to_vec();
        let decoded = self.tokenizer.decode(&generated)?;
        Ok(strip_think_tags(&decoded).trim().to_string())
    }

    fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let seq_len = token_ids.len();
        let mut hidden = ggml_embedding(&self.device, &self.global.embed_tokens, token_ids)?;

        let rotary_dim = self.config.rotary_dim();
        let (rope_cos, rope_sin) = build_rope_cache(seq_len, rotary_dim, self.config.rope_theta());

        for (idx, layer) in self.layers.iter().enumerate() {
            let input_norm = qwen_rms_norm(
                &hidden,
                seq_len,
                self.config.hidden_size,
                layer.input_layernorm_weight(),
                self.config.rms_norm_eps,
            );

            let attn_out = match layer {
                LayerWeights::Linear(linear) => self.linear_attention(
                    &input_norm, seq_len, linear,
                )?,
                LayerWeights::Full(full) => self.full_attention(
                    &input_norm,
                    seq_len,
                    full,
                    &rope_cos,
                    &rope_sin,
                )?,
            };

            add_in_place(&mut hidden, &attn_out);

            let post_norm = qwen_rms_norm(
                &hidden,
                seq_len,
                self.config.hidden_size,
                layer.post_attention_layernorm_weight(),
                self.config.rms_norm_eps,
            );

            let gate = ggml_linear(
                &self.device,
                &post_norm,
                seq_len,
                self.config.hidden_size,
                layer.gate_proj_weight(),
            )?;
            let up = ggml_linear(
                &self.device,
                &post_norm,
                seq_len,
                self.config.hidden_size,
                layer.up_proj_weight(),
            )?;
            let mut mlp_hidden = vec![0.0; gate.len()];
            for i in 0..gate.len() {
                mlp_hidden[i] = silu(gate[i]) * up[i];
            }
            let mlp_out = ggml_linear(
                &self.device,
                &mlp_hidden,
                seq_len,
                self.config.intermediate_size,
                layer.down_proj_weight(),
            )?;
            add_in_place(&mut hidden, &mlp_out);

            if idx + 1 < self.layers.len() {
                let _ = idx; // Placeholder to keep the loop structure explicit for future prefetch.
            }
        }

        let final_hidden = qwen_rms_norm(
            &hidden,
            seq_len,
            self.config.hidden_size,
            &self.global.norm_weight,
            self.config.rms_norm_eps,
        );
        let last = &final_hidden[(seq_len - 1) * self.config.hidden_size..seq_len * self.config.hidden_size];
        ggml_linear(
            &self.device,
            last,
            1,
            self.config.hidden_size,
            &self.global.lm_head_t,
        )
    }

    fn full_attention(
        &self,
        hidden: &[f32],
        seq_len: usize,
        layer: &FullAttentionLayer,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Result<Vec<f32>, String> {
        let q = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.q_proj_weight,
        )?;
        let k = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.k_proj_weight,
        )?;
        let v = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.v_proj_weight,
        )?;

        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let (mut q_states, gate) =
            split_qwen_attention_q_gate(&q, seq_len, self.config.num_attention_heads, self.config.head_dim);
        let mut k_states = k;
        let v_states = v;

        reshape_head_major(
            &mut q_states,
            seq_len,
            self.config.num_attention_heads,
            self.config.head_dim,
        );
        reshape_head_major(
            &mut k_states,
            seq_len,
            self.config.num_key_value_heads,
            self.config.head_dim,
        );

        q_states = qwen_rms_norm(
            &q_states,
            seq_len * self.config.num_attention_heads,
            self.config.head_dim,
            &layer.q_norm_weight,
            self.config.rms_norm_eps,
        );
        k_states = qwen_rms_norm(
            &k_states,
            seq_len * self.config.num_key_value_heads,
            self.config.head_dim,
            &layer.k_norm_weight,
            self.config.rms_norm_eps,
        );

        apply_rope_in_place(
            &mut q_states,
            seq_len,
            self.config.num_attention_heads,
            self.config.head_dim,
            self.config.rotary_dim(),
            rope_cos,
            rope_sin,
        );
        apply_rope_in_place(
            &mut k_states,
            seq_len,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.rotary_dim(),
            rope_cos,
            rope_sin,
        );

        let repeated_k = repeat_kv(
            &k_states,
            seq_len,
            self.config.num_key_value_heads,
            self.config.num_attention_heads / self.config.num_key_value_heads,
            self.config.head_dim,
        );
        let repeated_v = repeat_kv(
            &v_states,
            seq_len,
            self.config.num_key_value_heads,
            self.config.num_attention_heads / self.config.num_key_value_heads,
            self.config.head_dim,
        );

        let scale = (self.config.head_dim as f32).powf(-0.5);
        let mut attn_out = vec![0.0; seq_len * self.config.num_attention_heads * self.config.head_dim];
        for t in 0..seq_len {
            for h in 0..self.config.num_attention_heads {
                let q_offset = (t * self.config.num_attention_heads + h) * self.config.head_dim;
                let q_slice = &q_states[q_offset..q_offset + self.config.head_dim];

                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for j in 0..=t {
                    let k_offset = (j * self.config.num_attention_heads + h) * self.config.head_dim;
                    let k_slice = &repeated_k[k_offset..k_offset + self.config.head_dim];
                    scores[j] = dot(q_slice, k_slice) * scale;
                }

                let probs = softmax_prefix(&scores, t + 1);
                let out_offset = (t * self.config.num_attention_heads + h) * self.config.head_dim;
                let out_slice = &mut attn_out[out_offset..out_offset + self.config.head_dim];
                for j in 0..=t {
                    let v_offset = (j * self.config.num_attention_heads + h) * self.config.head_dim;
                    let v_slice = &repeated_v[v_offset..v_offset + self.config.head_dim];
                    for d in 0..self.config.head_dim {
                        out_slice[d] += probs[j] * v_slice[d];
                    }
                }
            }
        }

        let mut flat = vec![0.0; seq_len * q_dim];
        for t in 0..seq_len {
            for h in 0..self.config.num_attention_heads {
                let src = (t * self.config.num_attention_heads + h) * self.config.head_dim;
                let dst = t * q_dim + h * self.config.head_dim;
                flat[dst..dst + self.config.head_dim]
                    .copy_from_slice(&attn_out[src..src + self.config.head_dim]);
            }
        }
        for i in 0..flat.len() {
            flat[i] *= sigmoid(gate[i]);
        }

        ggml_linear(
            &self.device,
            &flat,
            seq_len,
            q_dim,
            &layer.o_proj_weight,
        )
    }

    fn linear_attention(
        &self,
        hidden: &[f32],
        seq_len: usize,
        layer: &LinearAttentionLayer,
    ) -> Result<Vec<f32>, String> {
        let qkv = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.in_proj_qkv_weight,
        )?;
        let z = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.in_proj_z_weight,
        )?;
        let b = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.in_proj_b_weight,
        )?;
        let a = ggml_linear(
            &self.device,
            hidden,
            seq_len,
            self.config.hidden_size,
            &layer.in_proj_a_weight,
        )?;

        let total_q = self.config.linear_num_key_heads * self.config.linear_key_head_dim;
        let total_k = total_q;
        let total_v = self.config.linear_num_value_heads * self.config.linear_value_head_dim;

        let qkv_conv = causal_depthwise_conv1d_silu(&qkv, seq_len, total_q + total_k + total_v, &layer.conv1d_weight);
        let (query_states, key_states, value_states) =
            split_rows_3(&qkv_conv, seq_len, total_q, total_k, total_v);

        let beta = apply_unary(&b, sigmoid);
        let mut g = vec![0.0; a.len()];
        for t in 0..seq_len {
            for h in 0..self.config.linear_num_value_heads {
                let idx = t * self.config.linear_num_value_heads + h;
                g[idx] = -(layer.a_log[h].exp()) * softplus(a[idx] + layer.dt_bias[h]);
            }
        }

        let mut q_norm = vec![0.0; query_states.len()];
        let mut k_norm = vec![0.0; key_states.len()];
        for t in 0..seq_len {
            for h in 0..self.config.linear_num_key_heads {
                let start = (t * self.config.linear_num_key_heads + h) * self.config.linear_key_head_dim;
                let end = start + self.config.linear_key_head_dim;
                let src_q = &query_states[start..end];
                let src_k = &key_states[start..end];
                let dst_q = &mut q_norm[start..end];
                let dst_k = &mut k_norm[start..end];
                let qn = l2_norm(src_q).max(1e-6);
                let kn = l2_norm(src_k).max(1e-6);
                for i in 0..self.config.linear_key_head_dim {
                    dst_q[i] = src_q[i] / qn;
                    dst_k[i] = src_k[i] / kn;
                }
            }
        }

        let mut output = vec![0.0; seq_len * total_v];
        let state_stride = self.config.linear_key_head_dim * self.config.linear_value_head_dim;
        let mut state = vec![0.0; self.config.linear_num_value_heads * state_stride];
        let scale = (self.config.linear_key_head_dim as f32).sqrt().recip();

        for t in 0..seq_len {
            for h in 0..self.config.linear_num_value_heads {
                let q_start = (t * self.config.linear_num_key_heads + h) * self.config.linear_key_head_dim;
                let k_start = q_start;
                let v_start = (t * self.config.linear_num_value_heads + h) * self.config.linear_value_head_dim;
                let q_slice = &q_norm[q_start..q_start + self.config.linear_key_head_dim];
                let k_slice = &k_norm[k_start..k_start + self.config.linear_key_head_dim];
                let v_slice = &value_states[v_start..v_start + self.config.linear_value_head_dim];
                let beta_value = beta[t * self.config.linear_num_value_heads + h];
                let g_value = g[t * self.config.linear_num_value_heads + h];

                let state_offset = h * state_stride;
                let state_slice = &mut state[state_offset..state_offset + state_stride];

                let decay = g_value.exp();
                for value in state_slice.iter_mut() {
                    *value *= decay;
                }

                let mut kv_mem = vec![0.0; self.config.linear_value_head_dim];
                for v_idx in 0..self.config.linear_value_head_dim {
                    let mut acc = 0.0;
                    for k_idx in 0..self.config.linear_key_head_dim {
                        acc += state_slice[k_idx * self.config.linear_value_head_dim + v_idx] * k_slice[k_idx];
                    }
                    kv_mem[v_idx] = acc;
                }

                let mut delta = vec![0.0; self.config.linear_value_head_dim];
                for i in 0..self.config.linear_value_head_dim {
                    delta[i] = (v_slice[i] - kv_mem[i]) * beta_value;
                }

                for k_idx in 0..self.config.linear_key_head_dim {
                    for v_idx in 0..self.config.linear_value_head_dim {
                        let idx = k_idx * self.config.linear_value_head_dim + v_idx;
                        state_slice[idx] += k_slice[k_idx] * delta[v_idx];
                    }
                }

                let out_offset = (t * self.config.linear_num_value_heads + h) * self.config.linear_value_head_dim;
                for v_idx in 0..self.config.linear_value_head_dim {
                    let mut acc = 0.0;
                    for k_idx in 0..self.config.linear_key_head_dim {
                        acc += state_slice[k_idx * self.config.linear_value_head_dim + v_idx]
                            * (q_slice[k_idx] * scale);
                    }
                    output[out_offset + v_idx] = acc;
                }
            }
        }

        let normalized = qwen_gated_rms_norm(
            &output,
            seq_len * self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
            &layer.norm_weight,
            &z,
            self.config.rms_norm_eps,
        );

        ggml_linear(
            &self.device,
            &normalized,
            seq_len,
            total_v,
            &layer.out_proj_weight,
        )
    }
}

#[derive(Debug, Deserialize)]
struct RootConfig {
    text_config: Qwen35TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
}

impl Qwen35TextConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|err| format!("failed to read {}: {err}", config_path.display()))?;
        let root: RootConfig = serde_json::from_str(&raw)
            .map_err(|err| format!("failed to parse {}: {err}", config_path.display()))?;
        Ok(root.text_config)
    }

    fn rotary_dim(&self) -> usize {
        ((self.head_dim as f32) * self.rope_parameters.partial_rotary_factor) as usize
    }

    fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }
}

struct PythonTokenizer {
    tokenizer_path: PathBuf,
}

impl PythonTokenizer {
    fn new(tokenizer_path: PathBuf) -> Self {
        Self { tokenizer_path }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let output = Command::new("python3")
            .arg("-c")
            .arg("import json,sys; from tokenizers import Tokenizer; t=Tokenizer.from_file(sys.argv[1]); print(json.dumps(t.encode(sys.argv[2]).ids))")
            .arg(&self.tokenizer_path)
            .arg(text)
            .output()
            .map_err(|err| format!("failed to run python tokenizer encode: {err}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        serde_json::from_slice(&output.stdout)
            .map_err(|err| format!("failed to parse tokenizer encode output: {err}"))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let ids_json = serde_json::to_string(ids)
            .map_err(|err| format!("failed to serialize token ids: {err}"))?;
        let output = Command::new("python3")
            .arg("-c")
            .arg("import json,sys; from tokenizers import Tokenizer; t=Tokenizer.from_file(sys.argv[1]); print(t.decode(json.loads(sys.argv[2])))")
            .arg(&self.tokenizer_path)
            .arg(ids_json)
            .output()
            .map_err(|err| format!("failed to run python tokenizer decode: {err}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

struct SafetensorStore {
    mmap: Mmap,
}

impl SafetensorStore {
    fn open(path: PathBuf) -> Result<Self, String> {
        let file = File::open(&path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|err| format!("failed to mmap {}: {err}", path.display()))? };
        Ok(Self { mmap })
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let tensors = SafeTensors::deserialize(self.mmap.as_ref())
            .map_err(|err| format!("failed to parse safetensors file: {err}"))?;
        let view = tensors
            .tensor(name)
            .map_err(|err| format!("missing tensor {name}: {err}"))?;
        let shape = view.shape().to_vec();
        let data = decode_to_f32(view.dtype(), view.data())?;
        Ok((data, shape))
    }

    fn load_ggml(
        &self,
        name: &str,
        ctx: &Arc<GgmlContext>,
    ) -> Result<GgmlTensor, String> {
        let tensors = SafeTensors::deserialize(self.mmap.as_ref())
            .map_err(|err| format!("failed to parse safetensors file: {err}"))?;
        let view = tensors
            .tensor(name)
            .map_err(|err| format!("missing tensor {name}: {err}"))?;
        tensor_from_raw_bytes(ctx.clone(), view.shape(), view.dtype(), view.data())
    }
}

struct GlobalWeights {
    _ctx: Arc<GgmlContext>,
    embed_tokens: GgmlTensor,
    lm_head_t: GgmlTensor,
    norm_weight: Vec<f32>,
}

impl GlobalWeights {
    fn load(store: Arc<SafetensorStore>, device: &GgmlDevice) -> Result<Self, String> {
        let ctx = GgmlContext::get(device).new_work_context();
        let embed_tokens = store.load_ggml("model.language_model.embed_tokens.weight", &ctx)?;
        let lm_head_t = embed_tokens.clone();
        let (norm_weight, _) = store.read_f32("model.language_model.norm.weight")?;
        Ok(Self {
            _ctx: ctx,
            embed_tokens,
            lm_head_t,
            norm_weight,
        })
    }
}

enum LayerWeights {
    Linear(LinearAttentionLayer),
    Full(FullAttentionLayer),
}

impl LayerWeights {
    fn load(
        store: Arc<SafetensorStore>,
        device: &GgmlDevice,
        config: &Qwen35TextConfig,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.language_model.layers.{layer_idx}");
        let input_layernorm_weight = store.read_f32(&format!("{prefix}.input_layernorm.weight"))?.0;
        let post_attention_layernorm_weight =
            store.read_f32(&format!("{prefix}.post_attention_layernorm.weight"))?.0;
        let mlp_ctx = GgmlContext::get(device).new_work_context();
        let gate_proj_weight = store.load_ggml(&format!("{prefix}.mlp.gate_proj.weight"), &mlp_ctx)?;
        let up_proj_weight = store.load_ggml(&format!("{prefix}.mlp.up_proj.weight"), &mlp_ctx)?;
        let down_proj_weight = store.load_ggml(&format!("{prefix}.mlp.down_proj.weight"), &mlp_ctx)?;

        if config.layer_types[layer_idx] == "full_attention" {
            let attn_ctx = GgmlContext::get(device).new_work_context();
            Ok(Self::Full(FullAttentionLayer {
                _attn_ctx: attn_ctx.clone(),
                _mlp_ctx: mlp_ctx,
                input_layernorm_weight,
                post_attention_layernorm_weight,
                q_proj_weight: store.load_ggml(&format!("{prefix}.self_attn.q_proj.weight"), &attn_ctx)?,
                k_proj_weight: store.load_ggml(&format!("{prefix}.self_attn.k_proj.weight"), &attn_ctx)?,
                v_proj_weight: store.load_ggml(&format!("{prefix}.self_attn.v_proj.weight"), &attn_ctx)?,
                o_proj_weight: store.load_ggml(&format!("{prefix}.self_attn.o_proj.weight"), &attn_ctx)?,
                q_norm_weight: store.read_f32(&format!("{prefix}.self_attn.q_norm.weight"))?.0,
                k_norm_weight: store.read_f32(&format!("{prefix}.self_attn.k_norm.weight"))?.0,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            }))
        } else {
            let attn_ctx = GgmlContext::get(device).new_work_context();
            Ok(Self::Linear(LinearAttentionLayer {
                _attn_ctx: attn_ctx.clone(),
                _mlp_ctx: mlp_ctx,
                input_layernorm_weight,
                post_attention_layernorm_weight,
                in_proj_qkv_weight: store.load_ggml(&format!("{prefix}.linear_attn.in_proj_qkv.weight"), &attn_ctx)?,
                in_proj_z_weight: store.load_ggml(&format!("{prefix}.linear_attn.in_proj_z.weight"), &attn_ctx)?,
                in_proj_b_weight: store.load_ggml(&format!("{prefix}.linear_attn.in_proj_b.weight"), &attn_ctx)?,
                in_proj_a_weight: store.load_ggml(&format!("{prefix}.linear_attn.in_proj_a.weight"), &attn_ctx)?,
                out_proj_weight: store.load_ggml(&format!("{prefix}.linear_attn.out_proj.weight"), &attn_ctx)?,
                norm_weight: store.read_f32(&format!("{prefix}.linear_attn.norm.weight"))?.0,
                conv1d_weight: squeeze_conv_weight(store.read_f32(&format!("{prefix}.linear_attn.conv1d.weight"))?),
                dt_bias: store.read_f32(&format!("{prefix}.linear_attn.dt_bias"))?.0,
                a_log: store.read_f32(&format!("{prefix}.linear_attn.A_log"))?.0,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            }))
        }
    }

    fn post_attention_layernorm_weight(&self) -> &[f32] {
        match self {
            Self::Linear(layer) => &layer.post_attention_layernorm_weight,
            Self::Full(layer) => &layer.post_attention_layernorm_weight,
        }
    }

    fn input_layernorm_weight(&self) -> &[f32] {
        match self {
            Self::Linear(layer) => &layer.input_layernorm_weight,
            Self::Full(layer) => &layer.input_layernorm_weight,
        }
    }

    fn gate_proj_weight(&self) -> &GgmlTensor {
        match self {
            Self::Linear(layer) => &layer.gate_proj_weight,
            Self::Full(layer) => &layer.gate_proj_weight,
        }
    }

    fn up_proj_weight(&self) -> &GgmlTensor {
        match self {
            Self::Linear(layer) => &layer.up_proj_weight,
            Self::Full(layer) => &layer.up_proj_weight,
        }
    }

    fn down_proj_weight(&self) -> &GgmlTensor {
        match self {
            Self::Linear(layer) => &layer.down_proj_weight,
            Self::Full(layer) => &layer.down_proj_weight,
        }
    }
}

struct FullAttentionLayer {
    _attn_ctx: Arc<GgmlContext>,
    _mlp_ctx: Arc<GgmlContext>,
    input_layernorm_weight: Vec<f32>,
    post_attention_layernorm_weight: Vec<f32>,
    q_proj_weight: GgmlTensor,
    k_proj_weight: GgmlTensor,
    v_proj_weight: GgmlTensor,
    o_proj_weight: GgmlTensor,
    q_norm_weight: Vec<f32>,
    k_norm_weight: Vec<f32>,
    gate_proj_weight: GgmlTensor,
    up_proj_weight: GgmlTensor,
    down_proj_weight: GgmlTensor,
}

struct LinearAttentionLayer {
    _attn_ctx: Arc<GgmlContext>,
    _mlp_ctx: Arc<GgmlContext>,
    input_layernorm_weight: Vec<f32>,
    post_attention_layernorm_weight: Vec<f32>,
    in_proj_qkv_weight: GgmlTensor,
    in_proj_z_weight: GgmlTensor,
    in_proj_b_weight: GgmlTensor,
    in_proj_a_weight: GgmlTensor,
    out_proj_weight: GgmlTensor,
    norm_weight: Vec<f32>,
    conv1d_weight: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    gate_proj_weight: GgmlTensor,
    up_proj_weight: GgmlTensor,
    down_proj_weight: GgmlTensor,
}

fn decode_to_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|chunk| f16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|chunk| bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect()),
        other => Err(format!("unsupported safetensors dtype: {other:?}")),
    }
}

fn tensor_from_raw_bytes(
    ctx: Arc<GgmlContext>,
    shape: &[usize],
    dtype: Dtype,
    bytes: &[u8],
) -> Result<GgmlTensor, String> {
    let mut dims = [1i64; 4];
    for (i, &d) in shape.iter().rev().enumerate() {
        dims[i] = d as i64;
    }
    let ggml_dtype = match dtype {
        Dtype::BF16 => ggml_sys::ggml_type_GGML_TYPE_BF16,
        Dtype::F16 => ggml_sys::ggml_type_GGML_TYPE_F16,
        Dtype::F32 => ggml_sys::ggml_type_GGML_TYPE_F32,
        other => return Err(format!("unsupported tensor dtype for ggml load: {other:?}")),
    };
    unsafe {
        let ptr = ggml_sys::ggml_new_tensor(
            ctx.ptr,
            ggml_dtype,
            shape.len() as i32,
            dims.as_ptr(),
        );
        if ptr.is_null() {
            return Err(format!("failed to allocate tensor with shape {shape:?}"));
        }
        ggml_sys::ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
        ggml_sys::ggml_backend_tensor_set(
            ptr,
            bytes.as_ptr() as *const std::ffi::c_void,
            0,
            bytes.len(),
        );
        Ok(GgmlTensor::from_raw(ptr, ctx))
    }
}

fn ggml_embedding(device: &GgmlDevice, weight: &GgmlTensor, ids: &[u32]) -> Result<Vec<f32>, String> {
    let ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
    let ids_len = ids.len();
    let indices = GgmlBackend::int_from_data(TensorData::new(ids, Shape::new([ids_len])), device);
    let out = GgmlBackend::embedding(weight.clone(), indices);
    let data = futures::executor::block_on(GgmlBackend::float_into_data(out))
        .map_err(|err| format!("failed to fetch embedding output: {err}"))?;
    Ok(data.as_slice::<f32>().unwrap().to_vec())
}

fn ggml_linear(
    device: &GgmlDevice,
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &GgmlTensor,
) -> Result<Vec<f32>, String> {
    if weight.shape.len() != 2 {
        return Err(format!("expected 2D weight, got shape {:?}", weight.shape));
    }
    if weight.shape[1] != cols {
        return Err(format!(
            "matmul shape mismatch: input [{} x {}], weight {:?}",
            rows, cols, weight.shape
        ));
    }
    let tensor =
        GgmlBackend::float_from_data(TensorData::new(input.to_vec(), Shape::new([rows, cols])), device);
    let out = GgmlBackend::float_matmul(tensor, weight.clone());
    let data = futures::executor::block_on(GgmlBackend::float_into_data(out))
        .map_err(|err| format!("failed to fetch matmul output: {err}"))?;
    Ok(data.as_slice::<f32>().unwrap().to_vec())
}

fn qwen_rms_norm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; input.len()];
    for row in 0..rows {
        let start = row * cols;
        let slice = &input[start..start + cols];
        let variance = slice.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for col in 0..cols {
            out[start + col] = slice[col] * scale * (1.0 + weight[col]);
        }
    }
    out
}

fn qwen_gated_rms_norm(
    input: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    gate: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; input.len()];
    for row in 0..rows {
        let start = row * cols;
        let slice = &input[start..start + cols];
        let variance = slice.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let scale = 1.0 / (variance + eps).sqrt();
        for col in 0..cols {
            out[start + col] = slice[col] * scale * weight[col] * silu(gate[start + col]);
        }
    }
    out
}

fn build_rope_cache(seq_len: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0; seq_len * rotary_dim];
    let mut sin = vec![0.0; seq_len * rotary_dim];
    let half = rotary_dim / 2;
    for pos in 0..seq_len {
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32);
            let angle = pos as f32 * inv_freq;
            cos[pos * rotary_dim + i] = angle.cos();
            cos[pos * rotary_dim + half + i] = angle.cos();
            sin[pos * rotary_dim + i] = angle.sin();
            sin[pos * rotary_dim + half + i] = angle.sin();
        }
    }
    (cos, sin)
}

fn apply_rope_in_place(
    data: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let base = (pos * num_heads + head) * head_dim;
            let half = rotary_dim / 2;
            for i in 0..half {
                let x0 = data[base + i];
                let x1 = data[base + half + i];
                let c = cos[pos * rotary_dim + i];
                let s = sin[pos * rotary_dim + i];
                data[base + i] = x0 * c - x1 * s;
                data[base + half + i] = x0 * s + x1 * c;
            }
        }
    }
}

fn reshape_head_major(data: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize) {
    let copy = data.to_vec();
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let src = pos * num_heads * head_dim + head * head_dim;
            let dst = (pos * num_heads + head) * head_dim;
            data[dst..dst + head_dim].copy_from_slice(&copy[src..src + head_dim]);
        }
    }
}

fn repeat_kv(
    input: &[f32],
    seq_len: usize,
    kv_heads: usize,
    repeats: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; seq_len * kv_heads * repeats * head_dim];
    for pos in 0..seq_len {
        for head in 0..kv_heads {
            let src = (pos * kv_heads + head) * head_dim;
            for r in 0..repeats {
                let dst = (pos * kv_heads * repeats + head * repeats + r) * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
            }
        }
    }
    out
}

fn causal_depthwise_conv1d_silu(
    input: &[f32],
    seq_len: usize,
    channels: usize,
    weights: &[f32],
) -> Vec<f32> {
    let kernel = weights.len() / channels;
    let mut out = vec![0.0; input.len()];
    for t in 0..seq_len {
        for c in 0..channels {
            let mut sum = 0.0;
            for k in 0..kernel {
                if t + k + 1 >= kernel {
                    let src_t = t + k + 1 - kernel;
                    sum += input[src_t * channels + c] * weights[c * kernel + k];
                }
            }
            out[t * channels + c] = silu(sum);
        }
    }
    out
}

fn squeeze_conv_weight((data, shape): (Vec<f32>, Vec<usize>)) -> Vec<f32> {
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[1], 1);
    let channels = shape[0];
    let kernel = shape[2];
    let mut out = vec![0.0; channels * kernel];
    for c in 0..channels {
        for k in 0..kernel {
            out[c * kernel + k] = data[c * kernel + k];
        }
    }
    out
}

fn add_in_place(lhs: &mut [f32], rhs: &[f32]) {
    for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
        *l += *r;
    }
}

fn split_qwen_attention_q_gate(
    input: &[f32],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let row_width = num_heads * head_dim * 2;
    let mut query = vec![0.0; rows * num_heads * head_dim];
    let mut gate = vec![0.0; rows * num_heads * head_dim];
    for row in 0..rows {
        let row_start = row * row_width;
        let query_row_start = row * num_heads * head_dim;
        let gate_row_start = row * num_heads * head_dim;
        for head in 0..num_heads {
            let src = row_start + head * head_dim * 2;
            let dst = query_row_start + head * head_dim;
            query[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
            gate[gate_row_start + head * head_dim..gate_row_start + (head + 1) * head_dim]
                .copy_from_slice(&input[src + head_dim..src + head_dim * 2]);
        }
    }
    (query, gate)
}

fn split_rows_3(
    input: &[f32],
    rows: usize,
    first_cols: usize,
    second_cols: usize,
    third_cols: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let row_width = first_cols + second_cols + third_cols;
    let mut first = vec![0.0; rows * first_cols];
    let mut second = vec![0.0; rows * second_cols];
    let mut third = vec![0.0; rows * third_cols];
    for row in 0..rows {
        let row_start = row * row_width;
        let first_start = row * first_cols;
        let second_start = row * second_cols;
        let third_start = row * third_cols;
        first[first_start..first_start + first_cols]
            .copy_from_slice(&input[row_start..row_start + first_cols]);
        second[second_start..second_start + second_cols]
            .copy_from_slice(&input[row_start + first_cols..row_start + first_cols + second_cols]);
        third[third_start..third_start + third_cols]
            .copy_from_slice(&input[row_start + first_cols + second_cols..row_start + row_width]);
    }
    (first, second, third)
}


fn apply_unary(input: &[f32], f: fn(f32) -> f32) -> Vec<f32> {
    input.iter().copied().map(f).collect()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn softmax_prefix(values: &[f32], len: usize) -> Vec<f32> {
    let max = values[..len]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0; values.len()];
    let mut sum = 0.0;
    for i in 0..len {
        out[i] = (values[i] - max).exp();
        sum += out[i];
    }
    if sum != 0.0 {
        for i in 0..len {
            out[i] /= sum;
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(idx, _)| idx)
}

fn format_generation_prompt(prompt: &str) -> String {
    if prompt.contains("<|im_start|>") {
        prompt.to_string()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt.trim()
        )
    }
}

fn strip_think_tags(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut remaining = text;
    loop {
        if let Some(start) = remaining.find("<think>") {
            cleaned.push_str(&remaining[..start]);
            let after_start = &remaining[start + "<think>".len()..];
            if let Some(end) = after_start.find("</think>") {
                remaining = &after_start[end + "</think>".len()..];
            } else {
                remaining = "";
            }
        } else {
            cleaned.push_str(remaining);
            break;
        }
    }
    cleaned
}

fn should_stop_early(prompt: &str, cleaned: &str) -> bool {
    let prompt = prompt.to_ascii_lowercase();
    let expects_number = prompt.contains("only answer with number")
        || prompt.contains("only answer with the number")
        || prompt.contains("answer with only the number");
    expects_number && is_numeric_answer(cleaned)
}

fn is_numeric_answer(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed
        .strip_suffix('.')
        .unwrap_or(trimmed)
        .parse::<f64>()
        .is_ok()
}
