use burn::tensor::ops::ModuleOps;
use crate::{GgmlBackend, GgmlTensor};
use ggml_sys::*;
use burn::tensor::ops::*;

impl ModuleOps<GgmlBackend> for GgmlBackend {
    fn embedding(weight: GgmlTensor, indices: GgmlTensor) -> GgmlTensor {
        let ctx = indices.ctx.clone();
        println!("DEBUG: embedding start");
        unsafe {
            // ggml_get_rows: rows of weight indexed by indices
            let out = ggml_get_rows(ctx.ptr, weight.ptr, indices.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn embedding_backward(_weight: GgmlTensor, _output: GgmlTensor, _indices: GgmlTensor) -> GgmlTensor { todo!() }
    fn conv1d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<1>) -> GgmlTensor { todo!() }
    fn conv2d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<2>) -> GgmlTensor { todo!() }
    fn deform_conv2d(_x: GgmlTensor, _offset: GgmlTensor, _weight: GgmlTensor, _mask: Option<GgmlTensor>, _bias: Option<GgmlTensor>, _options: DeformConvOptions<2>) -> GgmlTensor { todo!() }
    fn deform_conv2d_backward(_x: GgmlTensor, _offset: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _mask: Option<GgmlTensor>, _out_grad: GgmlTensor, _options: DeformConvOptions<2>) -> DeformConv2dBackward<GgmlBackend> { todo!() }
    fn conv3d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvOptions<3>) -> GgmlTensor { todo!() }
    fn conv_transpose2d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvTransposeOptions<2>) -> GgmlTensor { todo!() }
    fn conv_transpose3d(_x: GgmlTensor, _weight: GgmlTensor, _bias: Option<GgmlTensor>, _options: ConvTransposeOptions<3>) -> GgmlTensor { todo!() }
    fn avg_pool2d(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _count_include_pad: bool, _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn avg_pool2d_backward(_x: GgmlTensor, _grad: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _count_include_pad: bool, _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn adaptive_avg_pool2d(_x: GgmlTensor, _output_size: [usize; 2]) -> GgmlTensor { todo!() }
    fn adaptive_avg_pool2d_backward(_x: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor { todo!() }
    fn max_pool2d(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool) -> GgmlTensor { todo!() }
    fn max_pool2d_with_indices(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool) -> MaxPool2dWithIndices<GgmlBackend> { todo!() }
    fn max_pool2d_with_indices_backward(_x: GgmlTensor, _kernel_size: [usize; 2], _stride: [usize; 2], _padding: [usize; 2], _dilation: [usize; 2], _ceil_mode: bool, _output_grad: GgmlTensor, _indices: GgmlTensor) -> MaxPool2dBackward<GgmlBackend> { todo!() }
    fn interpolate(_x: GgmlTensor, _output_size: [usize; 2], _options: InterpolateOptions) -> GgmlTensor { todo!() }
    fn interpolate_backward(_x: GgmlTensor, _grad: GgmlTensor, _output_size: [usize; 2], _options: InterpolateOptions) -> GgmlTensor { todo!() }
    fn attention(_query: GgmlTensor, _key: GgmlTensor, _value: GgmlTensor, _mask: Option<BoolTensor<GgmlBackend>>, _attn_bias: Option<GgmlTensor>, _options: AttentionModuleOptions) -> GgmlTensor { todo!() }
}

/// GGML-specific operations not covered by standard Burn traits.
pub trait GgmlOps {
    fn rms_norm(x: GgmlTensor, weight: GgmlTensor, eps: f32) -> GgmlTensor;
    fn silu(x: GgmlTensor) -> GgmlTensor;
    fn softmax(x: GgmlTensor) -> GgmlTensor;
    fn rope(x: GgmlTensor, positions: GgmlTensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32) -> GgmlTensor;
    fn flash_attn(q: GgmlTensor, k: GgmlTensor, v: GgmlTensor, mask: Option<GgmlTensor>, scale: f32) -> GgmlTensor;
}

impl GgmlOps for GgmlBackend {
    fn rms_norm(x: GgmlTensor, weight: GgmlTensor, eps: f32) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let normed = ggml_rms_norm(ctx.ptr, x.ptr, eps);
            let out = ggml_mul(ctx.ptr, normed, weight.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn silu(x: GgmlTensor) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let out = ggml_silu(ctx.ptr, x.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn softmax(x: GgmlTensor) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let out = ggml_soft_max(ctx.ptr, x.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn rope(x: GgmlTensor, positions: GgmlTensor, n_dims: i32, mode: i32, n_ctx_orig: i32, freq_base: f32, freq_scale: f32) -> GgmlTensor {
        let ctx = x.ctx.clone();
        unsafe {
            let out = ggml_rope_ext(
                ctx.ptr, x.ptr, positions.ptr, std::ptr::null_mut(),
                n_dims, mode, n_ctx_orig, freq_base, freq_scale,
                0.0, 1.0, 0.0, 0.0
            );
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn flash_attn(q: GgmlTensor, k: GgmlTensor, v: GgmlTensor, mask: Option<GgmlTensor>, scale: f32) -> GgmlTensor {
        let ctx = q.ctx.clone();
        unsafe {
            let mask_ptr = mask.map(|m| m.ptr).unwrap_or(std::ptr::null_mut());
            let out = ggml_flash_attn_ext(ctx.ptr, q.ptr, k.ptr, v.ptr, mask_ptr, scale, 0.0, 0.0);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }
}
