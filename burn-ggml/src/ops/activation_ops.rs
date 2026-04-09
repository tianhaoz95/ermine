use crate::{GgmlBackend, GgmlTensor};
use burn::tensor::ops::ActivationOps;
use burn::tensor::Scalar;
use ggml_sys::*;

impl ActivationOps<GgmlBackend> for GgmlBackend {
    fn relu(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let out = ggml_relu(ctx.ptr, tensor.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn sigmoid(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let out = ggml_sigmoid(ctx.ptr, tensor.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, out);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(out, ctx.clone())
        }
    }

    fn gelu(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn leaky_relu(_tensor: GgmlTensor, _negative_slope: Scalar) -> GgmlTensor {
        todo!()
    }
}
