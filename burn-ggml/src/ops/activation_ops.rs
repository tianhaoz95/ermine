use burn::tensor::ops::ActivationOps;
use burn::tensor::Scalar;
use crate::{GgmlBackend, GgmlTensor};
use ggml_sys::*;

impl ActivationOps<GgmlBackend> for GgmlBackend {
    fn relu(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_relu(ctx.ptr, tensor.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            let executor = ctx.executor.clone(); let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn gelu(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn sigmoid(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_sigmoid(ctx.ptr, tensor.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            let executor = ctx.executor.clone(); let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn log_sigmoid(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn leaky_relu(_tensor: GgmlTensor, _negative_slope: Scalar) -> GgmlTensor {
        todo!()
    }

    fn prelu(_tensor: GgmlTensor, _alpha: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn hard_sigmoid(_tensor: GgmlTensor, _alpha: Scalar, _beta: Scalar) -> GgmlTensor {
        todo!()
    }

    fn relu_backward(_output: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn gelu_backward(_x: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn sigmoid_backward(_output: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn log_sigmoid_backward(_x: GgmlTensor, _grad: GgmlTensor) -> GgmlTensor {
        todo!()
    }
}
