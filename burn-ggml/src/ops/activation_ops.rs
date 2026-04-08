use burn::tensor::ops::ActivationOps;
use burn::tensor::Scalar;
use crate::{GgmlBackend, GgmlTensor};

impl ActivationOps<GgmlBackend> for GgmlBackend {
    fn relu(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn gelu(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn sigmoid(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn log_sigmoid(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn leaky_relu(tensor: GgmlTensor, _negative_slope: Scalar) -> GgmlTensor {
        todo!()
    }

    fn prelu(tensor: GgmlTensor, _alpha: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn hard_sigmoid(tensor: GgmlTensor, _alpha: Scalar, _beta: Scalar) -> GgmlTensor {
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
