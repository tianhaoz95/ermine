use crate::{GgmlBackend, GgmlQuantizedTensor, GgmlTensor};
use burn::tensor::backend::ExecutionError;
use burn::tensor::ops::QTensorOps;
use burn::tensor::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn::tensor::Slice;
use burn::tensor::{Shape, TensorData};
use std::future::Future;

impl QTensorOps<GgmlBackend> for GgmlBackend {
    fn q_from_data(data: TensorData, device: &crate::GgmlDevice) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_into_data(
        tensor: GgmlQuantizedTensor,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { todo!() }
    }

    fn quantize(
        tensor: GgmlTensor,
        scheme: &QuantScheme,
        parameters: QuantizationParametersPrimitive<GgmlBackend>,
    ) -> GgmlQuantizedTensor {
        todo!()
    }

    fn dequantize(tensor: GgmlQuantizedTensor) -> GgmlTensor {
        todo!()
    }

    fn q_device(tensor: &GgmlQuantizedTensor) -> crate::GgmlDevice {
        todo!()
    }

    fn q_to_device(tensor: GgmlQuantizedTensor, device: &crate::GgmlDevice) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_reshape(tensor: GgmlQuantizedTensor, shape: Shape) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_expand(tensor: GgmlQuantizedTensor, shape: Shape) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_swap_dims(tensor: GgmlQuantizedTensor, dim1: usize, dim2: usize) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_permute(tensor: GgmlQuantizedTensor, dims: &[usize]) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_flip(tensor: GgmlQuantizedTensor, dims: &[usize]) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_select(
        tensor: GgmlQuantizedTensor,
        dim: usize,
        indices: GgmlTensor,
    ) -> GgmlQuantizedTensor {
        todo!()
    }

    fn q_slice(tensor: GgmlQuantizedTensor, ranges: &[Slice]) -> GgmlQuantizedTensor {
        todo!()
    }
}
