use crate::{GgmlBackend, GgmlTensor};
use burn::tensor::backend::ExecutionError;
use burn::tensor::ops::BoolTensorOps;
use burn::tensor::Slice;
use burn::tensor::{Scalar, Shape, TensorData};
use std::future::Future;

impl BoolTensorOps<GgmlBackend> for GgmlBackend {
    fn bool_empty(shape: Shape, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn bool_from_data(data: TensorData, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn bool_into_data(
        tensor: GgmlTensor,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { todo!() }
    }

    fn bool_device(tensor: &GgmlTensor) -> crate::GgmlDevice {
        tensor.ctx.device.clone()
    }

    fn bool_to_device(tensor: GgmlTensor, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn bool_reshape(tensor: GgmlTensor, shape: Shape) -> GgmlTensor {
        todo!()
    }

    fn bool_slice(tensor: GgmlTensor, ranges: &[Slice]) -> GgmlTensor {
        todo!()
    }

    fn bool_slice_assign(tensor: GgmlTensor, ranges: &[Slice], value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_cat(tensors: Vec<GgmlTensor>, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn bool_equal(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_not(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_swap_dims(tensor: GgmlTensor, dim1: usize, dim2: usize) -> GgmlTensor {
        todo!()
    }

    fn bool_permute(tensor: GgmlTensor, dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn bool_flip(tensor: GgmlTensor, dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn bool_select(tensor: GgmlTensor, dim: usize, indices: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_and(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_or(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_zeros(shape: Shape, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn bool_ones(shape: Shape, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn bool_into_int(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_into_float(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_mask_where(condition: GgmlTensor, lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_mask_fill(tensor: GgmlTensor, mask: GgmlTensor, value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bool_gather(dim: usize, tensor: GgmlTensor, indices: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: GgmlTensor,
        indices: GgmlTensor,
        value: GgmlTensor,
    ) -> GgmlTensor {
        todo!()
    }

    fn bool_equal_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bool_expand(tensor: GgmlTensor, shape: Shape) -> GgmlTensor {
        todo!()
    }

    fn bool_unfold(tensor: GgmlTensor, dim: usize, size: usize, step: usize) -> GgmlTensor {
        todo!()
    }
}
