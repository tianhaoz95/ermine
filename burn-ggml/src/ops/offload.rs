use burn::tensor::backend::{Backend, ExecutionError};
use burn::tensor::ops::*;
use burn::tensor::{DType, Distribution, FloatDType, IntDType, Scalar, Shape, Slice, TensorData};
use core::future::Future;
use core::ops::Range;
use enumset::EnumSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static OFFLOAD_PREFETCH_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A backend wrapper that adds offloading/prefetching capabilities.
#[derive(Clone, Debug, Default)]
pub struct OffloadBackend<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> Backend for OffloadBackend<B> {
    type Device = B::Device;
    type FloatTensorPrimitive = B::FloatTensorPrimitive;
    type FloatElem = B::FloatElem;
    type IntTensorPrimitive = B::IntTensorPrimitive;
    type IntElem = B::IntElem;
    type BoolTensorPrimitive = B::BoolTensorPrimitive;
    type BoolElem = B::BoolElem;
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn name(device: &Self::Device) -> String {
        format!("offload({})", B::name(device))
    }

    fn seed(device: &Self::Device, seed: u64) {
        B::seed(device, seed);
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        B::sync(device)
    }

    fn dtype_usage(
        device: &Self::Device,
        dtype: DType,
    ) -> EnumSet<burn::tensor::backend::DTypeUsage> {
        B::dtype_usage(device, dtype)
    }
}

impl<B: Backend> FloatTensorOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn float_empty(shape: Shape, device: &B::Device, dtype: FloatDType) -> B::FloatTensorPrimitive {
        B::float_empty(shape, device, dtype)
    }

    fn float_from_data(data: TensorData, device: &B::Device) -> B::FloatTensorPrimitive {
        B::float_from_data(data, device)
    }

    fn float_into_data(
        tensor: B::FloatTensorPrimitive,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::float_into_data(tensor)
    }

    fn float_device(tensor: &B::FloatTensorPrimitive) -> B::Device {
        B::float_device(tensor)
    }

    fn float_to_device(
        tensor: B::FloatTensorPrimitive,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        B::float_to_device(tensor, device)
    }

    fn float_reshape(tensor: B::FloatTensorPrimitive, shape: Shape) -> B::FloatTensorPrimitive {
        B::float_reshape(tensor, shape)
    }

    fn float_slice(tensor: B::FloatTensorPrimitive, ranges: &[Slice]) -> B::FloatTensorPrimitive {
        B::float_slice(tensor, ranges)
    }

    fn float_slice_assign(
        tensor: B::FloatTensorPrimitive,
        ranges: &[Slice],
        value: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_slice_assign(tensor, ranges, value)
    }

    fn float_cat(tensors: Vec<B::FloatTensorPrimitive>, dim: usize) -> B::FloatTensorPrimitive {
        B::float_cat(tensors, dim)
    }

    fn float_equal(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::float_equal(lhs, rhs)
    }

    fn float_equal_elem(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_equal_elem(lhs, rhs)
    }

    fn float_greater(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::float_greater(lhs, rhs)
    }

    fn float_greater_elem(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_greater_elem(lhs, rhs)
    }

    fn float_greater_equal(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::float_greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(
        lhs: B::FloatTensorPrimitive,
        rhs: Scalar,
    ) -> B::BoolTensorPrimitive {
        B::float_greater_equal_elem(lhs, rhs)
    }

    fn float_lower(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::float_lower(lhs, rhs)
    }

    fn float_lower_elem(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_lower_elem(lhs, rhs)
    }

    fn float_lower_equal(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::float_lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::float_lower_equal_elem(lhs, rhs)
    }

    fn float_add(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_add(lhs, rhs)
    }

    fn float_add_scalar(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::FloatTensorPrimitive {
        B::float_add_scalar(lhs, rhs)
    }

    fn float_sub(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::FloatTensorPrimitive {
        B::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::FloatTensorPrimitive {
        B::float_mul_scalar(lhs, rhs)
    }

    fn float_div(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_div(lhs, rhs)
    }

    fn float_div_scalar(lhs: B::FloatTensorPrimitive, rhs: Scalar) -> B::FloatTensorPrimitive {
        B::float_div_scalar(lhs, rhs)
    }

    fn float_matmul(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_matmul(lhs, rhs)
    }

    fn float_swap_dims(
        tensor: B::FloatTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> B::FloatTensorPrimitive {
        B::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        B::float_random(shape, distribution, device)
    }

    fn float_into_int(tensor: B::FloatTensorPrimitive) -> B::IntTensorPrimitive {
        B::float_into_int(tensor)
    }

    fn float_exp(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_exp(tensor)
    }

    fn float_log(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_log(tensor)
    }

    fn float_log1p(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_log1p(tensor)
    }

    fn float_powf_scalar(
        tensor: B::FloatTensorPrimitive,
        value: Scalar,
    ) -> B::FloatTensorPrimitive {
        B::float_powf_scalar(tensor, value)
    }

    fn float_sqrt(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_sqrt(tensor)
    }

    fn float_abs(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_abs(tensor)
    }

    fn float_cos(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_cos(tensor)
    }

    fn float_sin(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_sin(tensor)
    }

    fn float_tanh(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_tanh(tensor)
    }

    fn float_erf(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_erf(tensor)
    }

    fn float_argmax(tensor: B::FloatTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::float_argmax(tensor, dim)
    }

    fn float_argmin(tensor: B::FloatTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::float_argmin(tensor, dim)
    }

    fn float_max_dim(tensor: B::FloatTensorPrimitive, dim: usize) -> B::FloatTensorPrimitive {
        B::float_max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: B::FloatTensorPrimitive,
        dim: usize,
    ) -> (B::FloatTensorPrimitive, B::IntTensorPrimitive) {
        B::float_max_dim_with_indices(tensor, dim)
    }

    fn float_min_dim(tensor: B::FloatTensorPrimitive, dim: usize) -> B::FloatTensorPrimitive {
        B::float_min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices(
        tensor: B::FloatTensorPrimitive,
        dim: usize,
    ) -> (B::FloatTensorPrimitive, B::IntTensorPrimitive) {
        B::float_min_dim_with_indices(tensor, dim)
    }

    fn float_recip(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_recip(tensor)
    }

    fn float_transpose(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_transpose(tensor)
    }

    fn float_permute(tensor: B::FloatTensorPrimitive, dims: &[usize]) -> B::FloatTensorPrimitive {
        B::float_permute(tensor, dims)
    }

    fn float_flip(tensor: B::FloatTensorPrimitive, dims: &[usize]) -> B::FloatTensorPrimitive {
        B::float_flip(tensor, dims)
    }

    fn float_mask_where(
        tensor: B::FloatTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: B::FloatTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> B::FloatTensorPrimitive {
        B::float_mask_fill(tensor, mask, value)
    }

    fn float_gather(
        dim: usize,
        tensor: B::FloatTensorPrimitive,
        indices: B::IntTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_gather(dim, tensor, indices)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: B::FloatTensorPrimitive,
        indices: B::IntTensorPrimitive,
        value: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_scatter_add(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: B::FloatTensorPrimitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_select(tensor, dim, indices)
    }

    fn float_clamp(
        tensor: B::FloatTensorPrimitive,
        min: Scalar,
        max: Scalar,
    ) -> B::FloatTensorPrimitive {
        B::float_clamp(tensor, min, max)
    }

    fn float_powf(
        lhs: B::FloatTensorPrimitive,
        rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        B::float_powf(lhs, rhs)
    }

    fn float_sign(tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        B::float_sign(tensor)
    }

    fn float_expand(tensor: B::FloatTensorPrimitive, shape: Shape) -> B::FloatTensorPrimitive {
        B::float_expand(tensor, shape)
    }

    fn float_unfold(
        tensor: B::FloatTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> B::FloatTensorPrimitive {
        B::float_unfold(tensor, dim, size, step)
    }

    fn float_remainder(
        _lhs: B::FloatTensorPrimitive,
        _rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_remainder_scalar(
        _lhs: B::FloatTensorPrimitive,
        _rhs: Scalar,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cross(
        _lhs: B::FloatTensorPrimitive,
        _rhs: B::FloatTensorPrimitive,
        _dim: usize,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_select_add(
        _tensor: B::FloatTensorPrimitive,
        _dim: usize,
        _indices: B::IntTensorPrimitive,
        _value: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_sum(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_sum_dim(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_mean_dim(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cumsum(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cumprod(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cummin(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cummax(_tensor: B::FloatTensorPrimitive, _dim: usize) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cast(_tensor: B::FloatTensorPrimitive, _dtype: FloatDType) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_powf_scalar_impl(
        _tensor: B::FloatTensorPrimitive,
        _value: Scalar,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_tan(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_cosh(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_sinh(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_acos(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_acosh(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_asin(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_asinh(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_atan(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_atanh(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_atan2(
        _lhs: B::FloatTensorPrimitive,
        _rhs: B::FloatTensorPrimitive,
    ) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_round(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_floor(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_ceil(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
    fn float_trunc(_tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        todo!()
    }
}

impl<B: Backend> IntTensorOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn int_empty(shape: Shape, device: &B::Device, dtype: IntDType) -> B::IntTensorPrimitive {
        B::int_empty(shape, device, dtype)
    }

    fn int_from_data(data: TensorData, device: &B::Device) -> B::IntTensorPrimitive {
        B::int_from_data(data, device)
    }

    fn int_into_data(
        tensor: B::IntTensorPrimitive,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::int_into_data(tensor)
    }

    fn int_device(tensor: &B::IntTensorPrimitive) -> B::Device {
        B::int_device(tensor)
    }

    fn int_to_device(tensor: B::IntTensorPrimitive, device: &B::Device) -> B::IntTensorPrimitive {
        B::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: B::IntTensorPrimitive, shape: Shape) -> B::IntTensorPrimitive {
        B::int_reshape(tensor, shape)
    }

    fn int_slice(tensor: B::IntTensorPrimitive, ranges: &[Slice]) -> B::IntTensorPrimitive {
        B::int_slice(tensor, ranges)
    }

    fn int_slice_assign(
        tensor: B::IntTensorPrimitive,
        ranges: &[Slice],
        value: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<B::IntTensorPrimitive>, dim: usize) -> B::IntTensorPrimitive {
        B::int_cat(tensors, dim)
    }

    fn int_equal(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::BoolTensorPrimitive {
        B::int_equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_equal_elem(lhs, rhs)
    }

    fn int_greater(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::int_greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::BoolTensorPrimitive {
        B::int_lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_add(lhs, rhs)
    }

    fn int_add_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::int_add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::int_mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_div(lhs, rhs)
    }

    fn int_div_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::int_div_scalar(lhs, rhs)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
    ) -> B::IntTensorPrimitive {
        B::int_random(shape, distribution, device)
    }

    fn int_into_float(tensor: B::IntTensorPrimitive) -> B::FloatTensorPrimitive {
        B::int_into_float(tensor)
    }

    fn int_swap_dims(
        tensor: B::IntTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> B::IntTensorPrimitive {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_argmax(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_argmax(tensor, dim)
    }

    fn int_argmin(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_argmin(tensor, dim)
    }

    fn int_max_dim(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_max_dim(tensor, dim)
    }

    fn int_max_dim_with_indices(
        tensor: B::IntTensorPrimitive,
        dim: usize,
    ) -> (B::IntTensorPrimitive, B::IntTensorPrimitive) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn int_min_dim(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_min_dim(tensor, dim)
    }

    fn int_min_dim_with_indices(
        tensor: B::IntTensorPrimitive,
        dim: usize,
    ) -> (B::IntTensorPrimitive, B::IntTensorPrimitive) {
        B::int_min_dim_with_indices(tensor, dim)
    }

    fn int_abs(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_abs(tensor)
    }

    fn int_transpose(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_transpose(tensor)
    }

    fn int_permute(tensor: B::IntTensorPrimitive, axes: &[usize]) -> B::IntTensorPrimitive {
        B::int_permute(tensor, axes)
    }

    fn int_flip(tensor: B::IntTensorPrimitive, axes: &[usize]) -> B::IntTensorPrimitive {
        B::int_flip(tensor, axes)
    }

    fn int_mask_where(
        tensor: B::IntTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: B::IntTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> B::IntTensorPrimitive {
        B::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: B::IntTensorPrimitive,
        indices: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_gather(dim, tensor, indices)
    }

    fn int_scatter_add(
        dim: usize,
        tensor: B::IntTensorPrimitive,
        indices: B::IntTensorPrimitive,
        value: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_scatter_add(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: B::IntTensorPrimitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_select(tensor, dim, indices)
    }

    fn int_clamp(tensor: B::IntTensorPrimitive, min: Scalar, max: Scalar) -> B::IntTensorPrimitive {
        B::int_clamp(tensor, min, max)
    }

    fn int_sign(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_sign(tensor)
    }

    fn int_expand(tensor: B::IntTensorPrimitive, shape: Shape) -> B::IntTensorPrimitive {
        B::int_expand(tensor, shape)
    }

    fn int_unfold(
        tensor: B::IntTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> B::IntTensorPrimitive {
        B::int_unfold(tensor, dim, size, step)
    }

    fn bitwise_and(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::bitwise_not(tensor)
    }

    fn bitwise_left_shift(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(
        lhs: B::IntTensorPrimitive,
        rhs: Scalar,
    ) -> B::IntTensorPrimitive {
        B::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_remainder(
        lhs: B::IntTensorPrimitive,
        rhs: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        B::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: B::IntTensorPrimitive, rhs: Scalar) -> B::IntTensorPrimitive {
        B::int_remainder_scalar(lhs, rhs)
    }

    fn int_matmul(lhs: B::IntTensorPrimitive, rhs: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_matmul(lhs, rhs)
    }

    fn int_select_add(
        _tensor: B::IntTensorPrimitive,
        _dim: usize,
        _indices: B::IntTensorPrimitive,
        _value: B::IntTensorPrimitive,
    ) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_sum(_tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_sum_dim(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_mean_dim(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_cumsum(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_cumprod(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_cummin(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_cummax(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_cast(_tensor: B::IntTensorPrimitive, _dtype: IntDType) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_arange(_range: Range<i64>, _device: &B::Device) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_arange_step(
        _range: Range<i64>,
        _step: usize,
        _device: &B::Device,
    ) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_prod(_tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        todo!()
    }
    fn int_prod_dim(_tensor: B::IntTensorPrimitive, _dim: usize) -> B::IntTensorPrimitive {
        todo!()
    }
}

impl<B: Backend> BoolTensorOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn bool_empty(shape: Shape, device: &B::Device) -> B::BoolTensorPrimitive {
        B::bool_empty(shape, device)
    }

    fn bool_from_data(data: TensorData, device: &B::Device) -> B::BoolTensorPrimitive {
        B::bool_from_data(data, device)
    }

    fn bool_into_data(
        tensor: B::BoolTensorPrimitive,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::bool_into_data(tensor)
    }

    fn bool_device(tensor: &B::BoolTensorPrimitive) -> B::Device {
        B::bool_device(tensor)
    }

    fn bool_to_device(
        tensor: B::BoolTensorPrimitive,
        device: &B::Device,
    ) -> B::BoolTensorPrimitive {
        B::bool_to_device(tensor, device)
    }

    fn bool_reshape(tensor: B::BoolTensorPrimitive, shape: Shape) -> B::BoolTensorPrimitive {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice(tensor: B::BoolTensorPrimitive, ranges: &[Slice]) -> B::BoolTensorPrimitive {
        B::bool_slice(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: B::BoolTensorPrimitive,
        ranges: &[Slice],
        value: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn bool_cat(tensors: Vec<B::BoolTensorPrimitive>, dim: usize) -> B::BoolTensorPrimitive {
        B::bool_cat(tensors, dim)
    }

    fn bool_equal(
        lhs: B::BoolTensorPrimitive,
        rhs: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_equal(lhs, rhs)
    }

    fn bool_not(tensor: B::BoolTensorPrimitive) -> B::BoolTensorPrimitive {
        B::bool_not(tensor)
    }

    fn bool_swap_dims(
        tensor: B::BoolTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> B::BoolTensorPrimitive {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: B::BoolTensorPrimitive, axes: &[usize]) -> B::BoolTensorPrimitive {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip(tensor: B::BoolTensorPrimitive, axes: &[usize]) -> B::BoolTensorPrimitive {
        B::bool_flip(tensor, axes)
    }

    fn bool_select(
        tensor: B::BoolTensorPrimitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_select(tensor, dim, indices)
    }

    fn bool_and(
        lhs: B::BoolTensorPrimitive,
        rhs: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: B::BoolTensorPrimitive, rhs: B::BoolTensorPrimitive) -> B::BoolTensorPrimitive {
        B::bool_or(lhs, rhs)
    }

    fn bool_xor(
        lhs: B::BoolTensorPrimitive,
        rhs: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_xor(lhs, rhs)
    }

    fn bool_into_float(tensor: B::BoolTensorPrimitive) -> B::FloatTensorPrimitive {
        B::bool_into_float(tensor)
    }

    fn bool_into_int(tensor: B::BoolTensorPrimitive) -> B::IntTensorPrimitive {
        B::bool_into_int(tensor)
    }

    fn bool_expand(tensor: B::BoolTensorPrimitive, shape: Shape) -> B::BoolTensorPrimitive {
        B::bool_expand(tensor, shape)
    }

    fn bool_zeros(shape: Shape, device: &B::Device) -> B::BoolTensorPrimitive {
        B::bool_zeros(shape, device)
    }
    fn bool_ones(shape: Shape, device: &B::Device) -> B::BoolTensorPrimitive {
        B::bool_ones(shape, device)
    }
    fn bool_mask_where(
        tensor: B::BoolTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_mask_where(tensor, mask, value)
    }
    fn bool_mask_fill(
        tensor: B::BoolTensorPrimitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> B::BoolTensorPrimitive {
        B::bool_mask_fill(tensor, mask, value)
    }
    fn bool_gather(
        dim: usize,
        tensor: B::BoolTensorPrimitive,
        indices: B::IntTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_gather(dim, tensor, indices)
    }
    fn bool_scatter_or(
        dim: usize,
        tensor: B::BoolTensorPrimitive,
        indices: B::IntTensorPrimitive,
        value: B::BoolTensorPrimitive,
    ) -> B::BoolTensorPrimitive {
        B::bool_scatter_or(dim, tensor, indices, value)
    }
    fn bool_equal_elem(lhs: B::BoolTensorPrimitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::bool_equal_elem(lhs, rhs)
    }
    fn bool_unfold(
        tensor: B::BoolTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> B::BoolTensorPrimitive {
        B::bool_unfold(tensor, dim, size, step)
    }
}

// Delegate ModuleOps, ActivationOps, QTensorOps as todo!() for now to ensure compilation
impl<B: Backend> ModuleOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn embedding(weights: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B> {
        B::embedding(weights, indices)
    }
    fn embedding_backward(
        weights: FloatTensor<B>,
        output_grad: FloatTensor<B>,
        indices: IntTensor<B>,
    ) -> FloatTensor<B> {
        B::embedding_backward(weights, output_grad, indices)
    }
    fn conv1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<1>,
    ) -> FloatTensor<B> {
        B::conv1d(x, weight, bias, options)
    }
    fn conv2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<B> {
        B::conv2d(x, weight, bias, options)
    }
    fn conv_transpose1d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<1>,
    ) -> FloatTensor<B> {
        B::conv_transpose1d(x, weight, bias, options)
    }
    fn conv_transpose2d(
        x: FloatTensor<B>,
        weight: FloatTensor<B>,
        bias: Option<FloatTensor<B>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<B> {
        B::conv_transpose2d(x, weight, bias, options)
    }
    fn avg_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        B::avg_pool1d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    fn avg_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        B::avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }
    fn max_pool1d(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        B::max_pool1d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }
    fn max_pool1d_with_indices(
        x: FloatTensor<B>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> MaxPool1dWithIndices<OffloadBackend<B>> {
        let b_res =
            B::max_pool1d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool1dWithIndices {
            output: b_res.output,
            indices: b_res.indices,
        }
    }
    fn max_pool2d(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<B> {
        B::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    }
    fn max_pool2d_with_indices(
        x: FloatTensor<B>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<OffloadBackend<B>> {
        let b_res =
            B::max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode);
        MaxPool2dWithIndices {
            output: b_res.output,
            indices: b_res.indices,
        }
    }
    fn adaptive_avg_pool1d(x: FloatTensor<B>, output_size: usize) -> FloatTensor<B> {
        B::adaptive_avg_pool1d(x, output_size)
    }
    fn adaptive_avg_pool2d(x: FloatTensor<B>, output_size: [usize; 2]) -> FloatTensor<B> {
        B::adaptive_avg_pool2d(x, output_size)
    }
    fn interpolate(
        x: FloatTensor<B>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<B> {
        B::interpolate(x, output_size, options)
    }
    fn attention(
        query: FloatTensor<B>,
        key: FloatTensor<B>,
        value: FloatTensor<B>,
        mask: Option<BoolTensor<B>>,
        attn_bias: Option<FloatTensor<B>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<B> {
        B::attention(query, key, value, mask, attn_bias, options)
    }

    fn deform_conv2d(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: Option<FloatTensor<B>>,
        _: Option<FloatTensor<B>>,
        _: DeformConvOptions<2>,
    ) -> FloatTensor<B> {
        todo!()
    }
    fn deform_conv2d_backward(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: Option<FloatTensor<B>>,
        _: Option<FloatTensor<B>>,
        _: FloatTensor<B>,
        _: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<OffloadBackend<B>> {
        todo!()
    }
    fn conv3d(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: Option<FloatTensor<B>>,
        _: ConvOptions<3>,
    ) -> FloatTensor<B> {
        todo!()
    }
    fn conv_transpose3d(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: Option<FloatTensor<B>>,
        _: ConvTransposeOptions<3>,
    ) -> FloatTensor<B> {
        todo!()
    }
    fn avg_pool2d_backward(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: [usize; 2],
        _: [usize; 2],
        _: [usize; 2],
        _: bool,
        _: bool,
    ) -> FloatTensor<B> {
        todo!()
    }
    fn adaptive_avg_pool2d_backward(_: FloatTensor<B>, _: FloatTensor<B>) -> FloatTensor<B> {
        todo!()
    }
    fn max_pool2d_with_indices_backward(
        _: FloatTensor<B>,
        _: [usize; 2],
        _: [usize; 2],
        _: [usize; 2],
        _: [usize; 2],
        _: bool,
        _: FloatTensor<B>,
        _: IntTensor<B>,
    ) -> MaxPool2dBackward<OffloadBackend<B>> {
        todo!()
    }
    fn interpolate_backward(
        _: FloatTensor<B>,
        _: FloatTensor<B>,
        _: [usize; 2],
        _: InterpolateOptions,
    ) -> FloatTensor<B> {
        todo!()
    }
}

impl<B: Backend> ActivationOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn relu(tensor: FloatTensor<B>) -> FloatTensor<B> {
        B::relu(tensor)
    }
    fn gelu(tensor: FloatTensor<B>) -> FloatTensor<B> {
        B::gelu(tensor)
    }
    fn sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B> {
        B::sigmoid(tensor)
    }
    fn log_sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B> {
        B::log_sigmoid(tensor)
    }
    fn leaky_relu(tensor: FloatTensor<B>, negative_slope: Scalar) -> FloatTensor<B> {
        B::leaky_relu(tensor, negative_slope)
    }
    fn prelu(tensor: FloatTensor<B>, alpha: FloatTensor<B>) -> FloatTensor<B> {
        B::prelu(tensor, alpha)
    }
    fn hard_sigmoid(tensor: FloatTensor<B>, alpha: Scalar, beta: Scalar) -> FloatTensor<B> {
        B::hard_sigmoid(tensor, alpha, beta)
    }
}

impl<B: Backend> QTensorOps<OffloadBackend<B>> for OffloadBackend<B> {
    fn q_from_data(data: TensorData, device: &B::Device) -> B::QuantizedTensorPrimitive {
        B::q_from_data(data, device)
    }
    fn q_into_data(
        tensor: B::QuantizedTensorPrimitive,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::q_into_data(tensor)
    }
    fn quantize(
        tensor: B::FloatTensorPrimitive,
        scheme: &burn::tensor::quantization::QuantScheme,
        parameters: burn::tensor::quantization::QuantizationParametersPrimitive<OffloadBackend<B>>,
    ) -> B::QuantizedTensorPrimitive {
        let b_params = burn::tensor::quantization::QuantizationParametersPrimitive {
            scales: parameters.scales,
        };
        B::quantize(tensor, scheme, b_params)
    }
    fn dequantize(tensor: B::QuantizedTensorPrimitive) -> B::FloatTensorPrimitive {
        B::dequantize(tensor)
    }
    fn q_device(tensor: &B::QuantizedTensorPrimitive) -> B::Device {
        B::q_device(tensor)
    }
    fn q_to_device(
        tensor: B::QuantizedTensorPrimitive,
        device: &B::Device,
    ) -> B::QuantizedTensorPrimitive {
        B::q_to_device(tensor, device)
    }
    fn q_reshape(tensor: B::QuantizedTensorPrimitive, shape: Shape) -> B::QuantizedTensorPrimitive {
        B::q_reshape(tensor, shape)
    }
    fn q_expand(tensor: B::QuantizedTensorPrimitive, shape: Shape) -> B::QuantizedTensorPrimitive {
        B::q_expand(tensor, shape)
    }
    fn q_swap_dims(
        tensor: B::QuantizedTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> B::QuantizedTensorPrimitive {
        B::q_swap_dims(tensor, dim1, dim2)
    }
    fn q_permute(
        tensor: B::QuantizedTensorPrimitive,
        dims: &[usize],
    ) -> B::QuantizedTensorPrimitive {
        B::q_permute(tensor, dims)
    }
    fn q_flip(tensor: B::QuantizedTensorPrimitive, dims: &[usize]) -> B::QuantizedTensorPrimitive {
        B::q_flip(tensor, dims)
    }
    fn q_select(
        tensor: B::QuantizedTensorPrimitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> B::QuantizedTensorPrimitive {
        B::q_select(tensor, dim, indices)
    }
    fn q_slice(
        tensor: B::QuantizedTensorPrimitive,
        ranges: &[Slice],
    ) -> B::QuantizedTensorPrimitive {
        B::q_slice(tensor, ranges)
    }
}

impl<B: Backend> TransactionOps<OffloadBackend<B>> for OffloadBackend<B> {}

// Now for our specialized logic

/// Specialized hook implementation for the OffloadBackend.
pub(crate) fn offload_prefetch_hook<B: Backend>(
    primitive: crate::ops::PrefetchTensorPrimitive<B>,
    device: &B::Device,
) {
    if B::name(device).starts_with("offload(") {
        OFFLOAD_PREFETCH_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}
