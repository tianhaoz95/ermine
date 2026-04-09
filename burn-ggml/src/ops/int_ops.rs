use burn::tensor::ops::IntTensorOps;
use burn::tensor::{Shape, TensorData, IntDType, Scalar, Distribution};
use burn::tensor::backend::ExecutionError;
use burn::tensor::Slice;
use crate::{GgmlBackend, GgmlTensor, GgmlContext};
use std::future::Future;
use ggml_sys::*;
use std::ffi::c_void;

impl IntTensorOps<GgmlBackend> for GgmlBackend {
    fn int_empty(shape: Shape, device: &crate::GgmlDevice, _dtype: IntDType) -> GgmlTensor {
        let ctx = GgmlContext::get(device);
        let mut dims = [1i64; 4];
        let shape_dims = match shape.num_dims() {
            1 => shape.dims::<1>().to_vec(),
            2 => shape.dims::<2>().to_vec(),
            3 => shape.dims::<3>().to_vec(),
            4 => shape.dims::<4>().to_vec(),
            _ => panic!("Unsupported dimensions: {}", shape.num_dims()),
        };
        for (i, &d) in shape_dims.iter().rev().enumerate() {
            dims[i] = d as i64;
        }
        
        unsafe {
            let t = ggml_new_tensor(ctx.ptr, ggml_type_GGML_TYPE_I32, shape.num_dims() as i32, dims.as_ptr());
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn int_from_data(data: TensorData, device: &crate::GgmlDevice) -> GgmlTensor {
        let shape = data.shape.clone();
        let tensor = Self::int_empty(shape.into(), device, IntDType::I32);
        unsafe {
            let bytes = data.as_slice::<i32>().unwrap();
            let _guard = tensor.ctx.executor.lock.lock().unwrap();
            ggml_backend_alloc_ctx_tensors(tensor.ctx.ptr, tensor.ctx.backend);
            ggml_backend_tensor_set(
                tensor.ptr,
                bytes.as_ptr() as *const c_void,
                0,
                bytes.len() * std::mem::size_of::<i32>(),
            );
        }
        tensor
    }

    fn int_into_data(tensor: GgmlTensor) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let ptr = tensor.ptr as usize;
        let shape = tensor.shape.clone();
        let ctx = tensor.ctx.clone();
        async move {
            let n = shape.iter().product::<usize>();
            let mut data = vec![0i32; n];
            unsafe {
                let ptr = ptr as *mut ggml_tensor;
                let _guard = ctx.executor.lock.lock().unwrap();
                ggml_backend_tensor_get(
                    ptr,
                    data.as_mut_ptr() as *mut c_void,
                    0,
                    data.len() * std::mem::size_of::<i32>(),
                );
            }
            Ok(TensorData::new(data, shape))
        }
    }

    fn int_device(tensor: &GgmlTensor) -> crate::GgmlDevice {
        tensor.ctx.device.clone()
    }

    fn int_to_device(tensor: GgmlTensor, device: &crate::GgmlDevice) -> GgmlTensor {
        if &tensor.ctx.device == device {
            return tensor;
        }
        
        let data = Self::int_into_data(tensor);
        let data = futures::executor::block_on(data).unwrap();
        Self::int_from_data(data, device)
    }

    fn int_reshape(tensor: GgmlTensor, shape: Shape) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        let mut dims = [1i64; 4];
        let shape_dims = match shape.num_dims() {
            1 => shape.dims::<1>().to_vec(),
            2 => shape.dims::<2>().to_vec(),
            3 => shape.dims::<3>().to_vec(),
            4 => shape.dims::<4>().to_vec(),
            _ => panic!("Unsupported dimensions: {}", shape.num_dims()),
        };
        for (i, &d) in shape_dims.iter().rev().enumerate() {
            dims[i] = d as i64;
        }

        unsafe {
            let t = ggml_reshape_4d(ctx.ptr, tensor.ptr, dims[0], dims[1], dims[2], dims[3]);
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn int_slice(tensor: GgmlTensor, ranges: &[Slice]) -> GgmlTensor {
        todo!()
    }

    fn int_slice_assign(tensor: GgmlTensor, ranges: &[Slice], value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_cat(tensors: Vec<GgmlTensor>, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_add(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_zeros(shape: Shape, device: &crate::GgmlDevice, dtype: IntDType) -> GgmlTensor {
        todo!()
    }

    fn int_ones(shape: Shape, device: &crate::GgmlDevice, dtype: IntDType) -> GgmlTensor {
        todo!()
    }

    fn int_random(shape: Shape, distribution: Distribution, device: &crate::GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn int_permute(tensor: GgmlTensor, dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn int_flip(tensor: GgmlTensor, dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn int_select(tensor: GgmlTensor, dim: usize, indices: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mask_where(tensor: GgmlTensor, mask: GgmlTensor, value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mask_fill(tensor: GgmlTensor, mask: GgmlTensor, value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_gather(dim: usize, tensor: GgmlTensor, indices: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_equal(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_equal_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_greater(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_greater_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_greater_equal(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_greater_equal_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_lower(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_lower_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_lower_equal(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_lower_equal_elem(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_sub(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_div(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_abs(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_clamp(tensor: GgmlTensor, min: Scalar, max: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_argmax(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_argmin(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_max(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_max_dim(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_max_dim_with_indices(tensor: GgmlTensor, dim: usize) -> (GgmlTensor, GgmlTensor) {
        todo!()
    }

    fn int_min(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_min_dim(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_min_dim_with_indices(tensor: GgmlTensor, dim: usize) -> (GgmlTensor, GgmlTensor) {
        todo!()
    }

    fn int_sum(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_sum_dim(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_mean(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mean_dim(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_into_float(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_scatter_add(dim: usize, tensor: GgmlTensor, indices: GgmlTensor, value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_select_add(tensor: GgmlTensor, dim: usize, indices: GgmlTensor, value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_add_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_sub_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_mul_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_div_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_remainder(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_remainder_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_matmul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_prod(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_prod_dim(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_cumsum(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_cumprod(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_cummin(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_cummax(tensor: GgmlTensor, dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_swap_dims(tensor: GgmlTensor, dim1: usize, dim2: usize) -> GgmlTensor {
        todo!()
    }

    fn int_expand(tensor: GgmlTensor, shape: Shape) -> GgmlTensor {
        todo!()
    }

    fn bitwise_and(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_and_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bitwise_or(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_or_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bitwise_xor(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_xor_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bitwise_not(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_left_shift(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_left_shift_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn bitwise_right_shift(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn bitwise_right_shift_scalar(tensor: GgmlTensor, scalar: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_cast(tensor: GgmlTensor, dtype: IntDType) -> GgmlTensor {
        todo!()
    }

    fn int_unfold(tensor: GgmlTensor, dim: usize, size: usize, step: usize) -> GgmlTensor {
        todo!()
    }
}
