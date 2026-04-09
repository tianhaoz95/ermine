use burn::tensor::ops::IntTensorOps;
use burn::tensor::{Shape, TensorData, IntDType, Scalar, Distribution};
use burn::tensor::backend::ExecutionError;
use burn::tensor::Slice;
use crate::{GgmlBackend, GgmlTensor, GgmlContext};
use crate::device::GgmlDevice;
use core::future::Future;
use ggml_sys::*;
use std::ffi::c_void;

impl IntTensorOps<GgmlBackend> for GgmlBackend {
    fn int_empty(shape: Shape, device: &GgmlDevice, _dtype: IntDType) -> GgmlTensor {
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
            let executor = tensor.ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            
            let buft = ggml_backend_cpu_buffer_type();
            if ggml_get_no_alloc(tensor.ctx.ptr) {
                ggml_backend_alloc_ctx_tensors_from_buft(tensor.ctx.ptr, buft);
            }
            
            let tensor_bytes = ggml_nbytes(tensor.ptr);
            let data_bytes = bytes.len() * std::mem::size_of::<i32>();
            if data_bytes > tensor_bytes {
                panic!("int_from_data: data size {} > tensor size {}", data_bytes, tensor_bytes);
            }

            ggml_backend_tensor_set(
                tensor.ptr,
                bytes.as_ptr() as *const c_void,
                0,
                data_bytes,
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
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

    fn int_device(tensor: &GgmlTensor) -> GgmlDevice {
        tensor.ctx.device.clone()
    }

    fn int_to_device(tensor: GgmlTensor, device: &GgmlDevice) -> GgmlTensor {
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
        let ctx = tensor.ctx.clone();
        // Simplified slicing: only handle 1D slice for now
        assert_eq!(tensor.shape.len(), 1, "Only 1D slicing implemented for GGML backend");
        assert_eq!(ranges.len(), 1, "Expected 1 range for 1D tensor");
        
        let range = &ranges[0];
        let start = range.start as usize;
        let end = range.end.unwrap_or(tensor.shape[0] as isize) as usize;
        let size = end - start;
        
        unsafe {
            let offset = start * std::mem::size_of::<i32>();
            let t = ggml_view_1d(ctx.ptr, tensor.ptr, size as i64, offset);
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn int_slice_assign(
        _tensor: GgmlTensor,
        _ranges: &[Slice],
        _value: GgmlTensor,
    ) -> GgmlTensor {
        todo!()
    }

    fn int_cat(_tensors: Vec<GgmlTensor>, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_greater(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_greater_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_greater_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_greater_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_lower(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_lower_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_lower_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_lower_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_add(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_add_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_sub(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_sub_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_mul(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mul_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_div(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_div_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_zeros(_shape: Shape, _device: &GgmlDevice, _dtype: IntDType) -> GgmlTensor {
        todo!()
    }

    fn int_ones(_shape: Shape, _device: &GgmlDevice, _dtype: IntDType) -> GgmlTensor {
        todo!()
    }

    fn int_sum(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_sum_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_mean_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_argmax(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_argmin(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_max_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_max_dim_with_indices(_tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, GgmlTensor) {
        todo!()
    }

    fn int_min_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn int_min_dim_with_indices(_tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, GgmlTensor) {
        todo!()
    }

    fn int_abs(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_swap_dims(_tensor: GgmlTensor, _dim1: usize, _dim2: usize) -> GgmlTensor {
        todo!()
    }

    fn int_random(_shape: Shape, _distribution: Distribution, _device: &GgmlDevice) -> GgmlTensor {
        todo!()
    }

    fn int_gather(_dim: usize, _tensor: GgmlTensor, _indices: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_scatter_add(_dim: usize, _tensor: GgmlTensor, _indices: GgmlTensor, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_select(tensor: GgmlTensor, dim: usize, indices: GgmlTensor) -> GgmlTensor {
        assert_eq!(dim, 0, "GGML backend only supports select on dim 0 for now");
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_get_rows(ctx.ptr, tensor.ptr, indices.ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            let executor = ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            executor.compute_graph(gf).expect("Compute failed");
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn int_mask_where(_tensor: GgmlTensor, _mask: GgmlTensor, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn int_mask_fill(_tensor: GgmlTensor, _mask: GgmlTensor, _value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn int_remainder(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_remainder_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor { todo!() }
    fn int_clamp(_tensor: GgmlTensor, _min: Scalar, _max: Scalar) -> GgmlTensor { todo!() }
    fn int_powf(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_powf_scalar(_tensor: GgmlTensor, _value: Scalar) -> GgmlTensor { todo!() }
    fn int_sign(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_expand(_tensor: GgmlTensor, _shape: Shape) -> GgmlTensor { todo!() }
    
    fn bitwise_and(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_and_scalar(_tensor: GgmlTensor, _scalar: Scalar) -> GgmlTensor { todo!() }
    fn bitwise_or(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_or_scalar(_tensor: GgmlTensor, _scalar: Scalar) -> GgmlTensor { todo!() }
    fn bitwise_xor(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_xor_scalar(_tensor: GgmlTensor, _scalar: Scalar) -> GgmlTensor { todo!() }
    fn bitwise_not(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_left_shift(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_left_shift_scalar(_tensor: GgmlTensor, _scalar: Scalar) -> GgmlTensor { todo!() }
    fn bitwise_right_shift(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn bitwise_right_shift_scalar(_tensor: GgmlTensor, _scalar: Scalar) -> GgmlTensor { todo!() }
    fn int_cast(_tensor: GgmlTensor, _dtype: IntDType) -> GgmlTensor { todo!() }
    fn int_unfold(_tensor: GgmlTensor, _dim: usize, _size: usize, _step: usize) -> GgmlTensor { todo!() }
    fn int_into_float(_tensor: GgmlTensor) -> crate::GgmlTensor { todo!() }
    fn int_select_add(_tensor: GgmlTensor, _dim: usize, _indices: GgmlTensor, _value: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_matmul(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_prod(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn int_prod_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn int_cumsum(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn int_cumprod(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn int_cummin(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn int_cummax(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn int_permute(_tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor { todo!() }
    fn int_flip(_tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor { todo!() }
}
