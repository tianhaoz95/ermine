use burn::tensor::ops::{FloatTensorOps, IntTensor};
use crate::{GgmlBackend, GgmlTensor, GgmlContext};
use burn::tensor::{Shape, TensorData, Distribution, Scalar, FloatDType, Slice};
use burn::tensor::backend::ExecutionError;
use crate::device::GgmlDevice;
use core::future::Future;
use ggml_sys::*;
use std::sync::Arc;
use std::ffi::c_void;
use num_traits::ToPrimitive;

impl FloatTensorOps<GgmlBackend> for GgmlBackend {
    fn float_empty(shape: Shape, device: &GgmlDevice, _dtype: FloatDType) -> GgmlTensor {
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
            let t = ggml_new_tensor(ctx.ptr, ggml_type_GGML_TYPE_F32, shape.num_dims() as i32, dims.as_ptr());
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_from_data(data: TensorData, device: &GgmlDevice) -> GgmlTensor {
        let shape = data.shape.clone();
        let tensor = Self::float_empty(shape.into(), device, FloatDType::F32);
        unsafe {
            let bytes = data.as_slice::<f32>().unwrap();
            let _guard = tensor.ctx.executor.lock.lock().unwrap();
            ggml_backend_alloc_ctx_tensors(tensor.ctx.ptr, tensor.ctx.backend);
            ggml_backend_tensor_set(
                tensor.ptr,
                bytes.as_ptr() as *const c_void,
                0,
                bytes.len() * std::mem::size_of::<f32>(),
            );
        }
        tensor
    }

    fn float_into_data(tensor: GgmlTensor) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        let ptr = tensor.ptr as usize;
        let shape = tensor.shape.clone();
        let ctx = tensor.ctx.clone();
        async move {
            let n = shape.iter().product::<usize>();
            let mut data = vec![0.0f32; n];
            unsafe {
                let ptr = ptr as *mut ggml_tensor;
                let _guard = ctx.executor.lock.lock().unwrap();
                ggml_backend_tensor_get(
                    ptr,
                    data.as_mut_ptr() as *mut c_void,
                    0,
                    data.len() * std::mem::size_of::<f32>(),
                );
            }
            Ok(TensorData::new(data, shape))
        }
    }

    fn float_device(tensor: &GgmlTensor) -> GgmlDevice {
        tensor.ctx.device.clone()
    }

    fn float_to_device(tensor: GgmlTensor, device: &GgmlDevice) -> GgmlTensor {
        if &tensor.ctx.device == device {
            return tensor;
        }
        
        let data = Self::float_into_data(tensor);
        // This is a bit inefficient but safe for a backend bridge
        let data = futures::executor::block_on(data).unwrap();
        Self::float_from_data(data, device)
    }

    fn float_reshape(tensor: GgmlTensor, shape: Shape) -> GgmlTensor {
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

    fn float_slice(tensor: GgmlTensor, _ranges: &[Slice]) -> GgmlTensor {
        todo!()
    }

    fn float_slice_assign(
        _tensor: GgmlTensor,
        _ranges: &[Slice],
        _value: GgmlTensor,
    ) -> GgmlTensor {
        todo!()
    }

    fn float_cat(_tensors: Vec<GgmlTensor>, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn float_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_greater(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_greater_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_greater_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_greater_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_lower(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_lower_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_lower_equal(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_lower_equal_elem(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_add(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if Arc::ptr_eq(&lhs.ctx, &rhs.ctx) {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let _guard_rhs = rhs.ctx.executor.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let _guard_lhs = ctx.executor.lock.lock().unwrap();
                    ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_add(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_add_scalar(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_val = rhs.to_f32().unwrap();
            let rhs_tensor = ggml_new_f32(ctx.ptr, rhs_val);
            let t = ggml_add(ctx.ptr, lhs.ptr, rhs_tensor);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_sub(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if Arc::ptr_eq(&lhs.ctx, &rhs.ctx) {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let _guard_rhs = rhs.ctx.executor.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let _guard_lhs = ctx.executor.lock.lock().unwrap();
                    ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_sub(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_sub_scalar(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_val = rhs.to_f32().unwrap();
            let rhs_tensor = ggml_new_f32(ctx.ptr, rhs_val);
            let t = ggml_sub(ctx.ptr, lhs.ptr, rhs_tensor);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_mul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if Arc::ptr_eq(&lhs.ctx, &rhs.ctx) {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let _guard_rhs = rhs.ctx.executor.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let _guard_lhs = ctx.executor.lock.lock().unwrap();
                    ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_mul(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_mul_scalar(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_val = rhs.to_f32().unwrap();
            let rhs_tensor = ggml_new_f32(ctx.ptr, rhs_val);
            let t = ggml_mul(ctx.ptr, lhs.ptr, rhs_tensor);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_div(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if Arc::ptr_eq(&lhs.ctx, &rhs.ctx) {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let _guard_rhs = rhs.ctx.executor.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let _guard_lhs = ctx.executor.lock.lock().unwrap();
                    ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_div(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_div_scalar(lhs: GgmlTensor, rhs: Scalar) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_val = rhs.to_f32().unwrap();
            let rhs_tensor = ggml_new_f32(ctx.ptr, rhs_val);
            let t = ggml_div(ctx.ptr, lhs.ptr, rhs_tensor);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_matmul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if Arc::ptr_eq(&lhs.ctx, &rhs.ctx) {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let _guard_rhs = rhs.ctx.executor.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let _guard_lhs = ctx.executor.lock.lock().unwrap();
                    ggml_backend_alloc_ctx_tensors(ctx.ptr, ctx.backend);
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            // Fix orientation: ggml_mul_mat(a, b) -> a @ b^T
            let rhs_t = ggml_transpose(ctx.ptr, rhs_ptr);
            let rhs_cont = ggml_cont(ctx.ptr, rhs_t);
            let t_view = ggml_mul_mat(ctx.ptr, lhs.ptr, rhs_cont);
            let t_trans = ggml_transpose(ctx.ptr, t_view);
            let t = ggml_cont(ctx.ptr, t_trans);
            
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let _guard = ctx.executor.lock.lock().unwrap();
                ctx.executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_swap_dims(tensor: GgmlTensor, _dim1: usize, _dim2: usize) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_transpose(ctx.ptr, tensor.ptr);
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_random(shape: Shape, _distribution: Distribution, device: &GgmlDevice) -> GgmlTensor {
        let mut data = vec![0.0f32; shape.iter().product::<usize>()];
        for x in data.iter_mut() {
            *x = rand::random::<f32>();
        }
        Self::float_from_data(TensorData::new(data, shape), device)
    }

    fn float_zeros(shape: Shape, device: &GgmlDevice, dtype: FloatDType) -> GgmlTensor { 
        let t = Self::float_empty(shape, device, dtype);
        unsafe {
            let _guard = t.ctx.executor.lock.lock().unwrap();
            ggml_backend_alloc_ctx_tensors(t.ctx.ptr, t.ctx.backend);
            std::ptr::write_bytes(ggml_get_data(t.ptr), 0, ggml_nbytes(t.ptr));
        }
        t
    }

    fn float_ones(shape: Shape, device: &GgmlDevice, dtype: FloatDType) -> GgmlTensor {
        let n = shape.iter().product::<usize>();
        let data = vec![1.0f32; n];
        Self::float_from_data(TensorData::new(data, shape), device)
    }

    fn float_into_int(tensor: GgmlTensor) -> IntTensor<GgmlBackend> {
        todo!()
    }

    fn float_powf_scalar(tensor: GgmlTensor, _value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_sqrt(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_abs(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_cos(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_sin(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_tanh(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_erf(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_argmax(tensor: GgmlTensor, _dim: usize) -> IntTensor<GgmlBackend> {
        todo!()
    }

    fn float_argmin(tensor: GgmlTensor, _dim: usize) -> IntTensor<GgmlBackend> {
        todo!()
    }

    fn float_max_dim(tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn float_max_dim_with_indices(tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, IntTensor<GgmlBackend>) {
        todo!()
    }

    fn float_min_dim(tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn float_min_dim_with_indices(tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, IntTensor<GgmlBackend>) {
        todo!()
    }

    fn float_recip(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_transpose(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_transpose(ctx.ptr, tensor.ptr);
            GgmlTensor::from_raw(t, ctx)
        }
    }

    fn float_permute(tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn float_flip(tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn float_mask_where(tensor: GgmlTensor, _mask: GgmlTensor, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_mask_fill(tensor: GgmlTensor, _mask: GgmlTensor, _value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_gather(_dim: usize, _tensor: GgmlTensor, _indices: IntTensor<GgmlBackend>) -> GgmlTensor {
        todo!()
    }

    fn float_scatter_add(_dim: usize, _tensor: GgmlTensor, _indices: IntTensor<GgmlBackend>, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_select(tensor: GgmlTensor, _dim: usize, _indices: IntTensor<GgmlBackend>) -> GgmlTensor {
        todo!()
    }

    fn float_clamp(tensor: GgmlTensor, _min: Scalar, _max: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_powf(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_sign(tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_expand(tensor: GgmlTensor, _shape: Shape) -> GgmlTensor {
        todo!()
    }

    fn float_unfold(tensor: GgmlTensor, _dim: usize, _size: usize, _step: usize) -> GgmlTensor {
        todo!()
    }

    fn float_remainder(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_remainder_scalar(_lhs: GgmlTensor, _rhs: Scalar) -> GgmlTensor { todo!() }
    fn float_cross(_lhs: GgmlTensor, _rhs: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_select_add(_tensor: GgmlTensor, _dim: usize, _indices: IntTensor<GgmlBackend>, _value: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_sum(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_sum_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_mean_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_cumsum(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_cumprod(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_cummin(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_cummax(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor { todo!() }
    fn float_cast(_tensor: GgmlTensor, _dtype: FloatDType) -> GgmlTensor { todo!() }
    fn float_powf_scalar_impl(_tensor: GgmlTensor, _value: Scalar) -> GgmlTensor { todo!() }
    fn float_tan(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_cosh(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_sinh(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_acos(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_acosh(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_asin(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_asinh(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_atan(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_atanh(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_atan2(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_round(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_floor(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_ceil(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_trunc(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_exp(tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_log(tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_log1p(tensor: GgmlTensor) -> GgmlTensor { todo!() }
}
