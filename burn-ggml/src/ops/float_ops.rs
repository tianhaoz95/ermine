use burn::tensor::ops::{FloatTensorOps, IntTensor};
use crate::{GgmlBackend, GgmlTensor};
use burn::tensor::{Shape, TensorData, Distribution, Scalar, FloatDType, Slice};
use burn::tensor::backend::ExecutionError;
use crate::device::GgmlDevice;
use core::future::Future;
use ggml_sys::*;
use std::ffi::c_void;
use num_traits::ToPrimitive;
use burn::tensor::ops::IntTensorOps;

impl FloatTensorOps<GgmlBackend> for GgmlBackend {
    fn float_empty(shape: Shape, device: &GgmlDevice, _dtype: FloatDType) -> GgmlTensor {
        let ctx = crate::context::GgmlContext::get(device);
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
            let executor = tensor.ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            
            let buft = ggml_backend_cpu_buffer_type();
            if ggml_get_no_alloc(tensor.ctx.ptr) {
                ggml_backend_alloc_ctx_tensors_from_buft(tensor.ctx.ptr, buft);
            }
            
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
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

    fn float_slice(tensor: GgmlTensor, ranges: &[Slice]) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        
        if tensor.shape.len() == 1 {
            let range = &ranges[0];
            let start = range.start as usize;
            let end = range.end.unwrap_or(tensor.shape[0] as isize) as usize;
            let size = end - start;
            
            unsafe {
                let offset = start * std::mem::size_of::<f32>();
                let t = ggml_view_1d(ctx.ptr, tensor.ptr, size as i64, offset);
                GgmlTensor::from_raw(t, ctx)
            }
        } else if tensor.shape.len() == 2 {
            let r0 = &ranges[0];
            let r1 = if ranges.len() > 1 { &ranges[1] } else { &Slice::new(0, None, 1) };
            
            let start0 = r0.start as usize;
            let end0 = r0.end.unwrap_or(tensor.shape[0] as isize) as usize;
            let start1 = r1.start as usize;
            let end1 = r1.end.unwrap_or(tensor.shape[1] as isize) as usize;
            
            let ne0 = tensor.shape[1]; // GGML ne0
            
            let size0 = end0 - start0;
            let size1 = end1 - start1;
            
            unsafe {
                let offset = (start0 * ne0 + start1) * std::mem::size_of::<f32>();
                let t = ggml_view_2d(ctx.ptr, tensor.ptr, size1 as i64, size0 as i64, ne0 * std::mem::size_of::<f32>(), offset);
                GgmlTensor::from_raw(t, ctx)
            }
        } else {
            todo!("Slicing for dims > 2 not implemented")
        }
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
            let rhs_ptr = if lhs.ctx.backend == rhs.ctx.backend {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let executor_rhs = rhs.ctx.executor.clone();
                    let _guard_rhs = executor_rhs.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let executor_lhs = ctx.executor.clone();
                    let _guard_lhs = executor_lhs.lock.lock().unwrap();
                    let buft = ggml_backend_cpu_buffer_type();
                    if ggml_get_no_alloc(ctx.ptr) {
                        ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft);
                    }
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_add(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_sub(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if lhs.ctx.backend == rhs.ctx.backend {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let executor_rhs = rhs.ctx.executor.clone();
                    let _guard_rhs = executor_rhs.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let executor_lhs = ctx.executor.clone();
                    let _guard_lhs = executor_lhs.lock.lock().unwrap();
                    let buft = ggml_backend_cpu_buffer_type();
                    if ggml_get_no_alloc(ctx.ptr) {
                        ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft);
                    }
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_sub(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_mul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if lhs.ctx.backend == rhs.ctx.backend {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let executor_rhs = rhs.ctx.executor.clone();
                    let _guard_rhs = executor_rhs.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let executor_lhs = ctx.executor.clone();
                    let _guard_lhs = executor_lhs.lock.lock().unwrap();
                    let buft = ggml_backend_cpu_buffer_type();
                    if ggml_get_no_alloc(ctx.ptr) {
                        ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft);
                    }
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_mul(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_div(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if lhs.ctx.backend == rhs.ctx.backend {
                rhs.ptr
            } else {
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let executor_rhs = rhs.ctx.executor.clone();
                    let _guard_rhs = executor_rhs.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let executor_lhs = ctx.executor.clone();
                    let _guard_lhs = executor_lhs.lock.lock().unwrap();
                    let buft = ggml_backend_cpu_buffer_type();
                    if ggml_get_no_alloc(ctx.ptr) {
                        ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft);
                    }
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            let t = ggml_div(ctx.ptr, lhs.ptr, rhs_ptr);
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
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
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_matmul(lhs: GgmlTensor, rhs: GgmlTensor) -> GgmlTensor {
        let ctx = lhs.ctx.clone();
        unsafe {
            let rhs_ptr = if lhs.ctx.backend == rhs.ctx.backend {
                rhs.ptr
            } else {
                // ... fallback copy if different backends ...
                let mut dims = [1i64; 4];
                for (i, &d) in rhs.shape.iter().rev().enumerate() {
                    dims[i] = d as i64;
                }
                let rhs_new = ggml_new_tensor(ctx.ptr, rhs.dtype, rhs.shape.len() as i32, dims.as_ptr());
                {
                    let executor_rhs = rhs.ctx.executor.clone();
                    let _guard_rhs = executor_rhs.lock.lock().unwrap();
                    let mut rhs_data = vec![0u8; ggml_nbytes(rhs.ptr)];
                    ggml_backend_tensor_get(rhs.ptr, rhs_data.as_mut_ptr() as *mut c_void, 0, rhs_data.len());
                    
                    let executor_lhs = ctx.executor.clone();
                    let _guard_lhs = executor_lhs.lock.lock().unwrap();
                    let buft = ggml_backend_cpu_buffer_type();
                    if ggml_get_no_alloc(ctx.ptr) {
                        ggml_backend_alloc_ctx_tensors_from_buft(ctx.ptr, buft);
                    }
                    ggml_backend_tensor_set(rhs_new, rhs_data.as_ptr() as *const c_void, 0, rhs_data.len());
                }
                rhs_new
            };

            // In GGML, ggml_mul_mat(w, x) where w is weights and x is activations.
            // w: [d_in, d_out], x: [d_in, seq] -> result: [seq, d_out]
            let t = ggml_mul_mat(ctx.ptr, rhs_ptr, lhs.ptr);
            
            let gf = ggml_new_graph(ctx.ptr);
            ggml_build_forward_expand(gf, t);
            
            {
                let executor = ctx.executor.clone();
                let _guard = executor.lock.lock().unwrap();
                executor.compute_graph(gf).expect("Compute failed");
            }
            
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_swap_dims(tensor: GgmlTensor, _dim1: usize, _dim2: usize) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_transpose(ctx.ptr, tensor.ptr);
            GgmlTensor::from_raw(t, ctx.clone())
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
            let executor = t.ctx.executor.clone();
            let _guard = executor.lock.lock().unwrap();
            let buft = ggml_backend_cpu_buffer_type();
            if ggml_get_no_alloc(t.ctx.ptr) {
                ggml_backend_alloc_ctx_tensors_from_buft(t.ctx.ptr, buft);
            }
            std::ptr::write_bytes(ggml_get_data(t.ptr), 0, ggml_nbytes(t.ptr));
        }
        t
    }

    fn float_ones(shape: Shape, device: &GgmlDevice, _dtype: FloatDType) -> GgmlTensor {
        let n = shape.iter().product::<usize>();
        let data = vec![1.0f32; n];
        Self::float_from_data(TensorData::new(data, shape), device)
    }

    fn float_into_int(_tensor: GgmlTensor) -> IntTensor<GgmlBackend> {
        todo!()
    }

    fn float_powf_scalar(_tensor: GgmlTensor, _value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_sqrt(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_abs(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_cos(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_sin(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_tanh(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_erf(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_argmax(tensor: GgmlTensor, dim: usize) -> IntTensor<GgmlBackend> {
        let data = futures::executor::block_on(Self::float_into_data(tensor.clone())).unwrap();
        let shape = tensor.shape.clone();
        let values = data.as_slice::<f32>().unwrap();
        
        let mut result_shape_dims = shape.clone();
        result_shape_dims.remove(dim);
        
        let mut argmax_indices = Vec::new();
        
        if dim == 0 && shape.len() == 1 {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            for (i, &v) in values.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            argmax_indices.push(max_idx as i32);
        } else {
            todo!("General argmax not implemented for shape {:?} dim {}", shape, dim)
        }
        
        GgmlBackend::int_from_data(
            TensorData::new(argmax_indices, result_shape_dims),
            &tensor.ctx.device
        )
    }

    fn float_argmin(_tensor: GgmlTensor, _dim: usize) -> IntTensor<GgmlBackend> {
        todo!()
    }

    fn float_max_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn float_max_dim_with_indices(_tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, IntTensor<GgmlBackend>) {
        todo!()
    }

    fn float_min_dim(_tensor: GgmlTensor, _dim: usize) -> GgmlTensor {
        todo!()
    }

    fn float_min_dim_with_indices(_tensor: GgmlTensor, _dim: usize) -> (GgmlTensor, IntTensor<GgmlBackend>) {
        todo!()
    }

    fn float_recip(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_transpose(tensor: GgmlTensor) -> GgmlTensor {
        let ctx = tensor.ctx.clone();
        unsafe {
            let t = ggml_transpose(ctx.ptr, tensor.ptr);
            GgmlTensor::from_raw(t, ctx.clone())
        }
    }

    fn float_permute(_tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn float_flip(_tensor: GgmlTensor, _dims: &[usize]) -> GgmlTensor {
        todo!()
    }

    fn float_mask_where(_tensor: GgmlTensor, _mask: GgmlTensor, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_mask_fill(_tensor: GgmlTensor, _mask: GgmlTensor, _value: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_gather(_dim: usize, _tensor: GgmlTensor, _indices: IntTensor<GgmlBackend>) -> GgmlTensor {
        todo!()
    }

    fn float_scatter_add(_dim: usize, _tensor: GgmlTensor, _indices: IntTensor<GgmlBackend>, _value: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_select(tensor: GgmlTensor, dim: usize, indices: IntTensor<GgmlBackend>) -> GgmlTensor {
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

    fn float_clamp(_tensor: GgmlTensor, _min: Scalar, _max: Scalar) -> GgmlTensor {
        todo!()
    }

    fn float_powf(_lhs: GgmlTensor, _rhs: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_sign(_tensor: GgmlTensor) -> GgmlTensor {
        todo!()
    }

    fn float_expand(_tensor: GgmlTensor, _shape: Shape) -> GgmlTensor {
        todo!()
    }

    fn float_unfold(_tensor: GgmlTensor, _dim: usize, _size: usize, _step: usize) -> GgmlTensor {
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
    fn float_exp(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_log(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
    fn float_log1p(_tensor: GgmlTensor) -> GgmlTensor { todo!() }
}
