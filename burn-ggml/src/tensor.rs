use crate::context::GgmlContext;
use burn::tensor::TensorMetadata;
use ggml_sys::*;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct GgmlTensor {
    pub(crate) ptr: *mut ggml_tensor,
    pub(crate) ctx: Arc<GgmlContext>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: ggml_type,
}

unsafe impl Send for GgmlTensor {}
unsafe impl Sync for GgmlTensor {}

impl TensorMetadata for GgmlTensor {
    fn dtype(&self) -> burn::tensor::DType {
        match self.dtype {
            ggml_sys::ggml_type_GGML_TYPE_F32 => burn::tensor::DType::F32,
            ggml_sys::ggml_type_GGML_TYPE_F16 => burn::tensor::DType::F16,
            ggml_sys::ggml_type_GGML_TYPE_I32 => burn::tensor::DType::I32,
            _ => burn::tensor::DType::F32,
        }
    }

    fn shape(&self) -> burn::tensor::Shape {
        burn::tensor::Shape::from(self.shape.clone())
    }
}

impl GgmlTensor {
    pub unsafe fn from_raw(ptr: *mut ggml_tensor, ctx: Arc<GgmlContext>) -> Self {
        let ne = (*ptr).ne;
        let n_dims = ggml_n_dims(ptr) as usize;
        let shape: Vec<usize> = (0..n_dims).rev().map(|i| ne[i] as usize).collect();
        let dtype = (*ptr).type_;
        GgmlTensor {
            ptr,
            ctx,
            shape,
            dtype,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GgmlQuantizedTensor {
    pub inner: GgmlTensor,
    pub scheme: burn::tensor::quantization::QuantScheme,
}

impl burn::tensor::quantization::QTensorPrimitive for GgmlQuantizedTensor {
    fn scheme(&self) -> &burn::tensor::quantization::QuantScheme {
        &self.scheme
    }
}

impl TensorMetadata for GgmlQuantizedTensor {
    fn dtype(&self) -> burn::tensor::DType {
        self.inner.dtype()
    }

    fn shape(&self) -> burn::tensor::Shape {
        self.inner.shape()
    }
}
