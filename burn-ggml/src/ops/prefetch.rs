use burn::tensor::ops::{BoolTensor, FloatTensor, IntTensor};
use burn::tensor::{backend::Backend, Tensor};
use burn::tensor::{Bool, Float, Int};

/// Extension trait to add prefetch capability to Burn's Tensor.
pub trait TensorPrefetch<B: Backend, const D: usize> {
    /// Hint to the backend to begin loading this tensor's data toward `device`.
    fn prefetch(self, device: &B::Device) -> Self;
}

impl<B: Backend, const D: usize> TensorPrefetch<B, D> for Tensor<B, D, Float> {
    fn prefetch(self, device: &B::Device) -> Self {
        prefetch_dispatch::<B>(self.clone().into_primitive().into(), device);
        self
    }
}

impl<B: Backend, const D: usize> TensorPrefetch<B, D> for Tensor<B, D, Int> {
    fn prefetch(self, device: &B::Device) -> Self {
        prefetch_dispatch::<B>(
            PrefetchTensorPrimitive::Int(self.clone().into_primitive()),
            device,
        );
        self
    }
}

impl<B: Backend, const D: usize> TensorPrefetch<B, D> for Tensor<B, D, Bool> {
    fn prefetch(self, device: &B::Device) -> Self {
        prefetch_dispatch::<B>(
            PrefetchTensorPrimitive::Bool(self.clone().into_primitive()),
            device,
        );
        self
    }
}

/// Tensors to be prefetched toward a device.
#[derive(Default)]
pub struct PrefetchPrimitive<B: Backend> {
    /// Float tensors to prefetch.
    pub floats: Vec<FloatTensor<B>>,
    /// Int tensors to prefetch.
    pub ints: Vec<IntTensor<B>>,
    /// Bool tensors to prefetch.
    pub bools: Vec<BoolTensor<B>>,
}

/// Optional backend capability: begin moving tensors toward `device`
/// asynchronously, before they are needed for computation.
pub trait PrefetchOps<B: Backend> {
    /// Hint to the backend to begin loading these tensors' data toward `device`.
    fn prefetch(primitive: PrefetchPrimitive<B>, device: &B::Device);
}

/// Internal dispatch function for prefetch.
fn prefetch_dispatch<B: Backend>(primitive: PrefetchTensorPrimitive<B>, device: &B::Device) {
    crate::ops::prefetch_backend_dispatch::<B>(primitive, device);
}

/// Enum to represent any tensor primitive kind for prefetching.
#[derive(Clone)]
pub enum PrefetchTensorPrimitive<B: Backend> {
    /// Float tensor primitive.
    Float(FloatTensor<B>),
    /// Int tensor primitive.
    Int(IntTensor<B>),
    /// Bool tensor primitive.
    Bool(BoolTensor<B>),
}

impl<B: Backend> From<burn::tensor::TensorPrimitive<B>> for PrefetchTensorPrimitive<B> {
    fn from(tp: burn::tensor::TensorPrimitive<B>) -> Self {
        match tp {
            burn::tensor::TensorPrimitive::Float(t) => PrefetchTensorPrimitive::Float(t),
            burn::tensor::TensorPrimitive::QFloat(t) => {
                PrefetchTensorPrimitive::Float(B::dequantize(t))
            }
        }
    }
}
