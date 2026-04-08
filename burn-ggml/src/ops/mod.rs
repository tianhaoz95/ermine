pub mod float_ops;
pub mod int_ops;
pub mod bool_ops;
pub mod module_ops;
pub mod activation_ops;
pub mod quant_ops;
pub mod prefetch;
pub mod offload;

pub use prefetch::*;
pub use offload::*;

use burn::tensor::backend::Backend;

/// Internal function to dispatch prefetch to backends.
/// This is used by the TensorPrefetch extension trait.
pub(crate) fn prefetch_backend_dispatch<B: Backend>(primitive: PrefetchTensorPrimitive<B>, device: &B::Device) {
    // We provide a way for our specific backends to "hook" into this.
    crate::backend::ggml_prefetch_hook::<B>(primitive.clone(), device);
    crate::ops::offload::offload_prefetch_hook::<B>(primitive, device);
}
