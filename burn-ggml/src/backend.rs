use crate::{GgmlDevice, GgmlQuantizedTensor, GgmlTensor};
use burn::tensor::backend::{Backend, DTypeUsage, ExecutionError};
use burn::tensor::ops::TransactionOps;
use burn::tensor::DType;
use enumset::EnumSet;
use std::sync::atomic::{AtomicUsize, Ordering};

static PREFETCH_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn get_prefetch_count() -> usize {
    PREFETCH_COUNT.load(Ordering::SeqCst)
}

/// Internal hook for prefetch dispatching.
pub(crate) fn ggml_prefetch_hook<B: Backend>(
    primitive: crate::ops::PrefetchTensorPrimitive<B>,
    device: &B::Device,
) {
    use crate::ops::PrefetchTensorPrimitive;

    // We check if B is GgmlBackend by checking its name.
    if B::name(device).starts_with("ggml-") {
        match primitive {
            PrefetchTensorPrimitive::Float(_)
            | PrefetchTensorPrimitive::Int(_)
            | PrefetchTensorPrimitive::Bool(_) => {
                PREFETCH_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgmlBackend;

impl Backend for GgmlBackend {
    type Device = GgmlDevice;

    type FloatTensorPrimitive = GgmlTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = GgmlTensor;
    type IntElem = i32;

    type BoolTensorPrimitive = GgmlTensor;
    type BoolElem = u8;

    type QuantizedTensorPrimitive = GgmlQuantizedTensor;

    fn name(device: &Self::Device) -> String {
        match device {
            GgmlDevice::Cpu => "ggml-cpu".into(),
            GgmlDevice::Metal => "ggml-metal".into(),
            GgmlDevice::MetalWithOffload { .. } => "ggml-metal-offload".into(),
        }
    }

    fn seed(_device: &Self::Device, _seed: u64) {}

    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn dtype_usage(_device: &Self::Device, _dtype: DType) -> EnumSet<DTypeUsage> {
        EnumSet::empty()
    }
}

impl TransactionOps<GgmlBackend> for GgmlBackend {}
