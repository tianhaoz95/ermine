use burn::tensor::backend::{Backend, ExecutionError, DTypeUsage};
use burn::tensor::ops::TransactionOps;
use burn::tensor::DType;
use crate::{GgmlDevice, GgmlTensor, GgmlQuantizedTensor};
use enumset::EnumSet;

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

    fn seed(_device: &Self::Device, _seed: u64) {
        // TODO
    }

    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn dtype_usage(_device: &Self::Device, _dtype: DType) -> EnumSet<DTypeUsage> {
        EnumSet::empty()
    }
}

impl TransactionOps<GgmlBackend> for GgmlBackend {}
