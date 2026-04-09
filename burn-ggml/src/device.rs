use burn::tensor::backend::{Device, DeviceId, DeviceOps};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GgmlDevice {
    Cpu,
    Metal,
    MetalWithOffload {
        kv_cache_dir: PathBuf,
        max_layers_in_ram: usize,
    },
}

impl Default for GgmlDevice {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return GgmlDevice::Metal;
        #[cfg(not(target_os = "macos"))]
        return GgmlDevice::Cpu;
    }
}

impl Device for GgmlDevice {
    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => GgmlDevice::Cpu,
            1 => GgmlDevice::Metal,
            _ => GgmlDevice::Cpu,
        }
    }

    fn to_id(&self) -> DeviceId {
        match self {
            GgmlDevice::Cpu => DeviceId::new(0, 0),
            GgmlDevice::Metal => DeviceId::new(1, 0),
            GgmlDevice::MetalWithOffload { .. } => DeviceId::new(1, 1),
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

impl DeviceOps for GgmlDevice {}
