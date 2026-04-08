use std::sync::{Arc, Mutex};
use ggml_sys::*;
use crate::device::GgmlDevice;
use crate::graph::GgmlGraphExecutor;
use crate::memory::{LayerWeightCache, KvOffloadManager};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use std::path::PathBuf;

/// A registry of GGML contexts by device.
static REGISTRY: Lazy<Mutex<HashMap<GgmlDevice, Arc<GgmlContext>>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

pub struct GgmlContext {
    pub(crate) ptr: *mut ggml_context,
    pub(crate) backend: *mut ggml_backend,
    pub(crate) sched: *mut ggml_backend_sched,
    pub(crate) executor: Arc<GgmlGraphExecutor>,
    pub(crate) device: GgmlDevice,
    pub(crate) layer_cache: Option<Arc<LayerWeightCache>>,
    pub(crate) kv_offload: Option<Arc<KvOffloadManager>>,
}

impl core::fmt::Debug for GgmlContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GgmlContext")
            .field("device", &self.device)
            .finish()
    }
}

impl GgmlContext {
    pub fn get(device: &GgmlDevice) -> Arc<Self> {
        let mut registry = REGISTRY.lock().unwrap();
        if let Some(ctx) = registry.get(device) {
            return ctx.clone();
        }

        let ctx = Arc::new(Self::new(device.clone()));
        registry.insert(device.clone(), ctx.clone());
        ctx
    }

    pub fn new(device: GgmlDevice) -> Self {
        let (backend, layer_cache, kv_offload) = match &device {
            GgmlDevice::Cpu => {
                let b = unsafe { ggml_backend_cpu_init() };
                (b, None, None)
            }
            GgmlDevice::Metal => {
                #[cfg(target_os = "macos")]
                let b = unsafe { ggml_backend_metal_init() };
                #[cfg(not(target_os = "macos"))]
                let b = unsafe { ggml_backend_cpu_init() };
                (b, None, None)
            }
            GgmlDevice::MetalWithOffload { kv_cache_dir, max_layers_in_ram } => {
                #[cfg(target_os = "macos")]
                let b = unsafe { ggml_backend_metal_init() };
                #[cfg(not(target_os = "macos"))]
                let b = unsafe { ggml_backend_cpu_init() };
                
                // Placeholder path for tests
                let model_path = PathBuf::from("model.gguf");
                let lc = Arc::new(LayerWeightCache::new(model_path, max_layers_in_ram.clone()));
                let kv = Arc::new(KvOffloadManager::new());
                (b, Some(lc), Some(kv))
            }
        };

        unsafe {
            let params = ggml_init_params {
                mem_size: 1 * 1024 * 1024, // Metadata only
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true, 
            };
            let ctx = ggml_init(params);

            let backends = Box::leak(Box::new([backend]));
            let sched = ggml_backend_sched_new(
                backends.as_mut_ptr(),
                std::ptr::null_mut(),
                backends.len() as i32,
                65536,
                false,
                false,
            );

            GgmlContext {
                ptr: ctx,
                backend,
                sched,
                executor: Arc::new(GgmlGraphExecutor::new(ctx, backend, sched)),
                device,
                layer_cache,
                kv_offload,
            }
        }
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe {
            // Note: We don't free the backend or sched anymore because they are shared in the REGISTRY
            // and we leaked the backends array. In a real app we might want a cleaner shutdown.
            ggml_free(self.ptr);
        }
    }
}

unsafe impl Send for GgmlContext {}
unsafe impl Sync for GgmlContext {}
