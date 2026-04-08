use std::sync::{Arc, Mutex};
use ggml_sys::*;
use crate::device::GgmlDevice;
use crate::graph::GgmlGraphExecutor;
use crate::memory::{LayerWeightCache, KvOffloadManager};
use std::collections::HashMap;
use once_cell::sync::Lazy;

struct BackendRegistry {
    backends: HashMap<GgmlDevice, ( *mut ggml_backend, *mut ggml_backend_sched)>,
}

unsafe impl Send for BackendRegistry {}
unsafe impl Sync for BackendRegistry {}

static REGISTRY: Lazy<Mutex<BackendRegistry>> = Lazy::new(|| Mutex::new(BackendRegistry {
    backends: HashMap::new(),
}));

pub struct GgmlContext {
    pub(crate) ptr: *mut ggml_context,
    pub(crate) backend: *mut ggml_backend,
    pub(crate) sched: *mut ggml_backend_sched,
    pub(crate) executor: Arc<GgmlGraphExecutor>,
    pub(crate) device: GgmlDevice,
    pub(crate) layer_cache: Option<Arc<LayerWeightCache>>,
    pub(crate) kv_offload: Option<Arc<KvOffloadManager>>,
}

impl std::fmt::Debug for GgmlContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgmlContext")
            .field("device", &self.device)
            .finish()
    }
}

impl GgmlContext {
    pub fn new(device: GgmlDevice) -> Self {
        let mut registry = REGISTRY.lock().unwrap();
        let (backend, sched) = if let Some(entry) = registry.backends.get(&device) {
            *entry
        } else {
            unsafe {
                let (backend, _, _) = match &device {
                    GgmlDevice::Cpu => {
                        let b = ggml_backend_cpu_init();
                        (b, Option::<Arc<LayerWeightCache>>::None, Option::<Arc<KvOffloadManager>>::None)
                        }
                        GgmlDevice::Metal => {
                        #[cfg(target_os = "macos")]
                        let b = ggml_backend_metal_init();
                        #[cfg(not(target_os = "macos"))]
                        let b = ggml_backend_cpu_init();
                        (b, Option::<Arc<LayerWeightCache>>::None, Option::<Arc<KvOffloadManager>>::None)
                        }
                        GgmlDevice::MetalWithOffload { .. } => {
                        #[cfg(target_os = "macos")]
                        let b = ggml_backend_metal_init();
                        #[cfg(not(target_os = "macos"))]
                        let b = ggml_backend_cpu_init();
                        (b, Option::<Arc<LayerWeightCache>>::None, Option::<Arc<KvOffloadManager>>::None)
                        }
                        };
                let mut backends = [backend];
                let sched = ggml_backend_sched_new(
                    backends.as_mut_ptr(),
                    std::ptr::null_mut(),
                    backends.len() as i32,
                    65536, // Increased from 8192 to support larger models
                    false,
                    false,
                );
                registry.backends.insert(device.clone(), (backend, sched));
                (backend, sched)
            }
        };

        unsafe {
            let params = ggml_init_params {
                mem_size: 1 * 1024 * 1024, // Metadata only
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true, 
            };
            let ctx = ggml_init(params);

            let (layer_cache, kv_offload) = match &device {
                GgmlDevice::MetalWithOffload { kv_cache_dir, max_layers_in_ram } => {
                    let lc = Arc::new(LayerWeightCache::new(kv_cache_dir.clone(), *max_layers_in_ram));
                    let kv = Arc::new(KvOffloadManager::new());
                    (Some(lc), Some(kv))
                }
                _ => (None, None),
            };

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
            ggml_free(self.ptr);
        }
    }
}

unsafe impl Send for GgmlContext {}
unsafe impl Sync for GgmlContext {}
