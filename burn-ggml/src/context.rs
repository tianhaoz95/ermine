use crate::device::GgmlDevice;
use crate::gguf::GgufIndex;
use crate::graph::GgmlGraphExecutor;
use crate::memory::{KvOffloadManager, LayerWeightCache};
use ggml_sys::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

#[derive(Clone, Copy)]
struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// A registry of GGML contexts by device.
static REGISTRY: Lazy<Mutex<HashMap<GgmlDevice, Arc<GgmlContext>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static BACKENDS: Lazy<Mutex<HashMap<GgmlDevice, SyncPtr<ggml_backend>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct GgmlContext {
    pub(crate) ptr: *mut ggml_context,
    pub(crate) backend: *mut ggml_backend,
    pub(crate) sched: *mut ggml_backend_sched,
    pub(crate) backends_ptr: *mut *mut ggml_backend,
    pub(crate) device: GgmlDevice,
    pub layer_cache: OnceCell<Arc<LayerWeightCache>>,
    pub(crate) kv_offload: Option<Arc<KvOffloadManager>>,
    pub(crate) executor: Arc<GgmlGraphExecutor>,
}

impl core::fmt::Debug for GgmlContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GgmlContext")
            .field("device", &self.device)
            .finish()
    }
}

tokio::task_local! {
    pub static CONTEXT_OVERRIDE: Arc<GgmlContext>;
}

impl GgmlContext {
    pub fn get(device: &GgmlDevice) -> Arc<Self> {
        if let Ok(ctx) = CONTEXT_OVERRIDE.try_with(|ctx| ctx.clone()) {
            return ctx;
        }

        let mut registry = REGISTRY.lock().unwrap();
        if let Some(ctx) = registry.get(device) {
            return ctx.clone();
        }

        let ctx = Arc::new(Self::new(device.clone()));
        registry.insert(device.clone(), ctx.clone());
        ctx
    }

    pub async fn init_cache(&self, index: Arc<GgufIndex>) {
        let max_layers = if let GgmlDevice::MetalWithOffload {
            max_layers_in_ram, ..
        } = &self.device
        {
            *max_layers_in_ram
        } else {
            0
        };

        if max_layers > 0 {
            self.layer_cache
                .get_or_init(|| async {
                    let ctx_arc = Arc::new(GgmlContext::new(self.device.clone()));
                    Arc::new(LayerWeightCache::new(index.clone(), ctx_arc, max_layers))
                })
                .await;
        }
    }

    fn get_backend(device: &GgmlDevice) -> *mut ggml_backend {
        let mut backends = BACKENDS.lock().unwrap();
        if let Some(p) = backends.get(device) {
            return p.0;
        }

        let b = match device {
            GgmlDevice::Cpu => unsafe {
                let b = ggml_backend_cpu_init();
                ggml_backend_cpu_set_n_threads(b, num_cpus::get() as i32);
                b
            },
            GgmlDevice::Metal => {
                #[cfg(target_os = "macos")]
                unsafe {
                    ggml_backend_metal_init()
                }
                #[cfg(not(target_os = "macos"))]
                unsafe {
                    let b = ggml_backend_cpu_init();
                    ggml_backend_cpu_set_n_threads(b, num_cpus::get() as i32);
                    b
                }
            }
            GgmlDevice::MetalWithOffload { .. } => {
                #[cfg(target_os = "macos")]
                unsafe {
                    ggml_backend_metal_init()
                }
                #[cfg(not(target_os = "macos"))]
                unsafe {
                    let b = ggml_backend_cpu_init();
                    ggml_backend_cpu_set_n_threads(b, num_cpus::get() as i32);
                    b
                }
            }
        };
        backends.insert(device.clone(), SyncPtr(b));
        b
    }

    pub fn new_work_context(&self) -> Arc<Self> {
        unsafe {
            let params = ggml_init_params {
                mem_size: 100 * 1024 * 1024,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx_ptr = ggml_init(params);

            let backends = Box::leak(Box::new([self.backend]));
            let sched = ggml_backend_sched_new(
                backends.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
                65536,
                false,
                false,
            );

            Arc::new(GgmlContext {
                ptr: ctx_ptr,
                backend: self.backend,
                sched,
                backends_ptr: backends.as_mut_ptr(),
                device: self.device.clone(),
                layer_cache: OnceCell::new(),
                kv_offload: None,
                executor: Arc::new(GgmlGraphExecutor::new(ctx_ptr, self.backend, sched)),
            })
        }
    }

    pub fn new(device: GgmlDevice) -> Self {
        let backend = Self::get_backend(&device);
        let kv_offload = match &device {
            GgmlDevice::MetalWithOffload { .. } => Some(Arc::new(KvOffloadManager::new())),
            _ => None,
        };

        unsafe {
            let params = ggml_init_params {
                mem_size: 100 * 1024 * 1024, // 100MB for metadata
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ggml_init(params);

            let backends = Box::leak(Box::new([backend]));
            let sched = ggml_backend_sched_new(
                backends.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
                65536,
                false,
                false,
            );

            GgmlContext {
                ptr: ctx,
                backend,
                sched,
                backends_ptr: backends.as_mut_ptr(),
                executor: Arc::new(GgmlGraphExecutor::new(ctx, backend, sched)),
                device,
                layer_cache: OnceCell::new(),
                kv_offload,
            }
        }
    }
}

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe {
            ggml_free(self.ptr);
        }
    }
}

unsafe impl Send for GgmlContext {}
unsafe impl Sync for GgmlContext {}
