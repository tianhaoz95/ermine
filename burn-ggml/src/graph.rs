use ggml_sys::*;
use std::sync::Mutex;

#[derive(Debug)]
pub enum GgmlError {
    AllocationFailed,
    ComputeFailed(ggml_status),
}

pub struct GgmlGraphExecutor {
    pub(crate) ctx_ptr: *mut ggml_context,
    pub(crate) backend: *mut ggml_backend,
    pub(crate) allocr: *mut ggml_backend_sched,
    pub lock: Mutex<()>,
}

impl GgmlGraphExecutor {
    pub fn new(ctx_ptr: *mut ggml_context, backend: *mut ggml_backend, allocr: *mut ggml_backend_sched) -> Self {
        Self {
            ctx_ptr,
            backend,
            allocr,
            lock: Mutex::new(()),
        }
    }

    pub unsafe fn compute_graph(&self, gf: *mut ggml_cgraph) -> Result<(), GgmlError> {
        let _guard = self.lock.lock().unwrap();

        println!("DEBUG: Starting compute_graph");
        ggml_backend_sched_reset(self.allocr);

        // Ensure enough workspace
        ggml_backend_sched_reserve(self.allocr, gf);

        let ok = ggml_backend_sched_alloc_graph(self.allocr, gf);
        if !ok {
            return Err(GgmlError::AllocationFailed);
        }

        let status = ggml_backend_sched_graph_compute(self.allocr, gf);
        if status != ggml_status_GGML_STATUS_SUCCESS {
            return Err(GgmlError::ComputeFailed(status));
        }

        Ok(())
    }
}
