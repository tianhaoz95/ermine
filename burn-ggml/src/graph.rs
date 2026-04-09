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
    pub fn new(
        ctx_ptr: *mut ggml_context,
        backend: *mut ggml_backend,
        allocr: *mut ggml_backend_sched,
    ) -> Self {
        Self {
            ctx_ptr,
            backend,
            allocr,
            lock: Mutex::new(()),
        }
    }

    pub unsafe fn compute_graph(&self, gf: *mut ggml_cgraph) -> Result<(), GgmlError> {
        println!("DEBUG: compute_graph: reset");
        ggml_backend_sched_reset(self.allocr);

        println!("DEBUG: compute_graph: reserve");
        // Ensure enough workspace
        ggml_backend_sched_reserve(self.allocr, gf);

        println!("DEBUG: compute_graph: alloc");
        let ok = ggml_backend_sched_alloc_graph(self.allocr, gf);
        if !ok {
            return Err(GgmlError::AllocationFailed);
        }

        println!("DEBUG: compute_graph: compute");
        let status = ggml_backend_sched_graph_compute(self.allocr, gf);
        if status != ggml_status_GGML_STATUS_SUCCESS {
            return Err(GgmlError::ComputeFailed(status));
        }

        println!("DEBUG: compute_graph: done");
        Ok(())
    }
}
