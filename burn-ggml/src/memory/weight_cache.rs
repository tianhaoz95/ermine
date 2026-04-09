
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::collections::HashMap;
use crate::tensor::GgmlTensor;
use crate::gguf::GgufIndex;
use crate::context::GgmlContext;

pub trait CacheKey: std::hash::Hash + Eq + Copy + Send + Sync + 'static {
    fn get_tensors(&self, index: &GgufIndex) -> Vec<String>;
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct LayerKey {
    pub layer: usize,
}
impl CacheKey for LayerKey {
    fn get_tensors(&self, index: &GgufIndex) -> Vec<String> {
        index.get_layer_tensors(self.layer)
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}
impl CacheKey for ExpertKey {
    fn get_tensors(&self, index: &GgufIndex) -> Vec<String> {
        index.get_expert_tensors(self.layer, self.expert)
    }
}

pub type LayerWeightCache = WeightCache<LayerKey>;
pub type ExpertWeightCache = WeightCache<ExpertKey>;

pub struct WeightSlot {
    pub tensors: OnceCell<HashMap<String, GgmlTensor>>,
}

pub struct WeightCache<K: CacheKey> {
    index: Arc<GgufIndex>,
    ctx: Arc<GgmlContext>,
    slots: Arc<Mutex<LruCache<K, Arc<WeightSlot>>>>,
}

impl<K: CacheKey> WeightCache<K> {
    pub fn new(index: Arc<GgufIndex>, ctx: Arc<GgmlContext>, max_slots: usize) -> Self {
        WeightCache {
            index,
            ctx,
            slots: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(max_slots).unwrap()))),
        }
    }

    async fn load_slot(index: Arc<GgufIndex>, ctx: Arc<GgmlContext>, key: K, slot: Arc<WeightSlot>) {
        slot.tensors.get_or_init(|| async {
            let res = tokio::task::spawn_blocking(move || {
                let tensor_names = key.get_tensors(&index);
                let mut loaded_tensors = HashMap::new();
                for name in tensor_names {
                    let tensor = unsafe { index.load_tensor(&name, &ctx).expect("Failed to load tensor") };
                    loaded_tensors.insert(name, tensor);
                }
                loaded_tensors
            }).await.expect("Task failed");
            res
        }).await;
    }

    pub async fn get(&self, key: K) -> Arc<WeightSlot> {
        let mut slots = self.slots.lock().await;
        if let Some(slot) = slots.get(&key) {
            let slot = slot.clone();
            drop(slots);
            // Ensure it's loaded (if prefetch is still running)
            Self::load_slot(self.index.clone(), self.ctx.clone(), key, slot.clone()).await;
            return slot;
        }

        // Cache miss
        let slot = Arc::new(WeightSlot {
            tensors: OnceCell::new(),
        });
        slots.put(key, slot.clone());
        drop(slots);
        
        Self::load_slot(self.index.clone(), self.ctx.clone(), key, slot.clone()).await;
        slot
    }

    pub fn prefetch(&self, keys: &[K]) {
        for &key in keys {
            let slots_mutex = self.slots.clone();
            let index = self.index.clone();
            let ctx = self.ctx.clone();
            
            tokio::spawn(async move {
                let mut slots = slots_mutex.lock().await;
                if slots.contains(&key) {
                    return;
                }
                let slot = Arc::new(WeightSlot {
                    tensors: OnceCell::new(),
                });
                slots.put(key, slot.clone());
                drop(slots);

                Self::load_slot(index, ctx, key, slot).await;
            });
        }
    }
}
