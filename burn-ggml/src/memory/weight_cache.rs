use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use lru::LruCache;
use std::num::NonZeroUsize;

pub trait CacheKey: std::hash::Hash + Eq + Copy + Send + Sync + 'static {}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct LayerKey {
    pub layer: usize,
}
impl CacheKey for LayerKey {}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct ExpertKey {
    pub layer: usize,
    pub expert: usize,
}
impl CacheKey for ExpertKey {}

pub type LayerWeightCache = WeightCache<LayerKey>;
pub type ExpertWeightCache = WeightCache<ExpertKey>;

pub struct WeightSlot {
    pub data: Vec<u8>, // Simplified for now, should be backend buffer
    pub prefetch_complete: Arc<Notify>,
}

pub struct WeightCache<K: CacheKey> {
    model_path: PathBuf,
    max_slots: usize,
    slots: Arc<Mutex<LruCache<K, Arc<WeightSlot>>>>,
}

impl<K: CacheKey> WeightCache<K> {
    pub fn new(model_path: PathBuf, max_slots: usize) -> Self {
        WeightCache {
            model_path,
            max_slots,
            slots: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(max_slots).unwrap()))),
        }
    }

    pub async fn get(&self, key: K) -> Arc<WeightSlot> {
        let mut slots = self.slots.lock().await;
        if let Some(slot) = slots.get(&key) {
            let slot = slot.clone();
            drop(slots);
            slot.prefetch_complete.notified().await;
            return slot;
        }

        // Cache miss
        let slot = Arc::new(WeightSlot {
            data: Vec::new(), // Should load from disk
            prefetch_complete: Arc::new(Notify::new()),
        });
        slots.put(key, slot.clone());
        
        let slot_clone = slot.clone();
        tokio::spawn(async move {
            // Simulate loading
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            slot_clone.prefetch_complete.notify_waiters();
        });

        drop(slots);
        slot.prefetch_complete.notified().await;
        slot
    }

    pub fn prefetch(&self, keys: &[K]) {
        for &key in keys {
            let slots_mutex = self.slots.clone();
            tokio::spawn(async move {
                let mut slots = slots_mutex.lock().await;
                if slots.contains(&key) {
                    return;
                }
                let slot = Arc::new(WeightSlot {
                    data: Vec::new(),
                    prefetch_complete: Arc::new(Notify::new()),
                });
                slots.put(key, slot.clone());
                drop(slots);

                // Simulate loading
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                slot.prefetch_complete.notify_waiters();
            });
        }
    }
}
