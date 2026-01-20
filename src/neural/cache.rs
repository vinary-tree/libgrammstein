//! Key-Value cache for efficient transformer inference.
//!
//! This module provides caching mechanisms for transformer model inference,
//! reducing redundant computation when processing sequences incrementally.

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

use super::{NeuralError, Result};

/// Configuration for KV cache.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Maximum sequence length to cache.
    pub max_seq_len: usize,
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Data type for cache tensors.
    pub dtype: DType,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 8192,
            num_layers: 12,
            num_heads: 12,
            head_dim: 64,
            dtype: DType::F32,
        }
    }
}

/// Key-Value cache for a single layer.
#[derive(Debug)]
pub struct LayerCache {
    /// Cached key tensor of shape (batch, num_heads, seq_len, head_dim).
    pub key: Option<Tensor>,
    /// Cached value tensor of shape (batch, num_heads, seq_len, head_dim).
    pub value: Option<Tensor>,
    /// Current sequence length in cache.
    pub seq_len: usize,
}

impl LayerCache {
    /// Create a new empty layer cache.
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
            seq_len: 0,
        }
    }

    /// Update cache with new key-value pairs.
    ///
    /// Concatenates new KV along the sequence dimension.
    pub fn update(&mut self, new_key: Tensor, new_value: Tensor) -> Result<(Tensor, Tensor)> {
        let (key, value) = match (&self.key, &self.value) {
            (Some(k), Some(v)) => {
                // Concatenate along sequence dimension (dim 2)
                let key = Tensor::cat(&[k, &new_key], 2)?;
                let value = Tensor::cat(&[v, &new_value], 2)?;
                (key, value)
            }
            _ => (new_key, new_value),
        };

        self.seq_len = key.dim(2)?;
        self.key = Some(key.clone());
        self.value = Some(value.clone());

        Ok((key, value))
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
        self.seq_len = 0;
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.key.is_none()
    }
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-layer KV cache for transformer inference.
#[derive(Debug)]
pub struct KvCache {
    /// Per-layer caches.
    layers: Vec<LayerCache>,
    /// Configuration.
    config: CacheConfig,
    /// Device for cache tensors.
    device: Device,
}

impl KvCache {
    /// Create a new KV cache with the given configuration.
    pub fn new(config: CacheConfig, device: Device) -> Self {
        let layers = (0..config.num_layers).map(|_| LayerCache::new()).collect();

        Self {
            layers,
            config,
            device,
        }
    }

    /// Get the cache for a specific layer.
    pub fn layer(&self, layer_idx: usize) -> Option<&LayerCache> {
        self.layers.get(layer_idx)
    }

    /// Get mutable cache for a specific layer.
    pub fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Update cache for a specific layer.
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        key: Tensor,
        value: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| NeuralError::Inference(format!("Invalid layer index: {}", layer_idx)))?;

        layer.update(key, value)
    }

    /// Get current sequence length in cache.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|l| l.is_empty())
    }

    /// Get the configuration.
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Preallocate cache tensors for a given batch size.
    ///
    /// This can improve performance by avoiding repeated allocations.
    pub fn preallocate(&mut self, batch_size: usize) -> Result<()> {
        let shape = (
            batch_size,
            self.config.num_heads,
            self.config.max_seq_len,
            self.config.head_dim,
        );

        for layer in &mut self.layers {
            if layer.key.is_none() {
                // Initialize with zeros - will be overwritten during forward pass
                layer.key = Some(Tensor::zeros(shape, self.config.dtype, &self.device)?);
                layer.value = Some(Tensor::zeros(shape, self.config.dtype, &self.device)?);
            }
        }

        Ok(())
    }
}

/// Sliding window cache for very long sequences.
///
/// Keeps only the last N tokens to bound memory usage.
#[derive(Debug)]
pub struct SlidingWindowCache {
    /// Inner KV cache.
    inner: KvCache,
    /// Maximum window size.
    window_size: usize,
}

impl SlidingWindowCache {
    /// Create a new sliding window cache.
    pub fn new(config: CacheConfig, window_size: usize, device: Device) -> Self {
        Self {
            inner: KvCache::new(config, device),
            window_size,
        }
    }

    /// Update cache, evicting old entries if window is exceeded.
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        key: Tensor,
        value: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let result = self.inner.update_layer(layer_idx, key, value)?;

        // Check if we need to evict old entries
        if let Some(layer) = self.inner.layer_mut(layer_idx) {
            if layer.seq_len > self.window_size {
                // Truncate to window size (keep last window_size tokens)
                if let (Some(k), Some(v)) = (&layer.key, &layer.value) {
                    let start = layer.seq_len - self.window_size;
                    let k_windowed = k.narrow(2, start, self.window_size)?;
                    let v_windowed = v.narrow(2, start, self.window_size)?;
                    layer.key = Some(k_windowed);
                    layer.value = Some(v_windowed);
                    layer.seq_len = self.window_size;
                }
            }
        }

        Ok(result)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get current sequence length.
    pub fn seq_len(&self) -> usize {
        self.inner.seq_len()
    }

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Cache for storing sentence embeddings to avoid recomputation.
///
/// This implementation uses lock-free data structures for concurrent access:
/// - `DashMap` for O(1) concurrent entry access without global locks
/// - `parking_lot::Mutex<VecDeque>` for O(1) LRU eviction tracking
/// - `Arc<[f32]>` for zero-copy embedding sharing across threads
#[derive(Debug)]
pub struct EmbeddingCache {
    /// Cached embeddings keyed by text hash (lock-free per-entry access).
    entries: dashmap::DashMap<u64, std::sync::Arc<[f32]>>,
    /// Maximum number of entries.
    max_entries: usize,
    /// Access order for LRU eviction (O(1) push/pop with VecDeque).
    access_order: parking_lot::Mutex<std::collections::VecDeque<u64>>,
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl EmbeddingCache {
    /// Create a new embedding cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: dashmap::DashMap::with_capacity(max_entries),
            max_entries,
            access_order: parking_lot::Mutex::new(std::collections::VecDeque::with_capacity(max_entries)),
        }
    }

    /// Get cached embedding if available.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent access.
    /// Returns a cloned `Arc` for zero-copy sharing.
    pub fn get(&self, text: &str) -> Option<std::sync::Arc<[f32]>> {
        let hash = Self::hash_text(text);

        if let Some(entry) = self.entries.get(&hash) {
            // Update access order for LRU (brief lock on VecDeque only)
            {
                let mut order = self.access_order.lock();
                // Remove and re-add to back (O(n) scan, but VecDeque operations are O(1))
                if let Some(pos) = order.iter().position(|&h| h == hash) {
                    order.remove(pos);
                }
                order.push_back(hash);
            }
            Some(std::sync::Arc::clone(&entry))
        } else {
            None
        }
    }

    /// Insert embedding into cache.
    ///
    /// This method takes `&self` instead of `&mut self`, enabling concurrent access.
    pub fn insert(&self, text: &str, embedding: Vec<f32>) {
        let hash = Self::hash_text(text);
        let embedding: std::sync::Arc<[f32]> = embedding.into();

        // Check if we need to evict (brief lock scope)
        let should_evict = self.entries.len() >= self.max_entries && !self.entries.contains_key(&hash);

        if should_evict {
            let oldest = {
                let mut order = self.access_order.lock();
                order.pop_front() // O(1) eviction
            };
            if let Some(oldest_hash) = oldest {
                self.entries.remove(&oldest_hash);
            }
        }

        // Insert new entry
        self.entries.insert(hash, embedding);

        // Update access order
        {
            let mut order = self.access_order.lock();
            // Remove if exists (update case)
            if let Some(pos) = order.iter().position(|&h| h == hash) {
                order.remove(pos);
            }
            order.push_back(hash);
        }
    }

    /// Clear the cache.
    pub fn clear(&self) {
        self.entries.clear();
        self.access_order.lock().clear();
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hash text for cache key.
    fn hash_text(text: &str) -> u64 {
        crate::util::hash::safe_hash(text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache() {
        let mut cache = LayerCache::new();
        assert!(cache.is_empty());

        // Would need tensors to test update - skipping for unit test
    }

    #[test]
    fn test_embedding_cache() {
        let cache = EmbeddingCache::new(2);

        cache.insert("hello", vec![1.0, 2.0, 3.0]);
        cache.insert("world", vec![4.0, 5.0, 6.0]);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("hello").as_deref(), Some([1.0, 2.0, 3.0].as_slice()));

        // Insert third entry, should evict oldest
        cache.insert("test", vec![7.0, 8.0, 9.0]);
        assert_eq!(cache.len(), 2);

        // "hello" was accessed more recently, so "world" should be evicted
        assert!(cache.get("world").is_none());
        assert!(cache.get("hello").is_some());
        assert!(cache.get("test").is_some());
    }

    #[test]
    fn test_embedding_cache_lru() {
        let cache = EmbeddingCache::new(3);

        cache.insert("a", vec![1.0]);
        cache.insert("b", vec![2.0]);
        cache.insert("c", vec![3.0]);

        // Access "a" to make it recently used
        let _ = cache.get("a");

        // Insert "d", should evict "b" (least recently used)
        cache.insert("d", vec![4.0]);

        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none()); // Evicted
        assert!(cache.get("c").is_some());
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn test_embedding_cache_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(EmbeddingCache::new(100));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let cache = Arc::clone(&cache);
                thread::spawn(move || {
                    for j in 0..25 {
                        let key = format!("key_{}_{}", i, j);
                        cache.insert(&key, vec![i as f32, j as f32]);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // Should have entries from all threads (up to max_entries)
        assert!(cache.len() <= 100);
        assert!(cache.len() > 0);
    }
}
