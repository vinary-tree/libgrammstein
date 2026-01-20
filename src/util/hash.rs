//! Safe hashing utilities that handle gxhash's SIMD buffer requirements.
//!
//! gxhash uses SSE2/AES operations that read 16-byte chunks, which can read
//! past buffer boundaries for inputs shorter than 16 bytes. This module
//! provides safe wrappers that use FNV-1a for short inputs and gxhash for
//! longer inputs.

use std::hash::{BuildHasher, Hasher};

/// Minimum buffer size for gxhash SIMD operations.
pub const GXHASH_MIN_SIZE: usize = 16;

/// FNV-1a offset basis.
const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// FNV-1a prime.
const FNV_PRIME: u64 = 0x100000001b3;

/// Hash bytes using gxhash for inputs >= 16 bytes, FNV-1a for shorter inputs.
///
/// This function is safe for all input lengths and uses hardware-accelerated
/// hashing (gxhash) when the input is long enough for safe SIMD reads.
#[inline]
pub fn safe_hash(bytes: &[u8]) -> u64 {
    if bytes.len() >= GXHASH_MIN_SIZE {
        gxhash::gxhash64(bytes, 0)
    } else {
        fnv1a(bytes)
    }
}

/// Hash bytes using gxhash with a seed for inputs >= 16 bytes, FNV-1a otherwise.
///
/// The seed is mixed into the hash for position-aware hashing.
#[inline]
pub fn safe_hash_with_seed(bytes: &[u8], seed: u64) -> u64 {
    if bytes.len() >= GXHASH_MIN_SIZE {
        gxhash::gxhash64(bytes, seed as i64)
    } else {
        fnv1a_with_seed(bytes, seed)
    }
}

/// FNV-1a hash for small inputs.
#[inline]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a hash with seed mixing for position-aware hashing.
#[inline]
pub fn fnv1a_with_seed(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed.wrapping_mul(FNV_PRIME);
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A safe BuildHasher that uses gxhash for long inputs and FNV-1a for short inputs.
///
/// Use this with `HashMap`, `HashSet`, or `DashMap` for safe hashing of
/// variable-length keys.
#[derive(Clone, Default)]
pub struct SafeGxBuildHasher;

impl BuildHasher for SafeGxBuildHasher {
    type Hasher = SafeGxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        SafeGxHasher {
            buffer: Vec::with_capacity(64),
        }
    }
}

/// A Hasher that collects bytes and uses gxhash or FNV-1a based on length.
pub struct SafeGxHasher {
    buffer: Vec<u8>,
}

impl Hasher for SafeGxHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    fn finish(&self) -> u64 {
        safe_hash(&self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_hash_short() {
        let hash1 = safe_hash(b"hello");
        let hash2 = safe_hash(b"hello");
        assert_eq!(hash1, hash2);

        let hash3 = safe_hash(b"world");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_safe_hash_long() {
        let long_input = b"this is a longer string that exceeds 16 bytes";
        let hash1 = safe_hash(long_input);
        let hash2 = safe_hash(long_input);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_safe_hash_boundary() {
        // Exactly 16 bytes - should use gxhash
        let exact = b"0123456789abcdef";
        assert_eq!(exact.len(), 16);
        let hash1 = safe_hash(exact);
        let hash2 = safe_hash(exact);
        assert_eq!(hash1, hash2);

        // 15 bytes - should use FNV-1a
        let short = b"0123456789abcde";
        assert_eq!(short.len(), 15);
        let hash3 = safe_hash(short);
        let hash4 = safe_hash(short);
        assert_eq!(hash3, hash4);
    }

    #[test]
    fn test_safe_hash_with_seed() {
        let bytes = b"test";
        let hash1 = safe_hash_with_seed(bytes, 0);
        let hash2 = safe_hash_with_seed(bytes, 1);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_safe_gx_hasher() {
        use std::hash::Hash;

        let mut hasher1 = SafeGxBuildHasher.build_hasher();
        "hello".hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = SafeGxBuildHasher.build_hasher();
        "hello".hash(&mut hasher2);
        let hash2 = hasher2.finish();

        assert_eq!(hash1, hash2);
    }
}
