//! Compatibility shim for bincode 2.x's serde adapter.
//!
//! bincode 2.x dropped the bincode 1.x crate-root `serialize_into` /
//! `deserialize_from` / `serialize` / `deserialize` functions in favour of the
//! [`bincode::serde`] sub-module, which takes an explicit `Config` and reports
//! `EncodeError` / `DecodeError` instead of a single boxed `bincode::Error`.
//! This shim re-exposes the 1.x surface on top of 2.x so the ~43 call sites
//! across this crate keep working unchanged apart from their import path.
//!
//! # The config is load-bearing
//!
//! Every function here uses [`bincode::config::legacy()`], which is
//! **fixed-integer little-endian** — *not* `standard()`'s varint encoding. That
//! is precisely the bincode 1.x default, so bytes written by this shim are
//! byte-for-byte identical to bytes written by bincode 1.3. This crate persists
//! n-gram models, topic models, embedding checkpoints, HNSW indices and RAG
//! backends with these functions, so switching to `standard()` would silently
//! invalidate every model file already on disk.
//!
//! The same shim, with the same reasoning, exists in libdictenstein as
//! `serialization::bincode_compat`.

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::{Read, Write};

/// Unified encode/decode failure, mirroring what bincode 1.x exposed as
/// `bincode::Error` so `#[from]` conversions in downstream error enums keep
/// working.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The value could not be encoded into the requested bincode stream.
    #[error("bincode encode error: {0}")]
    Encode(#[from] bincode::error::EncodeError),
    /// The input stream could not be decoded as the requested value type.
    #[error("bincode decode error: {0}")]
    Decode(#[from] bincode::error::DecodeError),
}

/// Drop-in replacement for `bincode::serialize_into` (bincode 1.x).
/// Takes the writer **by value**, exactly as bincode 1.x did, so existing call
/// sites that pass a `BufWriter<File>` directly keep compiling. `&mut W` also
/// satisfies `W: Write`, so callers that pass a mutable reference work too.
pub fn serialize_into<W: Write, T: Serialize>(mut writer: W, value: &T) -> Result<(), Error> {
    bincode::serde::encode_into_std_write(value, &mut writer, bincode::config::legacy())?;
    Ok(())
}

/// Drop-in replacement for `bincode::deserialize_from` (bincode 1.x).
/// Takes the reader **by value**, exactly as bincode 1.x did.
pub fn deserialize_from<R: Read, T: DeserializeOwned>(mut reader: R) -> Result<T, Error> {
    Ok(bincode::serde::decode_from_std_read(
        &mut reader,
        bincode::config::legacy(),
    )?)
}

/// Drop-in replacement for `bincode::serialize` (bincode 1.x).
pub fn serialize<T: Serialize>(value: &T) -> Result<Vec<u8>, Error> {
    Ok(bincode::serde::encode_to_vec(
        value,
        bincode::config::legacy(),
    )?)
}

/// Drop-in replacement for `bincode::deserialize` (bincode 1.x).
pub fn deserialize<T: DeserializeOwned>(slice: &[u8]) -> Result<T, Error> {
    let (value, _consumed): (T, usize) =
        bincode::serde::decode_from_slice(slice, bincode::config::legacy())?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the wire format. `legacy()` must stay fixed-int little-endian: a `u64`
    /// is exactly 8 LE bytes with no length prefix or varint compaction. If this
    /// fails, every persisted model file written by an earlier build has become
    /// unreadable, which is not something a test suite should discover indirectly.
    #[test]
    fn legacy_config_is_fixint_little_endian() {
        assert_eq!(
            serialize(&1u64).expect("encode u64"),
            vec![1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(serialize(&1u32).expect("encode u32"), vec![1, 0, 0, 0]);
        // A non-negative i64 and the same-valued u64 must be byte-identical.
        assert_eq!(
            serialize(&7i64).expect("encode i64"),
            serialize(&7u64).expect("encode u64")
        );
        // Sequences carry a u64 little-endian length prefix, as in bincode 1.x.
        assert_eq!(
            serialize(&vec![1u8, 2, 3]).expect("encode vec"),
            vec![3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
        );
    }

    #[test]
    fn round_trips_through_both_slice_and_writer() {
        let value = ("libgrammstein".to_string(), 42u64, vec![1i32, -2, 3]);

        let bytes = serialize(&value).expect("serialize");
        let decoded: (String, u64, Vec<i32>) = deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded, value);

        let mut buf = Vec::new();
        serialize_into(&mut buf, &value).expect("serialize_into");
        assert_eq!(buf, bytes, "writer and slice paths must agree");

        let decoded: (String, u64, Vec<i32>) =
            deserialize_from(&mut buf.as_slice()).expect("deserialize_from");
        assert_eq!(decoded, value);
    }
}
