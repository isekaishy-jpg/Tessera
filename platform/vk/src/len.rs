//! Length conversions for counts that setters and wrappers compute from slices. Every
//! conversion is checked: a slice too long for its count panics instead of being truncated.

/// `len` as a `uint32_t` count; panics naming the count (`field`) when it does not fit.
#[inline]
#[track_caller]
pub(crate) fn count_u32(len: usize, field: &str) -> u32 {
    match u32::try_from(len) {
        Ok(n) => n,
        Err(_) => panic!("{field}: {len} exceeds u32::MAX"),
    }
}

/// `len` as a `uint64_t`/`VkDeviceSize` count.
#[inline]
#[track_caller]
pub(crate) fn count_u64(len: usize) -> u64 {
    match u64::try_from(len) {
        Ok(n) => n,
        Err(_) => panic!("{len} exceeds u64::MAX"),
    }
}

/// A `uint32_t` count as a slice length.
#[inline]
#[track_caller]
pub(crate) fn to_usize(n: u32) -> usize {
    match usize::try_from(n) {
        Ok(n) => n,
        Err(_) => panic!("{n} exceeds usize::MAX"),
    }
}
