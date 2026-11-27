//! Byte storage: [`Plain`], the types whose values may be read from or written as raw bytes, and
//! [`AlignedBytes`], a byte buffer aligned to 16.

use core::{fmt, slice};

/// A type whose values are plain bytes: byte parameters and byte setters take `&[T]` for any
/// `T: Plain`, and driver output written into such a slice is a valid value.
///
/// Implemented for the fixed-width integers, `usize`, `isize`, `f32`, `f64`, arrays of `Plain`
/// types, and the generated enums, flags, handles and padding-free plain structs. Callers
/// implement it for their own `#[repr(C)]` types (push constants, buffer contents).
///
/// # Safety
///
/// Every bit pattern of `size_of::<Self>()` bytes is a valid value of the type, and the type has
/// no padding bytes (so its bytes are always initialised). `bool`, references, `NonZero*`,
/// `char`, enums with invalid discriminants and types with padding do not qualify.
pub unsafe trait Plain: Copy + 'static {}

// SAFETY: integers and floats accept every bit pattern and have no padding.
unsafe impl Plain for u8 {}
// SAFETY: as above.
unsafe impl Plain for u16 {}
// SAFETY: as above.
unsafe impl Plain for u32 {}
// SAFETY: as above.
unsafe impl Plain for u64 {}
// SAFETY: as above.
unsafe impl Plain for usize {}
// SAFETY: as above.
unsafe impl Plain for i8 {}
// SAFETY: as above.
unsafe impl Plain for i16 {}
// SAFETY: as above.
unsafe impl Plain for i32 {}
// SAFETY: as above.
unsafe impl Plain for i64 {}
// SAFETY: as above.
unsafe impl Plain for isize {}
// SAFETY: as above.
unsafe impl Plain for f32 {}
// SAFETY: as above.
unsafe impl Plain for f64 {}
// SAFETY: an array of `Plain` elements has no padding between elements (its size is `N` times
// theirs) and accepts every bit pattern its elements accept.
unsafe impl<T: Plain, const N: usize> Plain for [T; N] {}

/// One block of [`AlignedBytes::ALIGN`] bytes, aligned to it (`align` takes only a literal).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Block([u8; AlignedBytes::ALIGN]);

/// A byte buffer whose first byte is aligned to [`AlignedBytes::ALIGN`] (16), for data that
/// must be aligned: `get_shader_binary_data_ext` returns one, and a shader object's binary code
/// is read from one.
#[derive(Clone)]
pub struct AlignedBytes {
    /// The storage: `len.div_ceil(ALIGN)` blocks.
    blocks: Vec<Block>,
    /// The length in bytes.
    len: usize,
}

impl AlignedBytes {
    /// The alignment of the first byte.
    pub const ALIGN: usize = 16;

    /// `len` zero bytes.
    pub fn zeroed(len: usize) -> AlignedBytes {
        AlignedBytes {
            blocks: vec![Block([0; Self::ALIGN]); len.div_ceil(Self::ALIGN)],
            len,
        }
    }

    /// A copy of `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> AlignedBytes {
        let mut out = AlignedBytes::zeroed(bytes.len());
        out.as_mut_bytes().copy_from_slice(bytes);
        out
    }

    /// The length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when there are no bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Block` is `repr(C)` over `[u8; ALIGN]`, so the blocks are
        // `len.div_ceil(ALIGN) * ALIGN >= len` initialised, contiguous bytes, borrowed for `self`.
        unsafe { slice::from_raw_parts(self.blocks.as_ptr().cast(), self.len) }
    }

    /// The bytes, mutable.
    #[inline]
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_bytes`, borrowed mutably for `self`.
        unsafe { slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast(), self.len) }
    }

    /// The first byte, for a driver to write `len` bytes into. Aligned to 16; dangling (but
    /// aligned) when the buffer is empty.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.blocks.as_mut_ptr().cast()
    }

    /// Shortens the buffer to `len` bytes; panics when `len` is longer than the buffer.
    #[track_caller]
    pub(crate) fn truncate(&mut self, len: usize) {
        assert!(
            len <= self.len,
            "{len} is longer than the buffer ({})",
            self.len
        );
        self.len = len;
        self.blocks.truncate(len.div_ceil(Self::ALIGN));
    }
}

impl PartialEq for AlignedBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for AlignedBytes {}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .finish()
    }
}
