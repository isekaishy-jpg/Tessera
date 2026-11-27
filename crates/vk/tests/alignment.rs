//! `get_shader_binary_data_ext` passes a 16-aligned `pData` (`VUID-vkGetShaderBinaryDataEXT-
//! None-08499`) whatever the allocator does with byte buffers: this test binary's global
//! allocator returns odd addresses for every alignment-1 request, where a `Vec<u8>` would land.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicBool, Ordering};
use std::alloc::System;
use tessera_vk::{AlignedBytes, Device, DeviceTable, vk};

/// Wraps `System`; an alignment-1 request gets the byte after a 2-aligned block, so it is odd.
struct Misaligning;

impl Misaligning {
    /// The layout of the block behind an alignment-1 request of `layout.size()` bytes.
    fn padded(layout: Layout) -> Option<Layout> {
        Layout::from_size_align(layout.size().checked_add(1)?, 2).ok()
    }
}

// SAFETY: every block comes from `System` with a valid layout, and `dealloc` frees it with that
// same layout; an alignment-1 pointer is offset by one byte inside its block, which still holds
// `layout.size()` bytes after it.
unsafe impl GlobalAlloc for Misaligning {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() != 1 {
            // SAFETY: the caller's layout has a non-zero size.
            return unsafe { System.alloc(layout) };
        }
        let Some(padded) = Self::padded(layout) else {
            return ptr::null_mut();
        };
        // SAFETY: `padded` has a non-zero size.
        let block = unsafe { System.alloc(padded) };
        if block.is_null() {
            return block;
        }
        // SAFETY: the block holds `layout.size() + 1` bytes.
        unsafe { block.add(1) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if layout.align() != 1 {
            // SAFETY: `pointer` came from `System.alloc(layout)`.
            return unsafe { System.dealloc(pointer, layout) };
        }
        let Some(padded) = Self::padded(layout) else {
            return;
        };
        // SAFETY: `pointer` is one byte into a block allocated with `padded`.
        unsafe { System.dealloc(pointer.sub(1), padded) };
    }
}

#[global_allocator]
static ALLOCATOR: Misaligning = Misaligning;

/// The binary's length.
const LEN: usize = 20;

/// Set when the mock saw a misaligned `pData`.
static MISALIGNED: AtomicBool = AtomicBool::new(false);

/// `vkGetShaderBinaryDataEXT` returning `LEN` bytes `0, 1, ..`; it records a misaligned `pData`
/// instead of panicking across the FFI boundary.
unsafe extern "system" fn shader_binary(
    _: vk::Device,
    _: vk::ShaderEXT,
    size: *mut usize,
    data: *mut c_void,
) -> vk::Result {
    // SAFETY: the wrapper passes a live size and, when non-null, `*size` bytes at `data`.
    unsafe {
        if data.is_null() {
            *size = LEN;
            return vk::Result::SUCCESS;
        }
        if !data.addr().is_multiple_of(AlignedBytes::ALIGN) {
            MISALIGNED.store(true, Ordering::SeqCst);
        }
        for i in 0..LEN {
            *data.cast::<u8>().add(i) = i as u8;
        }
    }
    vk::Result::SUCCESS
}

#[test]
fn shader_binaries_are_written_to_aligned_storage() {
    // A heap byte buffer, as the old `Vec<u8>` output was.
    let heap: Vec<u8> = (0..LEN as u8).collect();
    assert!(
        !heap.as_ptr().addr().is_multiple_of(2),
        "the allocator misaligns byte buffers"
    );
    let device = Device::from_parts(
        vk::Device::from_raw(1),
        DeviceTable {
            get_shader_binary_data_ext: Some(shader_binary),
            ..Default::default()
        },
    );
    // SAFETY: the mock writes at most the size the wrapper passes.
    let binary = unsafe { device.get_shader_binary_data_ext(vk::ShaderEXT::from_raw(2)) }
        .expect("the mock succeeds");
    assert!(!MISALIGNED.load(Ordering::SeqCst), "pData is aligned to 16");
    assert!(
        binary
            .as_bytes()
            .as_ptr()
            .addr()
            .is_multiple_of(AlignedBytes::ALIGN)
    );
    assert_eq!(binary.as_bytes(), heap.as_slice());
    assert_eq!(binary, AlignedBytes::from_bytes(&heap));
}

#[test]
fn aligned_bytes_keep_their_length_and_alignment() {
    let empty = AlignedBytes::zeroed(0);
    assert!(empty.is_empty() && empty.as_bytes().is_empty());
    let mut bytes = AlignedBytes::zeroed(33);
    assert_eq!(bytes.len(), 33);
    bytes.as_mut_bytes()[32] = 7;
    assert_eq!(bytes.as_bytes()[32], 7);
    assert!(
        bytes
            .as_bytes()
            .as_ptr()
            .addr()
            .is_multiple_of(AlignedBytes::ALIGN)
    );
    assert_eq!(format!("{bytes:?}"), "AlignedBytes { len: 33 }");
    assert_eq!(bytes.clone(), bytes);
}
