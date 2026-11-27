//! `get_shader_binary_data_ext` passes a 16-aligned `pData` (`VUID-vkGetShaderBinaryDataEXT-
//! None-08499`) whatever the allocator does with byte buffers: this test binary's global
//! allocator returns odd addresses for every alignment-1 request, where a `Vec<u8>` would land.
//! The allocator is a bump allocator over a static arena with no `System` calls, so Miri can
//! check this binary.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tessera_vk::{AlignedBytes, Device, DeviceTable, vk};

/// The arena's size: 34 times the most this binary was measured to allocate (30 544 bytes).
const ARENA_SIZE: usize = 1 << 20;

/// The memory every allocation of this binary comes from.
struct Arena(UnsafeCell<[u8; ARENA_SIZE]>);

// SAFETY: the bytes are reached only through the pointers `Misaligning::alloc` returns, and no
// two of those ranges overlap.
unsafe impl Sync for Arena {}

/// The arena.
static ARENA: Arena = Arena(UnsafeCell::new([0; ARENA_SIZE]));

/// Bytes of `ARENA` handed out so far, alignment gaps included.
static USED: AtomicUsize = AtomicUsize::new(0);

/// A bump allocator over `ARENA`: an alignment-1 request gets an odd address, any other request
/// an address aligned to its alignment. `dealloc` frees nothing, and a request that does not fit
/// gets null.
struct Misaligning;

/// Where a `layout` request goes when `used` bytes of the arena at address `base` are taken:
/// `(start, end)` offsets with `end <= ARENA_SIZE`, or `None` when it does not fit.
fn place(base: usize, used: usize, layout: Layout) -> Option<(usize, usize)> {
    let next = base.checked_add(used)?;
    let addr = if layout.align() == 1 {
        next | 1
    } else {
        next.checked_next_multiple_of(layout.align())?
    };
    let start = addr - base;
    let end = start.checked_add(layout.size())?;
    (end <= ARENA_SIZE).then_some((start, end))
}

// SAFETY: every pointer `alloc` returns is `base.add(start)`, carrying the arena's provenance,
// with `start..start + layout.size()` inside the arena and aligned as `layout` asks (odd for
// alignment 1, which any address meets). Each range is claimed by one successful
// compare-exchange of `USED`, so no two overlap. `dealloc` frees nothing, so no range is reused.
// `alloc` never panics: `place` uses checked arithmetic.
unsafe impl GlobalAlloc for Misaligning {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let base = ARENA.0.get().cast::<u8>();
        let mut used = USED.load(Ordering::Relaxed);
        loop {
            let Some((start, end)) = place(base.addr(), used, layout) else {
                return ptr::null_mut();
            };
            match USED.compare_exchange_weak(used, end, Ordering::Relaxed, Ordering::Relaxed) {
                // SAFETY: `start <= end <= ARENA_SIZE`, so the pointer is inside the arena or
                // one past its end.
                Ok(_) => return unsafe { base.add(start) },
                Err(now) => used = now,
            }
        }
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
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
