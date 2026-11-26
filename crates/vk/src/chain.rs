//! `pNext` chains: the traits generated structs implement, and the one link operation
//! `push_next` uses. No stored `pNext` is ever followed: a chain is built only from structs the
//! caller lends, each pushed with no chain of its own.

use crate::vk;
use core::ffi::c_void;

/// A struct that starts with `sType` and `pNext`, laid out as `VkBaseOutStructure`.
///
/// # Safety
///
/// Implemented only by generated `#[repr(C)]` structs whose first two members are `sType`
/// (holding `STRUCTURE_TYPE` after `Default`) and `pNext`; the layout check verifies the offsets.
pub unsafe trait TaggedStructure {
    /// The struct's `VkStructureType`.
    const STRUCTURE_TYPE: vk::StructureType;
}

/// `Self` may be chained into the `pNext` chain of `P` (the registry's `structextends`).
///
/// # Safety
///
/// Implemented only by generated code, from the registry.
pub unsafe trait Extends<P>: TaggedStructure {}

/// Implements [`Extends`] for a generated struct, once per parent:
/// `extends!(Child<'_> => ParentA<'_>, ParentB<'_>,)`.
macro_rules! extends {
    ($child:ty => $($parent:ty),+ $(,)?) => {
        $(
            // SAFETY: generated from the registry: the parent is in the child's `structextends`.
            unsafe impl $crate::vk::Extends<$parent> for $child {}
        )+
    };
}

/// Links `next` in front of the chain `rest` and returns `next` as the new link.
///
/// Panics when `next` already has a chain: pushing it would follow a pointer the caller may have
/// written by hand.
pub(crate) fn link<T: TaggedStructure>(rest: *mut c_void, next: &mut T) -> *mut c_void {
    let header = core::ptr::from_mut(next).cast::<vk::BaseOutStructure<'static>>();
    // SAFETY: `T: TaggedStructure`, so `next` starts with `sType` and `pNext` laid out as
    // `BaseOutStructure`, and `header` comes from a live `&mut T`.
    let chained = unsafe { (*header).p_next };
    assert!(
        chained.is_null(),
        "push_next: the pushed struct already has a pNext chain; push each struct on its own"
    );
    // SAFETY: as above.
    unsafe { (*header).p_next = rest.cast() };
    header.cast()
}
