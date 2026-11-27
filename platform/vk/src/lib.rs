//! Tessera's Vulkan bindings: every type and command of Vulkan 1.0–1.4 and the selected
//! extensions, generated from the pinned SDK registry, over a small hand-written runtime.
//!
//! - [`vk`]: the spec namespace: types, constants, `Pfn*` function types and the chain traits.
//! - [`Entry`], [`Instance`], [`Device`]: loaded dispatch objects; every command is an
//!   `unsafe fn` on the one its first parameter belongs to.
//! - [`EntryTable`], [`InstanceTable`], [`DeviceTable`]: the raw function tables.
//! - [`system_library`], [`Library`]: the one library-loading policy (bare file names, the
//!   system directory only, never freed), which [`Entry::load`] uses for `vulkan-1.dll`.
//! - [`fixed_str`]: the text in a fixed-size `char` array (`vk::LayerProperties::layer_name`).
//! - [`Plain`]: the types byte parameters accept; [`AlignedBytes`]: a byte buffer aligned to 16.
//!
//! Command wrappers follow fixed rules (see `README.md`): success codes are never dropped,
//! count+array pairs are slices, size-then-fill commands return a `Vec` (or fill caller storage
//! with `foo_into`, the only form of a command whose records are returned once), and one trailing
//! output is returned by value. Struct setters borrow what they point to, and `push_next` builds
//! `pNext` chains with type checks. Valid usage is the caller's contract: wrappers check only what
//! they compute themselves (slice lengths, the addresses of byte slices that must be aligned).

#[macro_use]
mod fmt;
#[macro_use]
mod flags;
#[macro_use]
mod chain;
mod bytes;
mod device;
mod entry;
mod generated;
mod instance;
mod len;
mod loader;
mod result;
mod text;
pub mod vk;

pub use bytes::{AlignedBytes, Plain};
pub use device::Device;
pub use entry::Entry;
pub use generated::tables::{DeviceTable, EntryTable, InstanceTable};
pub use instance::Instance;
pub use loader::{Library, system_library};
pub use result::{LoadError, Result};
pub use text::fixed_str;

/// `f`, or a panic naming the command when the table does not hold it.
#[inline]
#[track_caller]
pub(crate) fn loaded<F>(f: Option<F>, name: &str) -> F {
    match f {
        Some(f) => f,
        None => panic!("{name} is not loaded"),
    }
}
