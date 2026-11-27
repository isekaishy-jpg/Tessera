//! The Vulkan namespace: every generated type, constant and `Pfn*` function type (named by the
//! rules in `README.md`: `VkBuffer` → `vk::Buffer`, `VK_FORMAT_R8G8B8A8_UNORM` →
//! `vk::Format::R8G8B8A8_UNORM`), the chain traits, and the API version helpers.

pub use crate::chain::{Extends, TaggedStructure};
pub use crate::generated::*;

/// A Vulkan handle type.
pub trait Handle: Copy {
    /// The object type, for debug names (`VkDebugUtilsObjectNameInfoEXT`).
    const OBJECT_TYPE: ObjectType;

    /// The handle as the 64-bit value debug-name structs take.
    fn object_handle(self) -> u64;
}

/// `VK_MAKE_API_VERSION`.
#[inline]
pub const fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

/// `VK_API_VERSION_VARIANT`.
#[inline]
pub const fn api_version_variant(version: u32) -> u32 {
    version >> 29
}

/// `VK_API_VERSION_MAJOR`.
#[inline]
pub const fn api_version_major(version: u32) -> u32 {
    (version >> 22) & 0x7f
}

/// `VK_API_VERSION_MINOR`.
#[inline]
pub const fn api_version_minor(version: u32) -> u32 {
    (version >> 12) & 0x3ff
}

/// `VK_API_VERSION_PATCH`.
#[inline]
pub const fn api_version_patch(version: u32) -> u32 {
    version & 0xfff
}

/// `VK_API_VERSION_1_0`.
pub const API_VERSION_1_0: u32 = make_api_version(0, 1, 0, 0);
/// `VK_API_VERSION_1_1`.
pub const API_VERSION_1_1: u32 = make_api_version(0, 1, 1, 0);
/// `VK_API_VERSION_1_2`.
pub const API_VERSION_1_2: u32 = make_api_version(0, 1, 2, 0);
/// `VK_API_VERSION_1_3`.
pub const API_VERSION_1_3: u32 = make_api_version(0, 1, 3, 0);
/// `VK_API_VERSION_1_4`.
pub const API_VERSION_1_4: u32 = make_api_version(0, 1, 4, 0);
