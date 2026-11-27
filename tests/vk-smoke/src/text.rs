//! Names in fixed-size C arrays (`char extensionName[256]`) compared with Rust text.

use core::ffi::CStr;
use tessera_vk::{fixed_str, vk};

/// True when `list` names extension `name`.
pub fn has_extension(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    let name = name.to_string_lossy();
    list.iter().any(|e| fixed_str(&e.extension_name) == name)
}

/// True when `list` names layer `name`.
pub fn has_layer(list: &[vk::LayerProperties], name: &CStr) -> bool {
    let name = name.to_string_lossy();
    list.iter().any(|l| fixed_str(&l.layer_name) == name)
}
