//! Fixed-size C name arrays (`char layerName[256]`) as Rust text.

use core::ffi::{CStr, c_char};
use tessera_vk::vk;

/// The NUL-terminated text in `chars` (all of it when there is no NUL).
pub fn fixed(chars: &[c_char]) -> String {
    // `c_char` is a byte; `as u8` reinterprets its sign, never truncates.
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// True when `list` names extension `name`.
pub fn has_extension(list: &[vk::ExtensionProperties], name: &CStr) -> bool {
    let name = name.to_string_lossy();
    list.iter().any(|e| fixed(&e.extension_name) == name)
}
