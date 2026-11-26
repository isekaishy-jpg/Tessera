//! Loading the Vulkan library (`vulkan-1.dll` on Windows; other platforms are an extension point)
//! and looking commands up by any of their names.

use crate::result::LoadError;
use crate::vk;
use core::ffi::CStr;

/// The first of `names` that `get` resolves.
pub(crate) fn first(
    get: &mut impl FnMut(&CStr) -> Option<vk::PfnVoidFunction>,
    names: &[&CStr],
) -> Option<vk::PfnVoidFunction> {
    names.iter().find_map(|n| get(n))
}

/// `f` as the function pointer type `F` (a table field's type, which infers `F`).
///
/// # Safety
///
/// `F` must be a function pointer type and `f` `None` or a function with `F`'s signature.
#[inline]
pub(crate) unsafe fn cast<F: Copy>(f: Option<vk::PfnVoidFunction>) -> Option<F> {
    const {
        assert!(
            core::mem::size_of::<Option<F>>()
                == core::mem::size_of::<Option<vk::PfnVoidFunction>>()
        );
    }
    // SAFETY: `F` is a function pointer type with `f`'s signature (the caller's contract), so
    // both `Option`s are one nullable function pointer of the same size (asserted above).
    unsafe { core::mem::transmute_copy::<Option<vk::PfnVoidFunction>, Option<F>>(&f) }
}

/// `vkGetInstanceProcAddr` of the system's Vulkan library. The library stays loaded for the life
/// of the process, so no function pointer taken from it can dangle.
///
/// # Safety
///
/// Loading a library runs its initialisation code.
#[cfg(windows)]
pub(crate) unsafe fn library() -> Result<vk::PfnGetInstanceProcAddr, LoadError> {
    use core::ffi::{c_char, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        #[link_name = "LoadLibraryW"]
        fn load_library_w(name: *const u16) -> *mut c_void;
        #[link_name = "GetProcAddress"]
        fn get_proc_address(
            module: *mut c_void,
            name: *const c_char,
        ) -> Option<vk::PfnVoidFunction>;
        #[link_name = "GetLastError"]
        fn get_last_error() -> u32;
    }

    let name: Vec<u16> = "vulkan-1.dll\0".encode_utf16().collect();
    // SAFETY: `name` is a NUL-terminated UTF-16 string; the caller accepts the library's
    // initialisation code running.
    let module = unsafe { load_library_w(name.as_ptr()) };
    if module.is_null() {
        // SAFETY: `GetLastError` has no preconditions; no other call ran since the failing one.
        let code = unsafe { get_last_error() };
        return Err(LoadError::Library { code });
    }
    // SAFETY: `module` is a loaded library and the name is NUL-terminated.
    let get = unsafe { get_proc_address(module, c"vkGetInstanceProcAddr".as_ptr()) };
    match get {
        // SAFETY: the library's `vkGetInstanceProcAddr` has the signature
        // `PfnGetInstanceProcAddr`; both are one function pointer.
        Some(f) => Ok(unsafe {
            core::mem::transmute::<vk::PfnVoidFunction, vk::PfnGetInstanceProcAddr>(f)
        }),
        None => Err(LoadError::Symbol),
    }
}

/// No loader for this platform yet.
///
/// # Safety
///
/// Always safe; `unsafe` matches the Windows loader.
#[cfg(not(windows))]
pub(crate) unsafe fn library() -> Result<vk::PfnGetInstanceProcAddr, LoadError> {
    Err(LoadError::Unsupported)
}
