//! Loading system libraries and looking commands up by any of their names. [`system_library`]
//! is the one library-loading policy: bare file names only, loaded from the system directory
//! (System32 on Windows, dependencies included), never from the program's directory, the working
//! directory or `PATH`, and never freed. Other platforms are an extension point.

use crate::result::{LoadError, Result};
use crate::vk;
use core::ffi::{CStr, c_void};
use core::mem;
use core::ptr::NonNull;

/// The Vulkan loader's file name, which `Entry::load` loads through [`system_library`].
pub(crate) const VULKAN: &str = "vulkan-1.dll";

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
        assert!(size_of::<Option<F>>() == size_of::<Option<vk::PfnVoidFunction>>());
    }
    // SAFETY: `F` is a function pointer type with `f`'s signature (the caller's contract), so
    // both `Option`s are one nullable function pointer of the same size (asserted above).
    unsafe { mem::transmute_copy::<Option<vk::PfnVoidFunction>, Option<F>>(&f) }
}

/// A library [`system_library`] loaded. It stays loaded for the life of the process, so no
/// function pointer taken from it can dangle.
#[derive(Clone, Copy, Debug)]
pub struct Library {
    /// The module handle.
    module: NonNull<c_void>,
}

// SAFETY: the module handle names a process-wide mapping that is never freed, and looking a
// symbol up in it is allowed from any thread.
unsafe impl Send for Library {}
// SAFETY: as above; `Library` has no interior mutability.
unsafe impl Sync for Library {}

/// True for a bare file name: non-empty, not `.` or `..`, and without a path separator, a drive
/// colon or a NUL.
fn bare(file: &str) -> bool {
    !matches!(file, "" | "." | "..") && !file.contains(['\\', '/', ':', '\0'])
}

#[cfg(windows)]
mod os {
    //! The kernel32 calls the policy uses.

    use core::ffi::{c_char, c_void};

    /// `LOAD_LIBRARY_SEARCH_SYSTEM32`: search the system directory only, for the library and its
    /// dependencies.
    pub(super) const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        /// `LoadLibraryExW`.
        #[link_name = "LoadLibraryExW"]
        pub(super) fn load_library_ex_w(
            name: *const u16,
            file: *mut c_void,
            flags: u32,
        ) -> *mut c_void;
        /// `GetProcAddress`.
        #[link_name = "GetProcAddress"]
        pub(super) fn get_proc_address(
            module: *mut c_void,
            name: *const c_char,
        ) -> Option<crate::vk::PfnVoidFunction>;
        /// `GetLastError`.
        #[link_name = "GetLastError"]
        pub(super) fn get_last_error() -> u32;
    }
}

/// Loads library `file`, a bare file name, from the system directory: `LoadLibraryExW` with
/// `LOAD_LIBRARY_SEARCH_SYSTEM32` searches System32 alone, for the library and its
/// dependencies. A failure is an error, never a wider search. The library is never freed.
///
/// # Errors
///
/// [`LoadError::Name`] when `file` is not a bare file name (refused before anything is loaded);
/// [`LoadError::Library`] with the OS error when the system cannot load it (126,
/// `ERROR_MOD_NOT_FOUND`, when System32 has no such file); [`LoadError::Unsupported`] on
/// platforms without a loader.
///
/// # Safety
///
/// Loading a library runs its initialisation code.
pub unsafe fn system_library(file: &str) -> Result<Library, LoadError> {
    if !bare(file) {
        return Err(LoadError::Name);
    }
    // SAFETY: `file` is a bare file name; the caller accepts its initialisation code running.
    unsafe { load(file) }
}

/// [`system_library`] after the name check.
///
/// # Safety
///
/// As [`system_library`].
#[cfg(windows)]
unsafe fn load(file: &str) -> Result<Library, LoadError> {
    let name: Vec<u16> = file.encode_utf16().chain([0]).collect();
    // SAFETY: `name` is a NUL-terminated UTF-16 string, the reserved file handle is null, and
    // the flag is a documented search flag; the caller accepts the initialisation code running.
    let module = unsafe {
        os::load_library_ex_w(
            name.as_ptr(),
            core::ptr::null_mut(),
            os::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    match NonNull::new(module) {
        Some(module) => Ok(Library { module }),
        None => {
            // SAFETY: `GetLastError` has no preconditions; no other call ran since the failing
            // one.
            let code = unsafe { os::get_last_error() };
            Err(LoadError::Library { code })
        }
    }
}

/// No loader for this platform yet.
///
/// # Safety
///
/// Always safe; `unsafe` matches the Windows loader.
#[cfg(not(windows))]
unsafe fn load(_: &str) -> Result<Library, LoadError> {
    Err(LoadError::Unsupported)
}

impl Library {
    /// The function the library exports as `name`, or `None`. Calling it is `unsafe`, with the
    /// signature the library documents.
    pub fn symbol(&self, name: &CStr) -> Option<vk::PfnVoidFunction> {
        #[cfg(windows)]
        {
            // SAFETY: `module` is a loaded library that is never freed, and `name` is
            // NUL-terminated.
            unsafe { os::get_proc_address(self.module.as_ptr(), name.as_ptr()) }
        }
        #[cfg(not(windows))]
        {
            // No `Library` exists on this platform: `load` never returns one.
            let _ = (self.module, name);
            None
        }
    }
}
