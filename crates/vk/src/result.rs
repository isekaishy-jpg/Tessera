//! Results: `vk::Result` as the error type, the result checks wrappers use, and `LoadError`.

use crate::vk;
use core::error::Error;
use core::fmt;

/// A Vulkan result: `Err` holds the failing `vk::Result` code unless `E` names another error
/// (`Entry::load` returns `Result<Entry, LoadError>`).
pub type Result<T, E = vk::Result> = core::result::Result<T, E>;

impl fmt::Display for vk::Result {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.name(), self.comment()) {
            (Some(name), Some(comment)) => write!(f, "{name}: {comment}"),
            (Some(name), None) => f.write_str(name),
            (None, _) => write!(f, "unknown VkResult {}", self.as_raw()),
        }
    }
}

impl Error for vk::Result {}

/// `Ok` only on `SUCCESS`: the check of commands whose only success code is `SUCCESS`. Any other
/// code, an unexpected positive one included, is an error, so no code is dropped.
#[inline]
pub(crate) fn check(code: vk::Result) -> Result<()> {
    if code == vk::Result::SUCCESS {
        Ok(())
    } else {
        Err(code)
    }
}

/// `Ok(code)` for any success code (non-negative), `Err` for an error code: the check of commands
/// with several success codes, which the caller gets back.
#[inline]
pub(crate) fn check_code(code: vk::Result) -> Result<vk::Result> {
    if code.as_raw() < 0 {
        Err(code)
    } else {
        Ok(code)
    }
}

/// Why a library could not be loaded (`system_library`, `Entry::load`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// The system could not load the library from the system directory; `code` is the OS error
    /// (`GetLastError`).
    Library {
        /// OS error code.
        code: u32,
    },
    /// The Vulkan library does not export `vkGetInstanceProcAddr`.
    Symbol,
    /// The library name is not a bare file name (it holds a path separator, a drive colon or a
    /// NUL, or is empty, `.` or `..`): libraries load from the system directory only.
    Name,
    /// There is no loader for this platform yet (the Linux loader is an extension point).
    Unsupported,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Library { code } => write!(
                f,
                "cannot load the library from the system directory (OS error {code})"
            ),
            LoadError::Symbol => f.write_str("the Vulkan library has no vkGetInstanceProcAddr"),
            LoadError::Name => f.write_str("the library name is not a bare file name"),
            LoadError::Unsupported => f.write_str("no library loader for this platform yet"),
        }
    }
}

impl Error for LoadError {}
