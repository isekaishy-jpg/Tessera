//! `Device`: a `VkDevice` and its device, queue and command-buffer commands.

use crate::generated::tables::DeviceTable;
use crate::instance::Instance;
use crate::vk;
use core::ffi::CStr;
use core::{fmt, ptr};

/// A loaded `VkDevice`: its handle and the device-level commands, loaded through
/// `vkGetDeviceProcAddr` (no loader trampoline). It does not destroy the device when dropped
/// (`destroy_device` does).
#[derive(Clone)]
pub struct Device {
    /// The `VkDevice`.
    handle: vk::Device,
    /// Its device-level commands.
    table: DeviceTable,
}

impl Device {
    /// Loads the device commands of `device` through `instance`'s `vkGetDeviceProcAddr`.
    ///
    /// # Safety
    ///
    /// `device` must be a live device created from `instance`.
    ///
    /// # Panics
    ///
    /// When `instance` has no `vkGetDeviceProcAddr`.
    pub unsafe fn load(instance: &Instance, device: vk::Device) -> Device {
        let get = crate::loaded(instance.table().get_device_proc_addr, "vkGetDeviceProcAddr");
        // SAFETY: the caller guarantees `device` was created from `instance`, whose
        // `vkGetDeviceProcAddr` is `get`.
        unsafe { Device::load_with(get, device) }
    }

    /// Loads the device commands of `device` through `get`.
    ///
    /// # Safety
    ///
    /// `get` is the `vkGetDeviceProcAddr` of the instance that created the live `device`.
    pub(crate) unsafe fn load_with(get: vk::PfnGetDeviceProcAddr, device: vk::Device) -> Device {
        let load = |name: &CStr| {
            // SAFETY: `get` is the instance's `vkGetDeviceProcAddr`, `device` is live and `name`
            // is NUL-terminated.
            unsafe { get(device, name.as_ptr()) }
        };
        // SAFETY: `load` returns a device command of `device` or null.
        let table = unsafe { DeviceTable::load(load) };
        Device {
            handle: device,
            table,
        }
    }

    /// A `Device` over a handle and a table built by the caller (tests, custom loaders).
    pub fn from_parts(handle: vk::Device, table: DeviceTable) -> Device {
        Device { handle, table }
    }

    /// The `VkDevice`.
    #[inline]
    pub fn handle(&self) -> vk::Device {
        self.handle
    }

    /// The function table.
    #[inline]
    pub fn table(&self) -> &DeviceTable {
        &self.table
    }

    /// `vkDestroyDevice`.
    ///
    /// # Safety
    ///
    /// The caller upholds every valid-usage rule of `vkDestroyDevice`; nothing loaded from this
    /// device is used afterwards.
    ///
    /// # Panics
    ///
    /// When the command is not loaded.
    pub unsafe fn destroy_device(&self, allocator: Option<&vk::AllocationCallbacks<'_>>) {
        let f = crate::loaded(self.table.destroy_device, "vkDestroyDevice");
        // SAFETY: the caller upholds the valid usage; the pointer comes from a live reference.
        unsafe {
            f(self.handle, allocator.map_or(ptr::null(), ptr::from_ref));
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Device").field(&self.handle).finish()
    }
}
