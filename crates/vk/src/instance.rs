//! `Instance`: a `VkInstance` and its instance-level commands.

use crate::device::Device;
use crate::entry::Entry;
use crate::generated::tables::InstanceTable;
use crate::result;
use crate::vk;
use core::ffi::CStr;
use core::fmt;

/// A loaded `VkInstance`: its handle and the instance and physical-device commands. It does not
/// destroy the instance when dropped (`destroy_instance` does).
#[derive(Clone)]
pub struct Instance {
    handle: vk::Instance,
    table: InstanceTable,
}

impl Instance {
    /// Loads the instance commands of `instance` through `entry`'s `vkGetInstanceProcAddr`.
    ///
    /// # Safety
    ///
    /// `instance` must be a live instance created through `entry`'s loader.
    ///
    /// # Panics
    ///
    /// When `entry` has no `vkGetInstanceProcAddr`.
    pub unsafe fn load(entry: &Entry, instance: vk::Instance) -> Instance {
        let get = crate::loaded(
            entry.table().get_instance_proc_addr,
            "vkGetInstanceProcAddr",
        );
        let load = |name: &CStr| {
            // SAFETY: `get` is the loader's `vkGetInstanceProcAddr`, `instance` is live and
            // `name` is NUL-terminated.
            unsafe { get(instance, name.as_ptr()) }
        };
        // SAFETY: `load` returns an instance command of `instance` or null.
        let table = unsafe { InstanceTable::load(load) };
        Instance {
            handle: instance,
            table,
        }
    }

    /// An `Instance` over a handle and a table built by the caller (tests, custom loaders).
    pub fn from_parts(handle: vk::Instance, table: InstanceTable) -> Instance {
        Instance { handle, table }
    }

    /// The `VkInstance`.
    #[inline]
    pub fn handle(&self) -> vk::Instance {
        self.handle
    }

    /// The function table.
    #[inline]
    pub fn table(&self) -> &InstanceTable {
        &self.table
    }

    /// `vkDestroyInstance`.
    ///
    /// # Safety
    ///
    /// The caller upholds every valid-usage rule of `vkDestroyInstance`; nothing loaded from
    /// this instance is used afterwards.
    ///
    /// # Panics
    ///
    /// When the command is not loaded.
    pub unsafe fn destroy_instance(&self, allocator: Option<&vk::AllocationCallbacks<'_>>) {
        let f = crate::loaded(self.table.destroy_instance, "vkDestroyInstance");
        // SAFETY: the caller upholds the valid usage; the pointer comes from a live reference.
        unsafe {
            f(
                self.handle,
                allocator.map_or(core::ptr::null(), core::ptr::from_ref),
            );
        }
    }

    /// `vkCreateDevice`, returning the loaded [`Device`].
    ///
    /// # Safety
    ///
    /// The caller upholds every valid-usage rule of `vkCreateDevice`.
    ///
    /// # Panics
    ///
    /// When `vkCreateDevice` or `vkGetDeviceProcAddr` is not loaded.
    pub unsafe fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo<'_>,
        allocator: Option<&vk::AllocationCallbacks<'_>>,
    ) -> crate::Result<Device> {
        let f = crate::loaded(self.table.create_device, "vkCreateDevice");
        let mut device = vk::Device::null();
        // SAFETY: the caller upholds the valid usage; every pointer comes from a live reference
        // or local.
        let code = unsafe {
            f(
                physical_device,
                create_info,
                allocator.map_or(core::ptr::null(), core::ptr::from_ref),
                &mut device,
            )
        };
        result::check(code)?;
        // SAFETY: `device` was just created from this instance.
        Ok(unsafe { Device::load(self, device) })
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Instance").field(&self.handle).finish()
    }
}
