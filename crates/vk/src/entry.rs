//! `Entry`: the global commands (`vkCreateInstance`, the instance enumerations).

use crate::generated::tables::EntryTable;
use crate::instance::Instance;
use crate::result::{self, LoadError};
use crate::{loader, vk};
use core::ffi::CStr;
use core::fmt;

/// The Vulkan entry point: the global commands, loaded from the system's Vulkan library or from
/// a given `vkGetInstanceProcAddr`.
#[derive(Clone)]
pub struct Entry {
    table: EntryTable,
}

impl Entry {
    /// Loads the system's Vulkan library (`vulkan-1.dll`) and the global commands. The library
    /// stays loaded for the life of the process.
    ///
    /// # Safety
    ///
    /// Loading the library runs its initialisation code.
    pub unsafe fn load() -> core::result::Result<Entry, LoadError> {
        // SAFETY: the caller accepts the library's initialisation code running.
        let get_instance_proc_addr = unsafe { loader::library()? };
        // SAFETY: the pointer is the library's own `vkGetInstanceProcAddr`.
        Ok(unsafe { Entry::from_proc_addr(get_instance_proc_addr) })
    }

    /// Loads the global commands through `get_instance_proc_addr`.
    ///
    /// # Safety
    ///
    /// `get_instance_proc_addr` must be a Vulkan loader's `vkGetInstanceProcAddr`, valid for the
    /// life of the returned `Entry` and everything loaded from it.
    pub unsafe fn from_proc_addr(get_instance_proc_addr: vk::PfnGetInstanceProcAddr) -> Entry {
        let get = |name: &CStr| {
            // SAFETY: the caller guarantees `get_instance_proc_addr`; `name` is NUL-terminated.
            unsafe { get_instance_proc_addr(vk::Instance::null(), name.as_ptr()) }
        };
        // SAFETY: with a null instance, `get` returns a global command or null.
        let mut table = unsafe { EntryTable::load(get) };
        table.get_instance_proc_addr = Some(get_instance_proc_addr);
        Entry { table }
    }

    /// An `Entry` over a table built by the caller (tests, custom loaders).
    pub fn from_parts(table: EntryTable) -> Entry {
        Entry { table }
    }

    /// The function table.
    #[inline]
    pub fn table(&self) -> &EntryTable {
        &self.table
    }

    /// `vkCreateInstance`, returning the loaded [`Instance`].
    ///
    /// # Safety
    ///
    /// The caller upholds every valid-usage rule of `vkCreateInstance`.
    ///
    /// # Panics
    ///
    /// When the command is not loaded.
    pub unsafe fn create_instance(
        &self,
        create_info: &vk::InstanceCreateInfo<'_>,
        allocator: Option<&vk::AllocationCallbacks<'_>>,
    ) -> crate::Result<Instance> {
        let f = crate::loaded(self.table.create_instance, "vkCreateInstance");
        let mut instance = vk::Instance::null();
        // SAFETY: the caller upholds the valid usage; every pointer comes from a live reference
        // or local.
        let code = unsafe {
            f(
                create_info,
                allocator.map_or(core::ptr::null(), core::ptr::from_ref),
                &mut instance,
            )
        };
        result::check(code)?;
        // SAFETY: `instance` was just created through this entry's `vkGetInstanceProcAddr`.
        Ok(unsafe { Instance::load(self, instance) })
    }
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Entry")
    }
}
