//! Owners: every Vulkan object the program creates is held by one of these from the moment it
//! exists, so every exit path destroys it exactly once. Command buffers are the exception: their
//! pool frees them.
//!
//! A struct's owned fields drop in declaration order, and locals in reverse order of
//! declaration, so children are declared after their parents and dropped before them.

use core::ops::Deref;
use tessera_vk::{Device, Instance, vk};

/// A handle created from `parent`, destroyed by `destroy` when dropped.
pub struct Owned<'p, P, H: Copy> {
    parent: &'p P,
    handle: H,
    destroy: unsafe fn(&P, H),
}

impl<'p, P, H: Copy> Owned<'p, P, H> {
    /// Takes ownership of `handle`.
    ///
    /// # Safety
    ///
    /// `handle` is live and was created from `parent`, `destroy` destroys it, and nothing else
    /// destroys it.
    pub unsafe fn new(parent: &'p P, handle: H, destroy: unsafe fn(&P, H)) -> Owned<'p, P, H> {
        Owned {
            parent,
            handle,
            destroy,
        }
    }

    /// The handle, still owned.
    pub fn handle(&self) -> H {
        self.handle
    }

    /// The handle, released: the caller destroys it, before its parent is destroyed.
    pub fn into_handle(self) -> H {
        let handle = self.handle;
        core::mem::forget(self);
        handle
    }
}

/// A handle owned by a [`Device`].
pub type DeviceOwned<'d, H> = Owned<'d, Device, H>;

impl<P, H: Copy> Drop for Owned<'_, P, H> {
    fn drop(&mut self) {
        // SAFETY: `Owned::new`'s caller vouched that `destroy` destroys the live `handle`,
        // created from `parent`, and that nothing else does; this is the one call.
        unsafe { (self.destroy)(self.parent, self.handle) }
    }
}

/// The destroy functions [`Owned::new`] takes for device children, one per handle type.
pub mod destroy {
    use tessera_vk::{Device, vk};

    /// One `unsafe fn(&Device, H)` per entry, calling the device's destroy command.
    macro_rules! destroy {
        ($($name:ident: $handle:ty => $command:ident,)*) => {$(
            #[doc = concat!("`Device::", stringify!($command), "`.")]
            ///
            /// # Safety
            ///
            /// `handle` is live, was created from `device`, is not in use, and is destroyed
            /// only here.
            pub unsafe fn $name(device: &Device, handle: $handle) {
                // SAFETY: the caller upholds the contract above.
                unsafe { device.$command(handle, None) }
            }
        )*};
    }

    destroy! {
        buffer: vk::Buffer => destroy_buffer,
        memory: vk::DeviceMemory => free_memory,
        image: vk::Image => destroy_image,
        image_view: vk::ImageView => destroy_image_view,
        command_pool: vk::CommandPool => destroy_command_pool,
        fence: vk::Fence => destroy_fence,
        shader_module: vk::ShaderModule => destroy_shader_module,
        pipeline_layout: vk::PipelineLayout => destroy_pipeline_layout,
        pipeline: vk::Pipeline => destroy_pipeline,
        pipeline_binary: vk::PipelineBinaryKHR => destroy_pipeline_binary_khr,
    }
}

/// A device that waits for its work and is destroyed when dropped. Everything created from it
/// must be dropped first.
pub struct OwnedDevice {
    device: Device,
}

impl OwnedDevice {
    /// Takes ownership of `device`.
    ///
    /// # Safety
    ///
    /// `device` is live and nothing else destroys it.
    pub unsafe fn new(device: Device) -> OwnedDevice {
        OwnedDevice { device }
    }
}

impl Deref for OwnedDevice {
    type Target = Device;

    fn deref(&self) -> &Device {
        &self.device
    }
}

impl Drop for OwnedDevice {
    fn drop(&mut self) {
        // SAFETY: the device is live (`OwnedDevice::new`) and its children were dropped first.
        // A failed wait (`ERROR_DEVICE_LOST`) still allows destruction, so its result is
        // ignored.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

/// An instance and its debug messenger, destroyed (messenger first) when dropped. It owns the
/// messenger too: a separate guard would borrow the instance it is stored beside.
pub struct OwnedInstance {
    instance: Instance,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl OwnedInstance {
    /// Takes ownership of `instance`, with no messenger yet.
    ///
    /// # Safety
    ///
    /// `instance` is live and nothing else destroys it. Everything created from it (devices
    /// included: a [`Device`] does not borrow its instance) is destroyed before this owner is
    /// dropped.
    pub unsafe fn new(instance: Instance) -> OwnedInstance {
        OwnedInstance {
            instance,
            messenger: None,
        }
    }

    /// Takes ownership of `messenger`.
    ///
    /// # Safety
    ///
    /// `messenger` is live, was created from this instance, and nothing else destroys it.
    ///
    /// # Panics
    ///
    /// When a messenger is already owned.
    pub unsafe fn set_messenger(&mut self, messenger: vk::DebugUtilsMessengerEXT) {
        assert!(
            self.messenger.is_none(),
            "the instance already owns a messenger"
        );
        self.messenger = Some(messenger);
    }
}

impl Deref for OwnedInstance {
    type Target = Instance;

    fn deref(&self) -> &Instance {
        &self.instance
    }
}

impl Drop for OwnedInstance {
    fn drop(&mut self) {
        // SAFETY: the instance and its messenger are live (`new`, `set_messenger`), and
        // everything else created from the instance was dropped first.
        unsafe {
            if let Some(messenger) = self.messenger {
                self.instance
                    .destroy_debug_utils_messenger_ext(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
