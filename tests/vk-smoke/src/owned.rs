//! Owners: every Vulkan object the program creates is held by one of these from the moment it
//! exists, so every exit path destroys it exactly once, after its work completed. Command buffers
//! are the exception: their pool frees them.
//!
//! Ownership is in the types: an [`OwnedDevice`] borrows its instance (through its [`Gpu`]) and
//! every [`Owned`] child borrows its device, so the borrow checker orders the drops. A struct's
//! owned fields drop in declaration order, and locals in reverse order of declaration.
//!
//! When completion of submitted work cannot be established, the instance is abandoned: every
//! owner of it leaks its handle instead of destroying it, since destroying an object the device
//! may still use is invalid (VUID-vkDestroyBuffer-buffer-00922 and its siblings).

use crate::failure::{Outcome, fail};
use crate::gpu::Gpu;
use core::cell::Cell;
use core::mem;
use core::ops::Deref;
use tessera_vk::{Device, Instance, vk};

/// A handle created on `device`, destroyed by `destroy` when dropped unless the device is
/// abandoned.
pub struct Owned<'d, H: Copy> {
    /// The device the handle was created on.
    device: &'d OwnedDevice<'d>,
    /// The owned handle.
    handle: H,
    /// Destroys the handle; called once, on drop.
    destroy: unsafe fn(&Device, H),
}

impl<'d, H: Copy> Owned<'d, H> {
    /// Takes ownership of `handle`.
    ///
    /// # Safety
    ///
    /// `handle` is live and was created on `device`, `destroy` destroys it, and nothing else
    /// destroys it.
    pub unsafe fn new(
        device: &'d OwnedDevice<'d>,
        handle: H,
        destroy: unsafe fn(&Device, H),
    ) -> Owned<'d, H> {
        Owned {
            device,
            handle,
            destroy,
        }
    }

    /// The handle, still owned.
    pub fn handle(&self) -> H {
        self.handle
    }

    /// The device the handle was created on.
    pub fn device(&self) -> &'d OwnedDevice<'d> {
        self.device
    }
}

impl<H: Copy> Drop for Owned<'_, H> {
    fn drop(&mut self) {
        if self.device.abandoned() {
            return;
        }
        // SAFETY: `Owned::new`'s caller vouched that `destroy` destroys the live `handle`,
        // created on the device, and that nothing else does; this is the one call. The device
        // is not abandoned, so every submit it made completed (`OwnedDevice::submit_and_wait`).
        unsafe { (self.destroy)(&self.device.device, self.handle) }
    }
}

/// The destroy functions [`Owned::new`] takes, one per handle type.
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

/// A device created on its [`Gpu`]. [`OwnedDevice::close`] waits for the device's work, then
/// destroys it, or abandons the instance and fails when the wait cannot establish completion.
/// Its drop closes it too, but a drop cannot report the failure, so owners call `close`.
pub struct OwnedDevice<'i> {
    /// The GPU the device was created on; it borrows the instance.
    gpu: Gpu<'i>,
    /// The owned device.
    device: Device,
    /// Set once the device is closed, so it is closed once.
    closed: bool,
}

impl<'i> OwnedDevice<'i> {
    /// Takes ownership of `device`.
    ///
    /// # Safety
    ///
    /// `device` is live and nothing else destroys it. It was created on `gpu`'s physical device
    /// through `gpu.instance()` (as their tables answer for it), with one queue in
    /// `gpu.queue_family()`, `VK_KHR_pipeline_binary` and the features [`Gpu::create_device`]
    /// enables. Every function pointer in its table stays valid while it lives. Every command
    /// this crate's safe items call through it is valid on it or not loaded: a `tessera-vk`
    /// wrapper panics on an unloaded command, which makes no Vulkan call.
    pub unsafe fn new(gpu: Gpu<'i>, device: Device) -> OwnedDevice<'i> {
        OwnedDevice {
            gpu,
            device,
            closed: false,
        }
    }

    /// Waits for the device's work, then destroys the device. When the wait fails, other than
    /// with `ERROR_DEVICE_LOST`, completion is not established: the instance is abandoned, so
    /// neither the device nor the instance is destroyed, and the failure says so. When the
    /// instance is already abandoned, nothing is done: the call that abandoned it failed.
    pub fn close(mut self) -> Outcome<()> {
        self.teardown()
    }

    /// The GPU the device was created on.
    pub fn gpu(&self) -> &Gpu<'i> {
        &self.gpu
    }

    /// True once the instance is abandoned: no owner destroys anything.
    pub fn abandoned(&self) -> bool {
        self.gpu.instance().abandoned()
    }

    /// Submits `submits` to `queue` with `fence` and waits for the fence.
    ///
    /// It returns with no work of the submit pending, or with the instance abandoned:
    /// - a submit that fails with `ERROR_OUT_OF_HOST_MEMORY` or `ERROR_OUT_OF_DEVICE_MEMORY`
    ///   left nothing pending (the `vkQueueSubmit2` reference page);
    /// - a fence wait that returns `SUCCESS` or fails with `ERROR_DEVICE_LOST` ended the work
    ///   (a lost device completes every command);
    /// - any other submit or wait failure is settled with `vkQueueWaitIdle`; when that wait
    ///   also fails, other than with `ERROR_DEVICE_LOST`, the instance is abandoned.
    ///
    /// # Safety
    ///
    /// The caller upholds `vkQueueSubmit2`'s valid usage: `queue` and `fence` belong to this
    /// device, the fence is unsignalled, and every command buffer in `submits` was recorded on
    /// this device.
    pub unsafe fn submit_and_wait(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Outcome<()> {
        // SAFETY: the caller upholds the command's valid usage.
        let submitted = unsafe { self.device.queue_submit2(queue, submits, fence) };
        let unsettled = match submitted {
            Err(
                code @ (vk::Result::ERROR_OUT_OF_HOST_MEMORY
                | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
            ) => return fail(format!("vkQueueSubmit2: {code}")),
            Err(code) => format!("vkQueueSubmit2: {code}"),
            Ok(()) => {
                // SAFETY: the fence belongs to this device (the caller's contract).
                match unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) } {
                    Ok(vk::Result::SUCCESS) => return Ok(()),
                    Err(code @ vk::Result::ERROR_DEVICE_LOST) => {
                        return fail(format!("vkWaitForFences after the submit: {code}"));
                    }
                    Ok(code) => format!("vkWaitForFences after the submit returned {code}"),
                    Err(code) => format!("vkWaitForFences after the submit: {code}"),
                }
            }
        };
        // SAFETY: the queue belongs to this device (the caller's contract).
        match unsafe { self.device.queue_wait_idle(queue) } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => fail(unsettled),
            Err(code) => {
                self.gpu.instance().abandon();
                fail(format!(
                    "{unsettled}; vkQueueWaitIdle: {code}; completion not established: nothing \
                     is destroyed"
                ))
            }
        }
    }

    /// [`OwnedDevice::close`]'s work; it runs once, from `close` or from the drop.
    fn teardown(&mut self) -> Outcome<()> {
        if mem::replace(&mut self.closed, true) || self.abandoned() {
            return Ok(());
        }
        // SAFETY: the device is live (`OwnedDevice::new`) and not yet destroyed (`closed`); its
        // children borrow it, so they were dropped first.
        match unsafe { self.device.device_wait_idle() } {
            Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {
                // SAFETY: as above; the wait ended every command on the device (a lost device
                // completes them), and only this owner destroys it, once.
                unsafe { self.device.destroy_device(None) };
                Ok(())
            }
            Err(code) => {
                self.gpu.instance().abandon();
                fail(format!(
                    "vkDeviceWaitIdle: {code}; completion not established: the device and the \
                     instance are not destroyed"
                ))
            }
        }
    }
}

impl Deref for OwnedDevice<'_> {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for OwnedDevice<'_> {
    /// Closes the device unless [`OwnedDevice::close`] did; a failure leaves the instance
    /// abandoned, which its owner reads from [`OwnedInstance::abandoned`].
    fn drop(&mut self) {
        let _ = self.teardown();
    }
}

/// An instance and its debug messenger, destroyed (messenger first) when dropped unless
/// abandoned. It owns the messenger too: a separate owner would borrow the instance it is stored
/// beside. It holds the abandonment flag every owner of the instance reads.
pub struct OwnedInstance {
    /// The owned instance.
    instance: Instance,
    /// The owned debug messenger, once [`OwnedInstance::set_messenger`] gives it.
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Set when completion of submitted work cannot be established.
    abandoned: Cell<bool>,
}

impl OwnedInstance {
    /// Takes ownership of `instance`, with no messenger yet.
    ///
    /// # Safety
    ///
    /// `instance` is live and nothing else destroys it. Every function pointer in its table
    /// stays valid while it lives. Every command this crate's safe items call through it
    /// ([`crate::gpu::pick`], [`Gpu::create_device`], its drop) is valid on it or not loaded: a
    /// `tessera-vk` wrapper panics on an unloaded command, which makes no Vulkan call.
    pub unsafe fn new(instance: Instance) -> OwnedInstance {
        OwnedInstance {
            instance,
            messenger: None,
            abandoned: Cell::new(false),
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

    /// True once the instance is abandoned: no owner destroys anything.
    pub fn abandoned(&self) -> bool {
        self.abandoned.get()
    }

    /// Abandons the instance: every owner of it leaks its handle from now on.
    pub(crate) fn abandon(&self) {
        self.abandoned.set(true);
    }
}

impl Deref for OwnedInstance {
    type Target = Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for OwnedInstance {
    fn drop(&mut self) {
        if self.abandoned() {
            return;
        }
        // SAFETY: the instance and its messenger are live (`new`, `set_messenger`), and the
        // devices created from it borrow it, so they were destroyed first.
        unsafe {
            if let Some(messenger) = self.messenger {
                self.instance
                    .destroy_debug_utils_messenger_ext(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
