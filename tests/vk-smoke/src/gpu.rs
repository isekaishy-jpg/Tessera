//! The GPU: the first one with Vulkan 1.4, `VK_KHR_pipeline_binary` (feature `pipelineBinaries`)
//! and a graphics queue; the device created on it; and the memory type for each use.

use crate::failure::{Context, Outcome, fail};
use crate::owned::{OwnedDevice, OwnedInstance};
use crate::text;
use tessera_vk::{fixed_str, vk};

/// A physical device of an instance, with what the smoke test needs of it. It is built by
/// [`pick`], which checks every requirement, or by the vouching [`Gpu::new`].
pub struct Gpu<'i> {
    /// The instance the physical device belongs to.
    instance: &'i OwnedInstance,
    /// Its handle.
    physical: vk::PhysicalDevice,
    /// `deviceName`.
    name: String,
    /// A queue family with graphics.
    queue_family: u32,
    /// Its memory types and heaps ([`Gpu::memory_type`] searches them).
    memory: vk::PhysicalDeviceMemoryProperties,
}

impl<'i> Gpu<'i> {
    /// A GPU described by the caller.
    ///
    /// # Safety
    ///
    /// `physical` is a physical device of `instance` (as its table answers for it) and supports
    /// everything [`Gpu::create_device`] enables. `queue_family` is one of its queue families
    /// with graphics, and `memory` is its memory properties.
    pub unsafe fn new(
        instance: &'i OwnedInstance,
        physical: vk::PhysicalDevice,
        name: String,
        queue_family: u32,
        memory: vk::PhysicalDeviceMemoryProperties,
    ) -> Gpu<'i> {
        Gpu {
            instance,
            physical,
            name,
            queue_family,
            memory,
        }
    }

    /// The instance the GPU belongs to.
    pub fn instance(&self) -> &'i OwnedInstance {
        self.instance
    }

    /// `deviceName`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The queue family the device's one queue comes from; it has graphics.
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The first memory type allowed by `bits` (a `memoryTypeBits` mask) that has every flag in
    /// `flags`.
    pub fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Outcome<u32> {
        let count = self.memory.memory_type_count as usize;
        let found = (0u32..)
            .zip(self.memory.memory_types.iter().take(count))
            .find(|(i, t)| bits & (1 << i) != 0 && t.property_flags.contains(flags));
        let Some((index, _)) = found else {
            return fail(format!(
                "{}: no memory type with {flags:?} in {bits:#b}",
                self.name
            ));
        };
        Ok(index)
    }

    /// The device: one queue of [`Gpu::queue_family`], `VK_KHR_pipeline_binary`, and the
    /// features the smoke test uses (`dynamicRendering`, `synchronization2`, `maintenance5`,
    /// `pipelineBinaries`). The device keeps the GPU.
    pub fn create_device(self) -> Outcome<OwnedDevice<'i>> {
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.queue_family)
            .queue_priorities(&priorities)];
        let extensions = [vk::KHR_PIPELINE_BINARY_EXTENSION_NAME.as_ptr()];
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut v14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
        let mut binary =
            vk::PhysicalDevicePipelineBinaryFeaturesKHR::default().pipeline_binaries(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut v13)
            .push_next(&mut v14)
            .push_next(&mut binary);
        // SAFETY: `physical` belongs to the instance and supports everything enabled (checked
        // by `pick`, or vouched for by `Gpu::new`'s caller; the Vulkan 1.3 and 1.4 features are
        // required by 1.4); every pointer in `info` borrows a local that outlives the call.
        let device = unsafe { self.instance.create_device(self.physical, &info, None) }
            .ctx("vkCreateDevice")?;
        // SAFETY: `device` was just created on this GPU through its instance, with exactly the
        // queue, extension and features `OwnedDevice::new` requires; it was loaded through the
        // instance's table, whose function pointers stay valid (`OwnedInstance::new`); only the
        // returned owner destroys it.
        Ok(unsafe { OwnedDevice::new(self, device) })
    }
}

/// The first GPU of `instance` that qualifies, or a failure listing why each was rejected.
pub fn pick(instance: &OwnedInstance) -> Outcome<Gpu<'_>> {
    // SAFETY: `instance` is live (`OwnedInstance::new`).
    let physicals =
        unsafe { instance.enumerate_physical_devices() }.ctx("vkEnumeratePhysicalDevices")?;
    let mut rejected = Vec::new();
    for physical in physicals {
        match qualify(instance, physical) {
            Ok(gpu) => return Ok(gpu),
            Err(why) => rejected.push(why.to_string()),
        }
    }
    fail(format!(
        "no GPU has Vulkan 1.4, VK_KHR_pipeline_binary with pipelineBinaries and a graphics \
         queue: [{}]",
        rejected.join("; ")
    ))
}

/// `physical` as a [`Gpu`], or why it does not qualify.
fn qualify(instance: &OwnedInstance, physical: vk::PhysicalDevice) -> Outcome<Gpu<'_>> {
    // SAFETY: `physical` was enumerated from the live `instance`.
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = fixed_str(&properties.device_name);
    if properties.api_version < vk::API_VERSION_1_4 {
        return fail(format!("{name}: Vulkan below 1.4"));
    }
    // SAFETY: as above.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical, None) }
        .ctx("vkEnumerateDeviceExtensionProperties")?;
    if !text::has_extension(&extensions, vk::KHR_PIPELINE_BINARY_EXTENSION_NAME) {
        return fail(format!("{name}: no VK_KHR_pipeline_binary"));
    }
    let mut binary = vk::PhysicalDevicePipelineBinaryFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut binary);
    // SAFETY: as above; the chain holds one struct the device supports (the extension is
    // listed).
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    if binary.pipeline_binaries != vk::TRUE {
        return fail(format!("{name}: pipelineBinaries not supported"));
    }
    // SAFETY: as above.
    let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let Some(queue_family) = (0u32..)
        .zip(&families)
        .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(i, _)| i)
    else {
        return fail(format!("{name}: no graphics queue"));
    };
    // SAFETY: as above.
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    // SAFETY: `physical` was enumerated from `instance`; it has Vulkan 1.4 (so the Vulkan 1.3
    // and 1.4 features `create_device` enables), `VK_KHR_pipeline_binary` with
    // `pipelineBinaries`, and `queue_family` has graphics (all checked above); `memory` is its
    // own.
    Ok(unsafe { Gpu::new(instance, physical, name, queue_family, memory) })
}
