//! The GPU: the first one with Vulkan 1.4, `VK_KHR_pipeline_binary` (feature `pipelineBinaries`)
//! and a graphics queue; the device created on it; and the memory type for each use.

use crate::failure::{Context, Failure, Outcome, fail};
use crate::text;
use tessera_vk::{Device, Instance, vk};

/// The chosen GPU.
pub struct Gpu {
    /// Its handle.
    pub physical: vk::PhysicalDevice,
    /// `deviceName`.
    pub name: String,
    /// The first queue family with graphics.
    pub queue_family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
}

impl Gpu {
    /// The first memory type allowed by `bits` (a `memoryTypeBits` mask) that has every flag in
    /// `flags`.
    pub fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Outcome<u32> {
        let count = self.memory.memory_type_count as usize;
        (0u32..)
            .zip(self.memory.memory_types.iter().take(count))
            .find(|(i, t)| bits & (1 << i) != 0 && t.property_flags.contains(flags))
            .map(|(i, _)| i)
            .ok_or_else(|| {
                Failure(format!(
                    "{}: no memory type with {flags:?} in {bits:#b}",
                    self.name
                ))
            })
    }
}

/// The first GPU that qualifies, or a failure listing why each was rejected.
pub fn pick(instance: &Instance) -> Outcome<Gpu> {
    // SAFETY: `instance` is live.
    let physicals =
        unsafe { instance.enumerate_physical_devices() }.ctx("vkEnumeratePhysicalDevices")?;
    let mut rejected = Vec::new();
    for physical in physicals {
        match qualify(instance, physical) {
            Ok(gpu) => return Ok(gpu),
            Err(why) => rejected.push(why.0),
        }
    }
    fail(format!(
        "no GPU has Vulkan 1.4, VK_KHR_pipeline_binary with pipelineBinaries and a graphics \
         queue: [{}]",
        rejected.join("; ")
    ))
}

/// `physical` as a [`Gpu`], or why it does not qualify.
fn qualify(instance: &Instance, physical: vk::PhysicalDevice) -> Outcome<Gpu> {
    // SAFETY: `physical` was enumerated from the live `instance`.
    let properties = unsafe { instance.get_physical_device_properties(physical) };
    let name = text::fixed(&properties.device_name);
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
    Ok(Gpu {
        physical,
        name,
        queue_family,
        memory,
    })
}

/// The device: one queue of `gpu.queue_family`, `VK_KHR_pipeline_binary`, and the features the
/// smoke test uses (`dynamicRendering`, `synchronization2`, `maintenance5`, `pipelineBinaries`).
pub fn create_device(instance: &Instance, gpu: &Gpu) -> Outcome<Device> {
    let priorities = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(gpu.queue_family)
        .queue_priorities(&priorities)];
    let extensions = [vk::KHR_PIPELINE_BINARY_EXTENSION_NAME.as_ptr()];
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);
    let mut v14 = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
    let mut binary = vk::PhysicalDevicePipelineBinaryFeaturesKHR::default().pipeline_binaries(true);
    let info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .push_next(&mut v13)
        .push_next(&mut v14)
        .push_next(&mut binary);
    // SAFETY: `gpu.physical` belongs to the live `instance` and supports everything enabled
    // (checked in `qualify`; the Vulkan 1.3 and 1.4 features are required by 1.4); every
    // pointer in `info` borrows a local that outlives the call.
    unsafe { instance.create_device(gpu.physical, &info, None) }.ctx("vkCreateDevice")
}
