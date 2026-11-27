//! `Gpu::new` vouches for a physical device, so it is called only in `unsafe` code.

use tessera_vk::{Instance, InstanceTable, vk};
use tessera_vk_smoke::gpu::Gpu;
use tessera_vk_smoke::owned::OwnedInstance;

fn main() {
    let table = Instance::from_parts(vk::Instance::null(), InstanceTable::default());
    // SAFETY: never runs; the code does not compile.
    let instance = unsafe { OwnedInstance::new(table) };
    let _gpu = Gpu::new(
        &instance,
        vk::PhysicalDevice::null(),
        String::new(),
        0,
        vk::PhysicalDeviceMemoryProperties::default(),
    );
}
