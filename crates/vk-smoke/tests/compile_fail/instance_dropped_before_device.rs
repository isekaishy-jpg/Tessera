//! A device borrows its instance through its `Gpu`: the instance cannot be dropped first.

use tessera_vk::{Instance, InstanceTable, vk};
use tessera_vk_smoke::gpu::{self, Gpu};
use tessera_vk_smoke::owned::OwnedInstance;

fn main() {
    let table = Instance::from_parts(vk::Instance::null(), InstanceTable::default());
    // SAFETY: never runs; the code does not compile.
    let instance = unsafe { OwnedInstance::new(table) };
    let device = gpu::pick(&instance).and_then(Gpu::create_device);
    drop(instance);
    drop(device);
}
