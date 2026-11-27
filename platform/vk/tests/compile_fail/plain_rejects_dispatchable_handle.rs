//! A byte parameter takes only `Plain` types: a dispatchable handle is a pointer, not data.

use tessera_vk::{Device, DeviceTable, vk};

fn main() {
    let device = Device::from_parts(vk::Device::null(), DeviceTable::default());
    // SAFETY: never runs; the call does not compile.
    unsafe {
        device.cmd_push_constants(
            vk::CommandBuffer::null(),
            vk::PipelineLayout::null(),
            vk::ShaderStageFlags::VERTEX,
            0,
            &[vk::Device::null()],
        );
    }
}
