//! A byte parameter takes only `Plain` types: a reference's bytes are an address, not data.

use tessera_vk::{Device, DeviceTable, vk};

static BYTE: u8 = 0;

fn main() {
    let device = Device::from_parts(vk::Device::null(), DeviceTable::default());
    // SAFETY: never runs; the call does not compile.
    unsafe {
        device.cmd_update_buffer(vk::CommandBuffer::null(), vk::Buffer::null(), 0, &[&BYTE]);
    }
}
