//! `VkPhysicalDeviceFeatures2` does not extend `VkBufferCreateInfo`.

use tessera_vk::vk;

fn main() {
    let mut features = vk::PhysicalDeviceFeatures2::default();
    let _ = vk::BufferCreateInfo::default().push_next(&mut features);
}
