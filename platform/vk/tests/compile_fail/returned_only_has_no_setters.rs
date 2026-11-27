//! `VkPhysicalDeviceProperties` is `returnedonly`: the implementation fills it, so it has no
//! setters.

use tessera_vk::vk;

fn main() {
    let _ = vk::PhysicalDeviceProperties::default().api_version(1);
}
