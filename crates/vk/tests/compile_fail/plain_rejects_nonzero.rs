//! A byte setter takes only `Plain` types: zero bytes are not a `NonZeroU32`.

use core::num::NonZeroU32;
use tessera_vk::vk;

fn main() {
    let values = [NonZeroU32::MIN];
    let _ = vk::SpecializationInfo::default().data(&values);
}
