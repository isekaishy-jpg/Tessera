//! `Shaders` vouch for SPIR-V, so they are not built from their fields.

use tessera_vk_smoke::shader::Shaders;

fn main() {
    let _shaders = Shaders {
        vertex: Vec::new(),
        fragment: Vec::new(),
    };
}
