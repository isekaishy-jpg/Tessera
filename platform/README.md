# platform

FFI and safe single-API wrappers over the OS, the GPU driver and native libraries, one crate per API, such as `vk`, `window`, `input`, `fs`, `audio-device`, `box3d-sys` and `luau-sys`.

- **Depends on:** `core/`.
- **`unsafe`:** allowed (FFI); every `unsafe` block has a `// SAFETY:` comment.
- **Not here:** engine services built on these APIs (`servers/`).
- **Crates:** `vk` (`tessera-vk`, the Vulkan bindings).
