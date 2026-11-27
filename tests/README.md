# tests

Cross-crate system tests, benches and on-hardware checks.

- **Depends on:** anything. Nothing depends on `tests/`.
- **`unsafe`:** allowed where a test drives the unsafe APIs of `core/` or `platform/`; every `unsafe` block has a `// SAFETY:` comment.
- **Not here:** a crate's own tests (they stay in that crate's `tests/`).
- **Crates:** `vk-smoke` (`tessera-vk-smoke`, the driver smoke test of `tessera-vk` under the Vulkan validation layer).
