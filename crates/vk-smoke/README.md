# tessera-vk-smoke

Drives `tessera-vk` against the real driver under `VK_LAYER_KHRONOS_validation` with synchronization validation on. Any validation warning or error, failed check or missing piece exits 1.

## Run

| Command | What |
|---|---|
| `cargo run -p tessera-vk-smoke` | The program; prints one `ok <step>` line per check, then `validation messages: N` |
| `dist/tessera-vk-smoke.exe` | The human testing build (`wg dist`) |
| `cargo test -p tessera-vk-smoke` | `tests/smoke.rs`: runs the program, requires success, every step in order and zero messages |

## Needs

- The Vulkan SDK (`VULKAN_SDK`): the validation layer with `VK_EXT_layer_settings`, and `Bin/slangc`.
- A Vulkan 1.4 GPU with `VK_KHR_pipeline_binary` (feature `pipelineBinaries`), a graphics queue, host-visible coherent memory and device-local memory.

A missing piece fails with its name; nothing is skipped.

## Steps

1. Compiles `src/smoke.slang` with `slangc` into one SPIR-V module per stage (entries `vertexMain` and `fragmentMain`, each emitted as `main`) in a temp directory. No SPIR-V is committed, and building the workspace does not need the SDK.
2. Creates a Vulkan 1.4 instance with the layer, `validate_sync` set through `VK_EXT_layer_settings`, and a debug messenger counting every WARNING and ERROR (one is also chained into instance creation, so creation and destruction are covered).
3. Picks the first qualifying GPU and creates the device (`dynamicRendering`, `synchronization2`, `maintenance5`, `pipelineBinaries`).
4. Checks that a zero-timeout `wait_for_fences` returns `TIMEOUT` and `get_fence_status` returns `NOT_READY` on an unsignalled fence.
5. Writes the triangle into a staging buffer through `map_memory2`.
6. Pipeline binaries: creates the graphics pipeline with `CAPTURE_DATA_KHR`, fetches each binary's key and bytes, releases the captured data and destroys the binaries and the pipeline, then rebuilds the binaries from the keys and bytes alone and creates the pipeline the frame renders with from them.
7. Records and submits: staging → device-local vertex buffer copy, dynamic rendering over a black clear, image → readback buffer copy, with synchronization2 barriers between them; waits on the fence.
8. Maps the readback through `map_memory2`: the centre pixel must be green, the corner black.
9. Destroys every object, children before parents.

## Layout

| Path | What |
|---|---|
| `src/main.rs` | Step order, exit code |
| `src/failure.rs` | `Failure`, `Outcome`, `Context::ctx` (names the failing call) |
| `src/validation.rs` | Instance, layer settings, messenger and message count |
| `src/gpu.rs` | GPU choice, device creation, memory types |
| `src/shader.rs`, `src/smoke.slang` | Shader source and its compilation |
| `src/pipeline.rs` | Pipelines and the binary round trip |
| `src/frame.rs` | Buffers, image, commands, readback and pixel check |
| `src/text.rs` | Extension lookup by name in fixed-size C name arrays |

## Known

- One module per stage is required on the NVIDIA driver (581.57): with one module holding both entry points, which is valid SPIR-V, it refuses to create a pipeline from the captured binaries (`PIPELINE_COMPILE_REQUIRED`).
