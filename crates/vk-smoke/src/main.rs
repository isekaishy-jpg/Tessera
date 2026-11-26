//! `tessera-vk-smoke`: drives `tessera-vk` against the real driver under the Khronos validation
//! layer with synchronization validation on, and fails on any validation warning or error.
//!
//! It compiles its Slang shaders with the SDK's `slangc`, creates a Vulkan 1.4 instance and
//! device, checks `TIMEOUT` and `NOT_READY` on an unsignalled fence, stages a device-local vertex
//! buffer through `vkMapMemory2`, builds a pipeline with `VK_KHR_pipeline_binary` capture and
//! rebuilds it from the binaries' bytes, renders a triangle offscreen with the rebuilt pipeline
//! and checks two read-back pixels. Each check prints `ok <step>`; the last line is the message
//! count. It exits 1 on any failure or message.

mod failure;
mod frame;
mod gpu;
mod pipeline;
mod shader;
mod text;
mod validation;

use failure::{Context, Outcome};
use std::process::ExitCode;
use tessera_vk::Entry;

fn main() -> ExitCode {
    let result = run();
    let messages = validation::messages();
    if let Err(failure) = &result {
        eprintln!("FAILED: {failure}");
    }
    println!("validation messages: {messages}");
    if result.is_ok() && messages == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Reports a passed step.
fn ok(step: &str) {
    println!("ok {step}");
}

/// Every step, in order; objects are destroyed in reverse order of creation.
fn run() -> Outcome<()> {
    let shaders = shader::compile()?;
    ok("shader compiled");
    // SAFETY: loading the system's Vulkan loader runs its initialisation code, which this
    // program accepts.
    let entry = unsafe { Entry::load() }.ctx("loading the Vulkan library")?;
    let (instance, messenger) = validation::create_instance(&entry)?;
    ok("instance with validation (sync on)");
    let gpu = gpu::pick(&instance)?;
    ok(&format!("gpu {}", gpu.name));
    let device = gpu::create_device(&instance, &gpu)?;
    ok("device");

    let frame = frame::Frame::create(&device, &gpu)?;
    // SAFETY: `frame` was created on `device`.
    unsafe { frame.check_fence(&device) }?;
    ok("fence TIMEOUT and NOT_READY");
    // SAFETY: as above; nothing has been submitted.
    unsafe { frame.stage(&device) }?;
    ok("staged vertex buffer (map_memory2)");

    let recipe = pipeline::Recipe::create(&device, &shaders)?;
    let mut captured = pipeline::capture(&device, &recipe)?;
    let bytes: usize = captured.iter().map(pipeline::Captured::len).sum();
    ok(&format!(
        "pipeline binaries captured: {}, {bytes} bytes",
        captured.len()
    ));
    let rebuilt = pipeline::rebuild(&device, &recipe, &mut captured)?;
    ok("pipeline rebuilt from binaries");

    // SAFETY: `frame` and `rebuilt` were created on `device`; the command buffer is unused.
    unsafe { frame.render(&device, rebuilt) }?;
    ok("rendered and read back");
    // SAFETY: as above; `render` waited for the submit.
    unsafe { frame.check_pixels(&device) }?;
    ok("pixels checked");

    // SAFETY: the GPU is idle (the only submit was waited for); each object is destroyed once,
    // children before their parents, and nothing is used afterwards.
    unsafe {
        device.destroy_pipeline(rebuilt, None);
        recipe.destroy(&device);
        frame.destroy(&device);
        device.destroy_device(None);
        instance.destroy_debug_utils_messenger_ext(messenger, None);
        instance.destroy_instance(None);
    }
    ok("destroyed");
    Ok(())
}
