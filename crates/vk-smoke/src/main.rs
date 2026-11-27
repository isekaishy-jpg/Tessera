//! `tessera-vk-smoke`: drives `tessera-vk` against the real driver under the Khronos validation
//! layer, proves synchronization validation is live, and fails on any validation warning or
//! error.
//!
//! It compiles its Slang shaders with the SDK's `slangc`, creates a Vulkan 1.4 instance and
//! device, records a known hazard and requires the layer to report it, checks `TIMEOUT` and
//! `NOT_READY` on an unsignalled fence, stages a device-local vertex buffer through
//! `vkMapMemory2`, builds a pipeline with `VK_KHR_pipeline_binary` capture and rebuilds it from
//! the binaries' bytes, renders a triangle offscreen with the rebuilt pipeline and checks two
//! read-back pixels. Each check prints `ok <step>`; the last line is the message count. It exits
//! 1 on any failure or message.

use std::process::ExitCode;
use tessera_vk::Entry;
use tessera_vk_smoke::failure::{Context, Outcome};
use tessera_vk_smoke::frame::Frame;
use tessera_vk_smoke::{gpu, pipeline, shader, validation};

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

/// Every step, then the report that everything was destroyed (on return from [`steps`]).
fn run() -> Outcome<()> {
    steps()?;
    ok("destroyed");
    Ok(())
}

/// Every step, in order. Each object is owned by a guard declared after its parent, so every
/// return, early or not, destroys them in reverse order of creation.
fn steps() -> Outcome<()> {
    let shaders = shader::compile()?;
    ok("shader compiled");
    // SAFETY: loading the system's Vulkan loader runs its initialisation code, which this
    // program accepts.
    let entry = unsafe { Entry::load() }.ctx("loading the Vulkan library")?;
    let instance = validation::create_instance(&entry)?;
    ok("instance with validation");
    let gpu = gpu::pick(&instance)?;
    ok(&format!("gpu {}", gpu.name));
    let device = gpu::create_device(&instance, &gpu)?;
    ok("device");

    let frame = Frame::create(&device, &gpu)?;
    // SAFETY: the frame's command buffer is new, in the initial state.
    validation::probe(|| unsafe { frame.record_hazard() })?;
    ok("sync validation live (probe hazard reported)");
    frame.check_fence()?;
    ok("fence TIMEOUT and NOT_READY");
    // SAFETY: nothing has been submitted.
    unsafe { frame.stage() }?;
    ok("staged vertex buffer (map_memory2)");

    let recipe = pipeline::Recipe::create(&device, &shaders)?;
    let mut captured = pipeline::capture(&device, &recipe)?;
    let bytes: usize = captured.iter().map(pipeline::Captured::size).sum();
    ok(&format!(
        "pipeline binaries captured: {}, {bytes} bytes",
        captured.len()
    ));
    let rebuilt = pipeline::rebuild(&device, &recipe, &mut captured)?;
    ok("pipeline rebuilt from binaries");

    // SAFETY: `rebuilt` was created on the frame's device; the probe reset the command buffer
    // to the initial state.
    unsafe { frame.render(rebuilt.handle()) }?;
    ok("rendered and read back");
    // SAFETY: `render` completed (it waited for its submit).
    unsafe { frame.check_pixels() }?;
    ok("pixels checked");
    Ok(())
}
