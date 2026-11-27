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
use tessera_vk_smoke::failure::{Outcome, fail};
use tessera_vk_smoke::frame::Frame;
use tessera_vk_smoke::owned::{OwnedDevice, OwnedInstance};
use tessera_vk_smoke::pipeline::Recipe;
use tessera_vk_smoke::shader::{self, Shaders};
use tessera_vk_smoke::{gpu, validation};

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

/// Every step, in order. Each object is held by an owner declared after its parent, so every
/// return, early or not, destroys them in reverse order of creation, unless completion of the
/// device's work could not be established: then nothing more is destroyed, and the run fails.
fn steps() -> Outcome<()> {
    let shaders = shader::compile()?;
    ok("shader compiled");
    // SAFETY: loading the system's Vulkan library runs its initialisation code, which this
    // program accepts.
    let instance = unsafe { validation::create_instance() }?;
    ok("instance with validation");
    on_device(&instance, &shaders)
}

/// The steps on the device, then its close; a failed close is reported with the steps' result.
fn on_device(instance: &OwnedInstance, shaders: &Shaders) -> Outcome<()> {
    let chosen = gpu::pick(instance)?;
    ok(&format!("gpu {}", chosen.name()));
    let device = chosen.create_device()?;
    ok("device");
    let result = frame_steps(&device, shaders);
    match (result, device.close()) {
        (Ok(()), closed) => closed,
        (Err(failure), Ok(())) => Err(failure),
        (Err(failure), Err(close)) => fail(format!("{failure}; {close}")),
    }
}

/// The steps on the device's objects; they are destroyed on return, before the device.
fn frame_steps(device: &OwnedDevice<'_>, shaders: &Shaders) -> Outcome<()> {
    let frame = Frame::create(device)?;
    // SAFETY: the frame's command buffer is new, in the initial state.
    validation::probe(|| unsafe { frame.record_hazard() })?;
    ok("sync validation live (probe hazard reported)");
    frame.check_fence()?;
    ok("fence TIMEOUT and NOT_READY");
    // SAFETY: nothing has been submitted.
    unsafe { frame.stage() }?;
    ok("staged vertex buffer (map_memory2)");

    let recipe = Recipe::create(device, shaders)?;
    let mut captured = recipe.capture()?;
    ok(&format!(
        "pipeline binaries captured: {}, {} bytes",
        captured.count(),
        captured.size()
    ));
    let rebuilt = captured.rebuild()?;
    ok("pipeline rebuilt from binaries");

    // SAFETY: `rebuild` builds the pipeline with the frame's format and vertex layout; the
    // probe reset the command buffer to the initial state; `render` has not run on this frame.
    unsafe { frame.render(&rebuilt) }?;
    ok("rendered and read back");
    // SAFETY: `render` completed (it waited for its submit).
    unsafe { frame.check_pixels() }?;
    ok("pixels checked");
    Ok(())
}
