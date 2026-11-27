//! Ownership under failure: `Frame::create`, `Recipe::create`, `capture` and `rebuild` run on a
//! mock device table that fails one fallible call per run (call 1, then call 2, ...), and every
//! handle created must be destroyed exactly once. Batch creates that fail return a partial
//! batch: a live handle in entry 0, null in the rest.
//!
//! The mocks keep their state in thread-locals (tests run in parallel threads) and record faults
//! instead of panicking: a panic cannot unwind out of an `extern "system"` function.

use core::ffi::c_void;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use tessera_vk::{Device, DeviceTable, Instance, InstanceTable, vk};
use tessera_vk_smoke::frame::Frame;
use tessera_vk_smoke::gpu::Gpu;
use tessera_vk_smoke::owned::{Owned, OwnedDevice, OwnedInstance, destroy};
use tessera_vk_smoke::pipeline::{self, Recipe};
use tessera_vk_smoke::shader::Shaders;

/// More fallible calls than any function under test makes.
const MAX_CALLS: u32 = 100;
/// The error an injected failure returns.
const FAILED: vk::Result = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;

thread_local! {
    /// The fallible call that fails (1-based); 0 for none.
    static FAIL_AT: Cell<u32> = const { Cell::new(0) };
    /// Fallible calls made since the last [`arm`].
    static CALLS: Cell<u32> = const { Cell::new(0) };
    /// The next handle value.
    static NEXT: Cell<u64> = const { Cell::new(0x100) };
    /// Live handles and their kinds.
    static LIVE: RefCell<BTreeMap<u64, &'static str>> = const { RefCell::new(BTreeMap::new()) };
    /// Destroys of null, unknown or other-kind handles.
    static FAULTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// Device and instance teardown calls, in order.
    static TEARDOWN: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// Restarts the call count and fails call `n` (1-based; 0 fails none).
fn arm(n: u32) {
    CALLS.with(|c| c.set(0));
    FAIL_AT.with(|f| f.set(n));
}

fn calls() -> u32 {
    CALLS.with(Cell::get)
}

/// Counts a fallible call; true when it is the one to fail.
fn fails() -> bool {
    let n = CALLS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    n == FAIL_AT.with(Cell::get)
}

/// A new live handle of `kind`.
fn create(kind: &'static str) -> u64 {
    let handle = NEXT.with(|n| {
        n.set(n.get() + 1);
        n.get()
    });
    LIVE.with(|l| l.borrow_mut().insert(handle, kind));
    handle
}

/// Records the destroy of `handle`, which must be a live handle of `kind`.
fn destroyed(kind: &'static str, handle: u64) {
    let fault = match LIVE.with(|l| l.borrow_mut().remove(&handle)) {
        _ if handle == 0 => Some(format!("{kind}: null handle destroyed")),
        Some(k) if k == kind => None,
        Some(k) => Some(format!("{kind}: destroyed a {k} ({handle:#x})")),
        None => Some(format!(
            "{kind}: {handle:#x} is unknown or already destroyed"
        )),
    };
    if let Some(fault) = fault {
        FAULTS.with(|f| f.borrow_mut().push(fault));
    }
}

fn live() -> BTreeMap<u64, &'static str> {
    LIVE.with(|l| l.borrow().clone())
}

fn faults() -> Vec<String> {
    FAULTS.with(|f| f.borrow().clone())
}

fn teardown() -> Vec<&'static str> {
    TEARDOWN.with(|t| t.borrow().clone())
}

/// A mock `vkCreate*` for one handle type.
macro_rules! create_mock {
    ($name:ident, $info:ty, $handle:ident, $kind:literal) => {
        unsafe extern "system" fn $name(
            _: vk::Device,
            _: *const $info,
            _: *const vk::AllocationCallbacks<'_>,
            out: *mut vk::$handle,
        ) -> vk::Result {
            if fails() {
                return FAILED;
            }
            // SAFETY: the wrapper passes a live output.
            unsafe { *out = vk::$handle::from_raw(create($kind)) };
            vk::Result::SUCCESS
        }
    };
}

/// A mock `vkDestroy*` (or `vkFreeMemory`) for one handle type.
macro_rules! destroy_mock {
    ($name:ident, $handle:ident, $kind:literal) => {
        unsafe extern "system" fn $name(
            _: vk::Device,
            handle: vk::$handle,
            _: *const vk::AllocationCallbacks<'_>,
        ) {
            destroyed($kind, handle.as_raw());
        }
    };
}

create_mock!(create_buffer, vk::BufferCreateInfo<'_>, Buffer, "buffer");
create_mock!(
    allocate_memory,
    vk::MemoryAllocateInfo<'_>,
    DeviceMemory,
    "memory"
);
create_mock!(create_image, vk::ImageCreateInfo<'_>, Image, "image");
create_mock!(
    create_image_view,
    vk::ImageViewCreateInfo<'_>,
    ImageView,
    "image view"
);
create_mock!(
    create_command_pool,
    vk::CommandPoolCreateInfo<'_>,
    CommandPool,
    "command pool"
);
create_mock!(create_fence, vk::FenceCreateInfo<'_>, Fence, "fence");
create_mock!(
    create_shader_module,
    vk::ShaderModuleCreateInfo<'_>,
    ShaderModule,
    "shader module"
);
create_mock!(
    create_pipeline_layout,
    vk::PipelineLayoutCreateInfo<'_>,
    PipelineLayout,
    "pipeline layout"
);
destroy_mock!(destroy_buffer, Buffer, "buffer");
destroy_mock!(free_memory, DeviceMemory, "memory");
destroy_mock!(destroy_image, Image, "image");
destroy_mock!(destroy_image_view, ImageView, "image view");
destroy_mock!(destroy_command_pool, CommandPool, "command pool");
destroy_mock!(destroy_fence, Fence, "fence");
destroy_mock!(destroy_shader_module, ShaderModule, "shader module");
destroy_mock!(destroy_pipeline_layout, PipelineLayout, "pipeline layout");
destroy_mock!(destroy_pipeline, Pipeline, "pipeline");
destroy_mock!(
    destroy_pipeline_binary,
    PipelineBinaryKHR,
    "pipeline binary"
);

/// Writes the requirements every resource gets: 256 bytes in memory type 0.
///
/// # Safety
///
/// `out` points to a live `MemoryRequirements`.
unsafe fn requirements(out: *mut vk::MemoryRequirements) {
    // SAFETY: the caller passes a live output.
    unsafe {
        (*out).size = 256;
        (*out).memory_type_bits = 1;
    }
}

unsafe extern "system" fn buffer_requirements(
    _: vk::Device,
    _: vk::Buffer,
    out: *mut vk::MemoryRequirements,
) {
    // SAFETY: the wrapper passes a live output.
    unsafe { requirements(out) };
}

unsafe extern "system" fn image_requirements(
    _: vk::Device,
    _: vk::Image,
    out: *mut vk::MemoryRequirements,
) {
    // SAFETY: the wrapper passes a live output.
    unsafe { requirements(out) };
}

unsafe extern "system" fn bind_buffer(
    _: vk::Device,
    _: vk::Buffer,
    _: vk::DeviceMemory,
    _: vk::DeviceSize,
) -> vk::Result {
    if fails() { FAILED } else { vk::Result::SUCCESS }
}

unsafe extern "system" fn bind_image(
    _: vk::Device,
    _: vk::Image,
    _: vk::DeviceMemory,
    _: vk::DeviceSize,
) -> vk::Result {
    if fails() { FAILED } else { vk::Result::SUCCESS }
}

/// Command buffers are freed with their pool, so they are not tracked.
unsafe extern "system" fn allocate_command_buffers(
    _: vk::Device,
    _: *const vk::CommandBufferAllocateInfo<'_>,
    out: *mut vk::CommandBuffer,
) -> vk::Result {
    if fails() {
        return FAILED;
    }
    // SAFETY: the wrapper passes `commandBufferCount` (1) handles.
    unsafe { *out = vk::CommandBuffer::from_raw(0x9000) };
    vk::Result::SUCCESS
}

unsafe extern "system" fn get_device_queue(_: vk::Device, _: u32, _: u32, out: *mut vk::Queue) {
    // SAFETY: the wrapper passes a live output.
    unsafe { *out = vk::Queue::from_raw(0xa000) };
}

/// One pipeline per create info; a failure still returns a live handle in entry 0.
unsafe extern "system" fn create_graphics_pipelines(
    _: vk::Device,
    _: vk::PipelineCache,
    count: u32,
    _: *const vk::GraphicsPipelineCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Pipeline,
) -> vk::Result {
    let failing = fails();
    for i in 0..count as usize {
        let pipeline = if failing && i > 0 {
            vk::Pipeline::null()
        } else {
            vk::Pipeline::from_raw(create("pipeline"))
        };
        // SAFETY: the wrapper passes `count` handles.
        unsafe { *out.add(i) = pipeline };
    }
    if failing { FAILED } else { vk::Result::SUCCESS }
}

/// Two binaries: the count with a null array, else a batch (partial when failing).
unsafe extern "system" fn create_pipeline_binaries(
    _: vk::Device,
    _: *const vk::PipelineBinaryCreateInfoKHR<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::PipelineBinaryHandlesInfoKHR<'_>,
) -> vk::Result {
    // SAFETY: the wrapper passes a live output whose array, when non-null, holds
    // `pipelineBinaryCount` handles.
    unsafe {
        let handles = &mut *out;
        if handles.p_pipeline_binaries.is_null() {
            if fails() {
                return FAILED;
            }
            handles.pipeline_binary_count = 2;
            return vk::Result::SUCCESS;
        }
        let failing = fails();
        for i in 0..handles.pipeline_binary_count as usize {
            let binary = if failing && i > 0 {
                vk::PipelineBinaryKHR::null()
            } else {
                vk::PipelineBinaryKHR::from_raw(create("pipeline binary"))
            };
            *handles.p_pipeline_binaries.add(i) = binary;
        }
        if failing { FAILED } else { vk::Result::SUCCESS }
    }
}

/// Four bytes per binary (the wrapper queries the size, then fills).
unsafe extern "system" fn pipeline_binary_data(
    _: vk::Device,
    _: *const vk::PipelineBinaryDataInfoKHR<'_>,
    _: *mut vk::PipelineBinaryKeyKHR<'_>,
    size: *mut usize,
    data: *mut c_void,
) -> vk::Result {
    if fails() {
        return FAILED;
    }
    // SAFETY: the wrapper passes a live size and, when non-null, `*size` bytes of data.
    unsafe {
        if data.is_null() {
            *size = 4;
        } else {
            data.cast::<[u8; 4]>().write_unaligned([1, 2, 3, 4]);
        }
    }
    vk::Result::SUCCESS
}

unsafe extern "system" fn release_captured_data(
    _: vk::Device,
    _: *const vk::ReleaseCapturedPipelineDataInfoKHR<'_>,
    _: *const vk::AllocationCallbacks<'_>,
) -> vk::Result {
    if fails() { FAILED } else { vk::Result::SUCCESS }
}

unsafe extern "system" fn device_wait_idle(_: vk::Device) -> vk::Result {
    TEARDOWN.with(|t| t.borrow_mut().push("wait idle"));
    vk::Result::SUCCESS
}

unsafe extern "system" fn destroy_device(_: vk::Device, _: *const vk::AllocationCallbacks<'_>) {
    TEARDOWN.with(|t| t.borrow_mut().push("destroy device"));
}

unsafe extern "system" fn destroy_messenger(
    _: vk::Instance,
    _: vk::DebugUtilsMessengerEXT,
    _: *const vk::AllocationCallbacks<'_>,
) {
    TEARDOWN.with(|t| t.borrow_mut().push("destroy messenger"));
}

unsafe extern "system" fn destroy_instance(_: vk::Instance, _: *const vk::AllocationCallbacks<'_>) {
    TEARDOWN.with(|t| t.borrow_mut().push("destroy instance"));
}

/// A device whose table holds every mock above.
fn device() -> Device {
    Device::from_parts(
        vk::Device::from_raw(0x1000),
        DeviceTable {
            create_buffer: Some(create_buffer),
            destroy_buffer: Some(destroy_buffer),
            get_buffer_memory_requirements: Some(buffer_requirements),
            bind_buffer_memory: Some(bind_buffer),
            allocate_memory: Some(allocate_memory),
            free_memory: Some(free_memory),
            create_image: Some(create_image),
            destroy_image: Some(destroy_image),
            get_image_memory_requirements: Some(image_requirements),
            bind_image_memory: Some(bind_image),
            create_image_view: Some(create_image_view),
            destroy_image_view: Some(destroy_image_view),
            create_command_pool: Some(create_command_pool),
            destroy_command_pool: Some(destroy_command_pool),
            allocate_command_buffers: Some(allocate_command_buffers),
            create_fence: Some(create_fence),
            destroy_fence: Some(destroy_fence),
            get_device_queue: Some(get_device_queue),
            create_shader_module: Some(create_shader_module),
            destroy_shader_module: Some(destroy_shader_module),
            create_pipeline_layout: Some(create_pipeline_layout),
            destroy_pipeline_layout: Some(destroy_pipeline_layout),
            create_graphics_pipelines: Some(create_graphics_pipelines),
            destroy_pipeline: Some(destroy_pipeline),
            create_pipeline_binaries_khr: Some(create_pipeline_binaries),
            destroy_pipeline_binary_khr: Some(destroy_pipeline_binary),
            get_pipeline_binary_data_khr: Some(pipeline_binary_data),
            release_captured_pipeline_data_khr: Some(release_captured_data),
            device_wait_idle: Some(device_wait_idle),
            destroy_device: Some(destroy_device),
            ..Default::default()
        },
    )
}

/// A GPU with one memory type that has every property the frame asks for.
fn gpu() -> Gpu {
    let mut memory = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::DEVICE_LOCAL;
    Gpu::new(vk::PhysicalDevice::from_raw(1), "mock".into(), 0, memory)
}

/// Two one-word modules (the mock does not read them).
fn shaders() -> Shaders {
    Shaders {
        vertex: vec![0x0723_0203],
        fragment: vec![0x0723_0203],
    }
}

/// Runs `attempt` once per fallible call it makes, failing call 1, then call 2, ...; after
/// each run, the handles live are exactly `before` and no fault was recorded. Every run with a
/// failure must fail; the first run that makes fewer calls than the one armed (no failure)
/// must succeed. Returns the number of failures injected.
fn every_failure_cleans_up(
    before: &BTreeMap<u64, &'static str>,
    mut attempt: impl FnMut() -> bool,
) -> u32 {
    for n in 1..=MAX_CALLS {
        arm(n);
        let succeeded = attempt();
        let injected = calls() >= n;
        arm(0);
        assert_eq!(&live(), before, "call {n} failing: a handle leaked");
        assert_eq!(faults(), Vec::<String>::new(), "call {n} failing");
        if !injected {
            assert!(succeeded, "no call failed, yet the run failed");
            return n - 1;
        }
        assert!(!succeeded, "call {n} failed, yet the run succeeded");
    }
    panic!("more than {MAX_CALLS} fallible calls");
}

#[test]
fn frame_create_cleans_up_on_every_failure() {
    let (device, gpu) = (device(), gpu());
    let failures =
        every_failure_cleans_up(&BTreeMap::new(), || Frame::create(&device, &gpu).is_ok());
    assert_ne!(failures, 0);
}

#[test]
fn recipe_create_cleans_up_on_every_failure() {
    let (device, shaders) = (device(), shaders());
    let failures = every_failure_cleans_up(&BTreeMap::new(), || {
        Recipe::create(&device, &shaders).is_ok()
    });
    assert_ne!(failures, 0);
}

#[test]
fn capture_cleans_up_on_every_failure() {
    let (device, shaders) = (device(), shaders());
    let recipe = Recipe::create(&device, &shaders).expect("the recipe is created");
    let before = live();
    let failures = every_failure_cleans_up(&before, || pipeline::capture(&device, &recipe).is_ok());
    assert_ne!(failures, 0);
}

#[test]
fn rebuild_cleans_up_on_every_failure() {
    let (device, shaders) = (device(), shaders());
    let recipe = Recipe::create(&device, &shaders).expect("the recipe is created");
    let mut captured = pipeline::capture(&device, &recipe).expect("the binaries are captured");
    let before = live();
    let failures = every_failure_cleans_up(&before, || {
        pipeline::rebuild(&device, &recipe, &mut captured).is_ok()
    });
    assert_ne!(failures, 0);
}

#[test]
fn into_handle_releases_ownership() {
    let device = device();
    let buffer = vk::Buffer::from_raw(create("buffer"));
    // SAFETY: the mock buffer is live and owned only here.
    let owned = unsafe { Owned::new(&device, buffer, destroy::buffer) };
    assert_eq!(owned.into_handle(), buffer);
    assert_eq!(live().len(), 1, "released, not destroyed");
    // SAFETY: the released buffer is live and destroyed only here.
    unsafe { destroy::buffer(&device, buffer) };
    assert!(live().is_empty());
    assert_eq!(faults(), Vec::<String>::new());
}

#[test]
fn owned_device_waits_then_destroys() {
    // SAFETY: the mock device is owned only here.
    drop(unsafe { OwnedDevice::new(device()) });
    assert_eq!(teardown(), ["wait idle", "destroy device"]);
}

fn instance() -> Instance {
    Instance::from_parts(
        vk::Instance::from_raw(0x2000),
        InstanceTable {
            destroy_debug_utils_messenger_ext: Some(destroy_messenger),
            destroy_instance: Some(destroy_instance),
            ..Default::default()
        },
    )
}

#[test]
fn owned_instance_destroys_messenger_then_instance() {
    // SAFETY: the mock instance and messenger are owned only here.
    unsafe {
        let mut owned = OwnedInstance::new(instance());
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(5));
    }
    assert_eq!(teardown(), ["destroy messenger", "destroy instance"]);
    TEARDOWN.with(|t| t.borrow_mut().clear());
    // SAFETY: as above, with no messenger.
    drop(unsafe { OwnedInstance::new(instance()) });
    assert_eq!(
        teardown(),
        ["destroy instance"],
        "no messenger, none destroyed"
    );
}

#[test]
#[should_panic(expected = "the instance already owns a messenger")]
fn a_second_messenger_panics() {
    // SAFETY: the mock instance and messengers are owned only here.
    unsafe {
        let mut owned = OwnedInstance::new(instance());
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(5));
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(6));
    }
}
