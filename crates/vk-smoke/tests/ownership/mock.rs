//! The mock instance and device: their tables, and the owners built on them.
//!
//! Creates count as fallible calls ([`fails`]); the recording commands note the handles they
//! name; `vkQueueSubmit2` marks what the submitted command buffers reach as pending; the waits
//! return their scripted result and end the pending uses when it settles them.

use super::{
    BOUND, PENDING, POOLS, RECORDED, VIEWS, create, destroyed, fails, parent_destroyed, reset,
    scripted, waited,
};
use core::ffi::c_void;
use core::slice;
use std::collections::BTreeSet;
use tessera_vk::vk::{self, Handle};
use tessera_vk::{Device, DeviceTable, Instance, InstanceTable};
use tessera_vk_smoke::gpu::Gpu;
use tessera_vk_smoke::owned::{OwnedDevice, OwnedInstance};
use tessera_vk_smoke::shader::{SPIRV_MAGIC, Shaders};

/// The error an injected failure returns.
const FAILED: vk::Result = vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;

/// The `count` elements at `ptr`, or none when `count` is 0 (`ptr` may then be null).
///
/// # Safety
///
/// When `count` is not 0, `ptr` points to `count` live elements that outlive `'a`.
unsafe fn items<'a, T>(ptr: *const T, count: u32) -> &'a [T] {
    if count == 0 {
        &[]
    } else {
        // SAFETY: the caller guarantees `count` live elements at `ptr`.
        unsafe { slice::from_raw_parts(ptr, count as usize) }
    }
}

/// Notes that the commands recorded in `cmd` name `handles`.
fn record(cmd: vk::CommandBuffer, handles: impl IntoIterator<Item = u64>) {
    RECORDED.with(|r| {
        r.borrow_mut()
            .entry(cmd.object_handle())
            .or_default()
            .extend(handles);
    });
}

/// Marks pending everything a submit of `cmd` reaches: the handles its commands name, the
/// memory bound to them, the image behind each view and that image's memory, and its pool.
fn submitted(cmd: vk::CommandBuffer) {
    let cmd = cmd.object_handle();
    let named = RECORDED.with(|r| r.borrow().get(&cmd).cloned().unwrap_or_default());
    let mut reached = BTreeSet::new();
    for handle in named {
        reached.insert(handle);
        if let Some(image) = VIEWS.with(|v| v.borrow().get(&handle).copied()) {
            reached.insert(image);
        }
    }
    let memory: Vec<u64> = reached
        .iter()
        .filter_map(|h| BOUND.with(|b| b.borrow().get(h).copied()))
        .collect();
    reached.extend(memory);
    reached.extend(POOLS.with(|p| p.borrow().get(&cmd).copied()));
    PENDING.with(|p| p.borrow_mut().extend(reached));
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
            destroyed($kind, handle.object_handle());
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

/// `vkCreateImageView`, a fallible call; the view is noted with its image.
unsafe extern "system" fn create_image_view(
    _: vk::Device,
    info: *const vk::ImageViewCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::ImageView,
) -> vk::Result {
    if fails() {
        return FAILED;
    }
    let view = create("image view");
    // SAFETY: the wrapper passes a live create info and a live output.
    let image = unsafe {
        *out = vk::ImageView::from_raw(view);
        (*info).image.object_handle()
    };
    VIEWS.with(|v| v.borrow_mut().insert(view, image));
    vk::Result::SUCCESS
}

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

/// `vkGetBufferMemoryRequirements` writing [`requirements`].
unsafe extern "system" fn buffer_requirements(
    _: vk::Device,
    _: vk::Buffer,
    out: *mut vk::MemoryRequirements,
) {
    // SAFETY: the wrapper passes a live output.
    unsafe { requirements(out) };
}

/// `vkGetImageMemoryRequirements` writing [`requirements`].
unsafe extern "system" fn image_requirements(
    _: vk::Device,
    _: vk::Image,
    out: *mut vk::MemoryRequirements,
) {
    // SAFETY: the wrapper passes a live output.
    unsafe { requirements(out) };
}

/// Notes `resource` bound to `memory` unless the bind is the call to fail.
fn bind(resource: u64, memory: u64) -> vk::Result {
    if fails() {
        return FAILED;
    }
    BOUND.with(|b| b.borrow_mut().insert(resource, memory));
    vk::Result::SUCCESS
}

/// `vkBindBufferMemory`, a fallible call.
unsafe extern "system" fn bind_buffer(
    _: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    _: vk::DeviceSize,
) -> vk::Result {
    bind(buffer.object_handle(), memory.object_handle())
}

/// `vkBindImageMemory`, a fallible call.
unsafe extern "system" fn bind_image(
    _: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    _: vk::DeviceSize,
) -> vk::Result {
    bind(image.object_handle(), memory.object_handle())
}

/// `vkAllocateCommandBuffers`, a fallible call. Command buffers are freed with their pool, so
/// they are not live handles; the one buffer is noted with its pool.
unsafe extern "system" fn allocate_command_buffers(
    _: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    out: *mut vk::CommandBuffer,
) -> vk::Result {
    if fails() {
        return FAILED;
    }
    let cmd = vk::CommandBuffer::from_raw(0x9000);
    // SAFETY: the wrapper passes a live allocate info and `commandBufferCount` (1) outputs.
    let pool = unsafe {
        *out = cmd;
        (*info).command_pool.object_handle()
    };
    POOLS.with(|p| p.borrow_mut().insert(cmd.object_handle(), pool));
    vk::Result::SUCCESS
}

/// `vkGetDeviceQueue` writing queue `0xa000`.
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

/// `vkReleaseCapturedPipelineDataKHR`, a fallible call.
unsafe extern "system" fn release_captured_data(
    _: vk::Device,
    _: *const vk::ReleaseCapturedPipelineDataInfoKHR<'_>,
    _: *const vk::AllocationCallbacks<'_>,
) -> vk::Result {
    if fails() { FAILED } else { vk::Result::SUCCESS }
}

/// `vkBeginCommandBuffer`: the buffer's recorded commands start over.
unsafe extern "system" fn begin_command_buffer(
    cmd: vk::CommandBuffer,
    _: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    RECORDED.with(|r| r.borrow_mut().insert(cmd.object_handle(), BTreeSet::new()));
    vk::Result::SUCCESS
}

/// `vkEndCommandBuffer`.
unsafe extern "system" fn end_command_buffer(_: vk::CommandBuffer) -> vk::Result {
    vk::Result::SUCCESS
}

/// `vkCmdCopyBuffer`: names both buffers.
unsafe extern "system" fn cmd_copy_buffer(
    cmd: vk::CommandBuffer,
    src: vk::Buffer,
    dst: vk::Buffer,
    _: u32,
    _: *const vk::BufferCopy,
) {
    record(cmd, [src.object_handle(), dst.object_handle()]);
}

/// `vkCmdPipelineBarrier2`: names every buffer and image of its barriers.
unsafe extern "system" fn cmd_pipeline_barrier2(
    cmd: vk::CommandBuffer,
    info: *const vk::DependencyInfo<'_>,
) {
    // SAFETY: the wrapper passes a live dependency info whose arrays hold their counts.
    let (buffers, images) = unsafe {
        let info = &*info;
        (
            items(
                info.p_buffer_memory_barriers,
                info.buffer_memory_barrier_count,
            ),
            items(
                info.p_image_memory_barriers,
                info.image_memory_barrier_count,
            ),
        )
    };
    record(cmd, buffers.iter().map(|b| b.buffer.object_handle()));
    record(cmd, images.iter().map(|i| i.image.object_handle()));
}

/// `vkCmdBeginRendering`: names every colour attachment's view.
unsafe extern "system" fn cmd_begin_rendering(
    cmd: vk::CommandBuffer,
    info: *const vk::RenderingInfo<'_>,
) {
    // SAFETY: the wrapper passes a live rendering info whose array holds its count.
    let attachments = unsafe {
        let info = &*info;
        items(info.p_color_attachments, info.color_attachment_count)
    };
    record(
        cmd,
        attachments.iter().map(|a| a.image_view.object_handle()),
    );
}

/// `vkCmdBindPipeline`: names the pipeline.
unsafe extern "system" fn cmd_bind_pipeline(
    cmd: vk::CommandBuffer,
    _: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    record(cmd, [pipeline.object_handle()]);
}

/// `vkCmdBindVertexBuffers`: names every buffer.
unsafe extern "system" fn cmd_bind_vertex_buffers(
    cmd: vk::CommandBuffer,
    _: u32,
    count: u32,
    buffers: *const vk::Buffer,
    _: *const vk::DeviceSize,
) {
    // SAFETY: the wrapper passes `count` buffers.
    let buffers = unsafe { items(buffers, count) };
    record(cmd, buffers.iter().map(|b| b.object_handle()));
}

/// `vkCmdDraw`: names nothing.
unsafe extern "system" fn cmd_draw(_: vk::CommandBuffer, _: u32, _: u32, _: u32, _: u32) {}

/// `vkCmdEndRendering`: names nothing.
unsafe extern "system" fn cmd_end_rendering(_: vk::CommandBuffer) {}

/// `vkCmdCopyImageToBuffer`: names the image and the buffer.
unsafe extern "system" fn cmd_copy_image_to_buffer(
    cmd: vk::CommandBuffer,
    image: vk::Image,
    _: vk::ImageLayout,
    buffer: vk::Buffer,
    _: u32,
    _: *const vk::BufferImageCopy,
) {
    record(cmd, [image.object_handle(), buffer.object_handle()]);
}

/// `vkQueueSubmit2` returning the scripted result. Unless it is out of host or device memory
/// (which leaves nothing pending), everything the submitted command buffers reach and the
/// fence become pending.
unsafe extern "system" fn queue_submit2(
    _: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let code = scripted().submit;
    if code == vk::Result::ERROR_OUT_OF_HOST_MEMORY
        || code == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
    {
        return code;
    }
    // SAFETY: the wrapper passes `count` live submits, each with its command buffer infos.
    let submits = unsafe { items(submits, count) };
    for submit in submits {
        // SAFETY: as above.
        let infos = unsafe {
            items(
                submit.p_command_buffer_infos,
                submit.command_buffer_info_count,
            )
        };
        for info in infos {
            submitted(info.command_buffer);
        }
    }
    PENDING.with(|p| p.borrow_mut().insert(fence.object_handle()));
    code
}

/// `vkWaitForFences` returning the scripted result, logged as "wait fence".
unsafe extern "system" fn wait_for_fences(
    _: vk::Device,
    _: u32,
    _: *const vk::Fence,
    _: vk::Bool32,
    _: u64,
) -> vk::Result {
    waited("wait fence", scripted().fence)
}

/// `vkQueueWaitIdle` returning the scripted result, logged as "wait queue".
unsafe extern "system" fn queue_wait_idle(_: vk::Queue) -> vk::Result {
    waited("wait queue", scripted().queue)
}

/// `vkDeviceWaitIdle` returning the scripted result, logged as "wait idle".
unsafe extern "system" fn device_wait_idle(_: vk::Device) -> vk::Result {
    waited("wait idle", scripted().idle)
}

/// `vkDestroyDevice`, logged as "destroy device".
unsafe extern "system" fn destroy_device(_: vk::Device, _: *const vk::AllocationCallbacks<'_>) {
    parent_destroyed("destroy device");
}

/// `vkDestroyDebugUtilsMessengerEXT`, logged as "destroy messenger".
unsafe extern "system" fn destroy_messenger(
    _: vk::Instance,
    _: vk::DebugUtilsMessengerEXT,
    _: *const vk::AllocationCallbacks<'_>,
) {
    parent_destroyed("destroy messenger");
}

/// `vkDestroyInstance`, logged as "destroy instance".
unsafe extern "system" fn destroy_instance(_: vk::Instance, _: *const vk::AllocationCallbacks<'_>) {
    parent_destroyed("destroy instance");
}

/// An instance whose table holds the messenger and instance teardown mocks, not yet owned.
pub fn unowned_instance() -> Instance {
    Instance::from_parts(
        vk::Instance::from_raw(0x2000),
        InstanceTable {
            destroy_debug_utils_messenger_ext: Some(destroy_messenger),
            destroy_instance: Some(destroy_instance),
            ..Default::default()
        },
    )
}

/// The mock instance, owned, with a messenger.
pub fn instance() -> OwnedInstance {
    // SAFETY: the mock instance is owned only here; its table holds static functions, each a
    // valid mock, and every other command is unloaded.
    let mut instance = unsafe { OwnedInstance::new(unowned_instance()) };
    // SAFETY: the mock messenger is owned only by `instance`.
    unsafe { instance.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(5)) };
    instance
}

/// A GPU of `instance` with one memory type that has every property the frame asks for.
fn gpu(instance: &OwnedInstance) -> Gpu<'_> {
    let mut memory = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 1,
        ..Default::default()
    };
    memory.memory_types[0].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::DEVICE_LOCAL;
    // SAFETY: the mock instance loads no physical-device command, so no call can contradict
    // what this vouches for (a physical device of the instance with a graphics queue family 0
    // and these memory types); `create_device` on it would panic (`vkCreateDevice` is not
    // loaded) before any Vulkan call.
    unsafe {
        Gpu::new(
            instance,
            vk::PhysicalDevice::from_raw(1),
            "mock".to_string(),
            0,
            memory,
        )
    }
}

/// A device whose table holds every device mock above.
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
            begin_command_buffer: Some(begin_command_buffer),
            end_command_buffer: Some(end_command_buffer),
            cmd_copy_buffer: Some(cmd_copy_buffer),
            cmd_pipeline_barrier2: Some(cmd_pipeline_barrier2),
            cmd_begin_rendering: Some(cmd_begin_rendering),
            cmd_bind_pipeline: Some(cmd_bind_pipeline),
            cmd_bind_vertex_buffers: Some(cmd_bind_vertex_buffers),
            cmd_draw: Some(cmd_draw),
            cmd_end_rendering: Some(cmd_end_rendering),
            cmd_copy_image_to_buffer: Some(cmd_copy_image_to_buffer),
            queue_submit2: Some(queue_submit2),
            wait_for_fences: Some(wait_for_fences),
            queue_wait_idle: Some(queue_wait_idle),
            device_wait_idle: Some(device_wait_idle),
            destroy_device: Some(destroy_device),
            ..Default::default()
        },
    )
}

/// A new mock device on [`gpu`] of `instance`, owned.
pub fn owned_device(instance: &OwnedInstance) -> OwnedDevice<'_> {
    // SAFETY: the mock device is new and owned only here; its table holds static functions,
    // each a valid mock, and every other command is unloaded; it vouches for being created on
    // the mock GPU.
    unsafe { OwnedDevice::new(gpu(instance), device()) }
}

/// Runs `f` on a mock device after [`reset`], then drops the device and its instance.
pub fn with_device<T>(f: impl FnOnce(&OwnedDevice<'_>) -> T) -> T {
    reset();
    let instance = instance();
    let device = owned_device(&instance);
    f(&device)
}

/// Two one-word modules (the mock does not read them).
pub fn shaders() -> Shaders {
    // SAFETY: the mock device never reads the modules.
    unsafe { Shaders::from_spirv(vec![SPIRV_MAGIC], vec![SPIRV_MAGIC]) }
}
