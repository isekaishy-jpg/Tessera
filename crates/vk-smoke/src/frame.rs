//! The offscreen frame: a staged device-local vertex buffer, a colour image rendered with dynamic
//! rendering, a readback buffer, one command buffer and a fence.
//!
//! Every struct field whose zeroed `Default` is invalid or wrong (mip and layer counts, sample
//! count, `load_op`) is set explicitly. Barriers keep source and destination queue family equal:
//! no ownership transfer.

use crate::failure::{Context, Outcome, fail};
use crate::gpu::Gpu;
use crate::owned::{DeviceOwned, Owned, destroy};
use tessera_vk::{Device, vk};

/// The image format; the pipeline renders to it.
pub const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// The image size; the pipeline's viewport and scissor cover it.
pub const EXTENT: vk::Extent2D = vk::Extent2D {
    width: 64,
    height: 64,
};

/// One vertex: a clip-space position, the shader's `float2` at location 0.
type Vertex = [f32; 2];
/// The format the pipeline reads a `Vertex` as.
pub const VERTEX_FORMAT: vk::Format = vk::Format::R32G32_SFLOAT;
/// Bytes per `Vertex`: the pipeline's binding stride.
pub const VERTEX_STRIDE: u32 = size_of::<Vertex>() as u32;
/// The triangle. It covers the centre, not the corners.
const VERTICES: [Vertex; 3] = [[0.0, -0.75], [0.75, 0.75], [-0.75, 0.75]];
/// `VERTICES` in bytes.
const VERTEX_BYTES: vk::DeviceSize = size_of::<[Vertex; 3]>() as u64;
/// Bytes per `R8G8B8A8` pixel.
const PIXEL_BYTES: u32 = 4;
/// The readback buffer: the whole image, tightly packed.
const READBACK_BYTES: vk::DeviceSize = (EXTENT.width * EXTENT.height * PIXEL_BYTES) as u64;
/// The clear colour, and so the expected corner pixel.
const BLACK: [u8; 4] = [0, 0, 0, 255];
/// The fragment shader's colour, and so the expected centre pixel.
const GREEN: [u8; 4] = [0, 255, 0, 255];

/// One mip level and one layer of colour: the image's only subresource.
const COLOR: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// A buffer or image and the memory bound to it at offset 0. Fields drop in order: the resource
/// before its memory.
struct Bound<'d, H: Copy> {
    resource: DeviceOwned<'d, H>,
    memory: DeviceOwned<'d, vk::DeviceMemory>,
}

/// Everything the frame owns. Fields drop in declaration order: fence, pool (which frees the
/// command buffer), view, image, then the buffers.
pub struct Frame<'d> {
    device: &'d Device,
    fence: DeviceOwned<'d, vk::Fence>,
    pool: DeviceOwned<'d, vk::CommandPool>,
    commands: vk::CommandBuffer,
    view: DeviceOwned<'d, vk::ImageView>,
    image: Bound<'d, vk::Image>,
    readback: Bound<'d, vk::Buffer>,
    vertex: Bound<'d, vk::Buffer>,
    staging: Bound<'d, vk::Buffer>,
    queue: vk::Queue,
}

impl<'d> Frame<'d> {
    /// Creates every object of the frame on `device` (created on `gpu`). Each is owned before
    /// the next fallible call, so a failure destroys what was created.
    pub fn create(device: &'d Device, gpu: &Gpu) -> Outcome<Frame<'d>> {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let staging = buffer(
            device,
            gpu,
            VERTEX_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
            host,
        )?;
        let vertex_usage = vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER;
        let vertex = buffer(device, gpu, VERTEX_BYTES, vertex_usage, local)?;
        let readback = buffer(
            device,
            gpu,
            READBACK_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
            host,
        )?;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D {
                width: EXTENT.width,
                height: EXTENT.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::COUNT_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `device` is live; `image_info` describes a valid 2D colour image.
        let image = unsafe { device.create_image(&image_info, None) }.ctx("vkCreateImage")?;
        // SAFETY: `image` was just created on `device`; only this owner destroys it.
        let image = unsafe { Owned::new(device, image, destroy::image) };
        // SAFETY: `image` is live on `device`.
        let requirements = unsafe { device.get_image_memory_requirements(image.handle()) };
        let image_memory = allocate(device, gpu, requirements, local)?;
        // SAFETY: `image_memory` was allocated for `image`'s requirements; neither is bound.
        unsafe { device.bind_image_memory(image.handle(), image_memory.handle(), 0) }
            .ctx("vkBindImageMemory")?;
        let image = Bound {
            resource: image,
            memory: image_memory,
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image.resource.handle())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(FORMAT)
            .subresource_range(COLOR);
        // SAFETY: `image` is bound to memory and has colour attachment usage.
        let view =
            unsafe { device.create_image_view(&view_info, None) }.ctx("vkCreateImageView")?;
        // SAFETY: `view` was just created on `device`; only this owner destroys it.
        let view = unsafe { Owned::new(device, view, destroy::image_view) };

        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(gpu.queue_family);
        // SAFETY: `device` has a queue in `gpu.queue_family`.
        let pool =
            unsafe { device.create_command_pool(&pool_info, None) }.ctx("vkCreateCommandPool")?;
        // SAFETY: `pool` was just created on `device`; only this owner destroys it, which also
        // frees its command buffer.
        let pool = unsafe { Owned::new(device, pool, destroy::command_pool) };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.handle())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let mut commands = [vk::CommandBuffer::null()];
        // SAFETY: `pool` is live; `commands` holds `command_buffer_count` handles.
        unsafe { device.allocate_command_buffers(&alloc_info, &mut commands) }
            .ctx("vkAllocateCommandBuffers")?;
        // SAFETY: `device` is live; the fence starts unsignalled.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .ctx("vkCreateFence")?;
        // SAFETY: `fence` was just created on `device`; only this owner destroys it.
        let fence = unsafe { Owned::new(device, fence, destroy::fence) };
        // SAFETY: the device was created with one queue in `gpu.queue_family`.
        let queue = unsafe { device.get_device_queue(gpu.queue_family, 0) };
        Ok(Frame {
            device,
            fence,
            pool,
            commands: commands[0],
            view,
            image,
            readback,
            vertex,
            staging,
            queue,
        })
    }

    /// Checks the unsignalled fence: a zero-timeout wait returns `TIMEOUT` and its status is
    /// `NOT_READY` (both success codes the wrappers return, not errors). Before
    /// [`Frame::render`], the fence has never been submitted.
    pub fn check_fence(&self) -> Outcome<()> {
        let fence = self.fence.handle();
        // SAFETY: the fence is live on the frame's device.
        let waited =
            unsafe { self.device.wait_for_fences(&[fence], true, 0) }.ctx("vkWaitForFences")?;
        if waited != vk::Result::TIMEOUT {
            return fail(format!(
                "vkWaitForFences on an unsignalled fence returned {waited}"
            ));
        }
        // SAFETY: as above.
        let status = unsafe { self.device.get_fence_status(fence) }.ctx("vkGetFenceStatus")?;
        if status != vk::Result::NOT_READY {
            return fail(format!(
                "vkGetFenceStatus on an unsignalled fence returned {status}"
            ));
        }
        Ok(())
    }

    /// Writes the triangle into the staging buffer through `vkMapMemory2`. The copy to the
    /// device-local vertex buffer is recorded by [`Frame::render`].
    ///
    /// # Safety
    ///
    /// The GPU is not reading the staging buffer.
    pub unsafe fn stage(&self) -> Outcome<()> {
        let memory = self.staging.memory.handle();
        let map = vk::MemoryMapInfo::default()
            .memory(memory)
            .size(VERTEX_BYTES);
        // SAFETY: the memory is the frame's own, host visible and not mapped.
        let ptr = unsafe { self.device.map_memory2(&map) }.ctx("vkMapMemory2")?;
        // SAFETY: `ptr` maps `VERTEX_BYTES` bytes of coherent memory, aligned to at least
        // `minMemoryMapAlignment` (64), so it holds the three vertices; the caller guarantees
        // the GPU does not touch it.
        unsafe {
            ptr.cast::<Vertex>()
                .copy_from_nonoverlapping(VERTICES.as_ptr(), VERTICES.len())
        };
        let unmap = vk::MemoryUnmapInfo::default().memory(memory);
        // SAFETY: the memory is mapped, and `ptr` is not used again.
        unsafe { self.device.unmap_memory2(&unmap) }.ctx("vkUnmapMemory2")
    }

    /// The probe's recording: two copies from the staging buffer to the vertex buffer with no
    /// barrier between them, a write-after-write hazard. The command buffer is never submitted;
    /// the pool is reset afterwards, returning it to the initial state for [`Frame::render`].
    ///
    /// # Safety
    ///
    /// The frame's command buffer is in the initial state.
    pub unsafe fn record_hazard(&self) -> Outcome<()> {
        let cmd = self.commands;
        let copy = [vk::BufferCopy::default().size(VERTEX_BYTES)];
        let (staging, vertex) = (
            self.staging.resource.handle(),
            self.vertex.resource.handle(),
        );
        // SAFETY: the caller guarantees `cmd` is in the initial state; the buffers are the
        // frame's own and outlive the recording; nothing is submitted, so the pool reset finds
        // no pending use.
        unsafe {
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .ctx("vkBeginCommandBuffer")?;
            self.device.cmd_copy_buffer(cmd, staging, vertex, &copy);
            self.device.cmd_copy_buffer(cmd, staging, vertex, &copy);
            self.device
                .end_command_buffer(cmd)
                .ctx("vkEndCommandBuffer")?;
            self.device
                .reset_command_pool(self.pool.handle(), vk::CommandPoolResetFlags::empty())
                .ctx("vkResetCommandPool")
        }
    }

    /// Records and submits the frame, then waits for its fence: copy staging to vertex buffer,
    /// render the triangle with `pipeline` over a black clear, copy the image to the readback
    /// buffer. It returns with no work in flight, also on failure: after a successful submit, a
    /// failed fence wait waits for the queue before returning.
    ///
    /// # Safety
    ///
    /// `pipeline` was created on the frame's device, and the frame's command buffer is in the
    /// initial state.
    pub unsafe fn render(&self, pipeline: vk::Pipeline) -> Outcome<()> {
        let device = self.device;
        let cmd = self.commands;
        let (staging, vertex) = (
            self.staging.resource.handle(),
            self.vertex.resource.handle(),
        );
        let (image, readback) = (
            self.image.resource.handle(),
            self.readback.resource.handle(),
        );
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let copy = [vk::BufferCopy::default().size(VERTEX_BYTES)];
        let to_vertex = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT)
            .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
            .buffer(vertex)
            .size(vk::WHOLE_SIZE)];
        let to_attachment = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(COLOR)];
        let before_render = vk::DependencyInfo::default()
            .buffer_memory_barriers(&to_vertex)
            .image_memory_barriers(&to_attachment);
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.view.handle())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            })];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(EXTENT))
            .layer_count(1)
            .color_attachments(&attachments);
        let to_transfer = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image)
            .subresource_range(COLOR)];
        let before_copy = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        let region = [vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: EXTENT.width,
                height: EXTENT.height,
                depth: 1,
            })];
        let to_host = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .buffer(readback)
            .size(vk::WHOLE_SIZE)];
        let before_read = vk::DependencyInfo::default().buffer_memory_barriers(&to_host);

        // SAFETY: every object is the frame's own (the caller guarantees `pipeline` and that
        // `cmd` is in the initial state); every pointer borrows a local that outlives its call;
        // each barrier orders the accesses on both sides of it (staging write → vertex read,
        // clear and draw → copy → host read).
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .ctx("vkBeginCommandBuffer")?;
            device.cmd_copy_buffer(cmd, staging, vertex, &copy);
            device.cmd_pipeline_barrier2(cmd, &before_render);
            device.cmd_begin_rendering(cmd, &rendering);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_vertex_buffers(cmd, 0, &[vertex], &[0]);
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_rendering(cmd);
            device.cmd_pipeline_barrier2(cmd, &before_copy);
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                &region,
            );
            device.cmd_pipeline_barrier2(cmd, &before_read);
            device.end_command_buffer(cmd).ctx("vkEndCommandBuffer")?;
        }
        let command_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&command_infos)];
        let fence = self.fence.handle();
        // SAFETY: `cmd` is recorded, the queue and fence belong to the frame's device, and the
        // fence is unsignalled. A failed submit leaves nothing pending.
        unsafe { device.queue_submit2(self.queue, &submits, fence) }.ctx("vkQueueSubmit2")?;
        // SAFETY: the fence is live; the wait makes the results visible before this returns.
        let waited = unsafe { device.wait_for_fences(&[fence], true, u64::MAX) };
        if waited == Ok(vk::Result::SUCCESS) {
            return Ok(());
        }
        // SAFETY: the queue is live. The result is ignored: the wait's failure is the one
        // reported, and `ERROR_DEVICE_LOST` still ends the queue's work.
        let _ = unsafe { device.queue_wait_idle(self.queue) };
        match waited {
            Ok(code) => fail(format!("vkWaitForFences after the submit returned {code}")),
            Err(e) => fail(format!("vkWaitForFences after the submit: {e}")),
        }
    }

    /// Maps the readback buffer through `vkMapMemory2` and checks a centre pixel (the green
    /// triangle) and a corner pixel (the black clear).
    ///
    /// # Safety
    ///
    /// [`Frame::render`] has completed.
    pub unsafe fn check_pixels(&self) -> Outcome<()> {
        let memory = self.readback.memory.handle();
        let map = vk::MemoryMapInfo::default()
            .memory(memory)
            .size(READBACK_BYTES);
        // SAFETY: the memory is the frame's own, host visible and not mapped.
        let ptr = unsafe { self.device.map_memory2(&map) }.ctx("vkMapMemory2")?;
        // SAFETY: `ptr` maps `READBACK_BYTES` bytes of coherent memory that the completed
        // submit wrote (made visible to the host by its last barrier and the fence wait).
        let pixels =
            unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), READBACK_BYTES as usize) };
        let centre = pixel(pixels, EXTENT.width / 2, EXTENT.height / 2);
        let corner = pixel(pixels, 0, 0);
        let unmap = vk::MemoryUnmapInfo::default().memory(memory);
        // SAFETY: the memory is mapped; `pixels` is not used again.
        unsafe { self.device.unmap_memory2(&unmap) }.ctx("vkUnmapMemory2")?;
        if centre != GREEN {
            return fail(format!("centre pixel is {centre:?}, expected {GREEN:?}"));
        }
        if corner != BLACK {
            return fail(format!("corner pixel is {corner:?}, expected {BLACK:?}"));
        }
        Ok(())
    }
}

/// The `R8G8B8A8` pixel at `(x, y)` of the tightly packed readback.
fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * EXTENT.width + x) * PIXEL_BYTES) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
}

/// A buffer of `size` bytes with `usage`, bound to memory with `flags`.
fn buffer<'d>(
    device: &'d Device,
    gpu: &Gpu,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    flags: vk::MemoryPropertyFlags,
) -> Outcome<Bound<'d, vk::Buffer>> {
    let info = vk::BufferCreateInfo::default().size(size).usage(usage);
    // SAFETY: `device` is live; `info` describes an exclusive buffer.
    let buffer = unsafe { device.create_buffer(&info, None) }.ctx("vkCreateBuffer")?;
    // SAFETY: `buffer` was just created on `device`; only this owner destroys it.
    let buffer = unsafe { Owned::new(device, buffer, destroy::buffer) };
    // SAFETY: `buffer` is live on `device`.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer.handle()) };
    let memory = allocate(device, gpu, requirements, flags)?;
    // SAFETY: `memory` was allocated for `buffer`'s requirements; neither is bound.
    unsafe { device.bind_buffer_memory(buffer.handle(), memory.handle(), 0) }
        .ctx("vkBindBufferMemory")?;
    Ok(Bound {
        resource: buffer,
        memory,
    })
}

/// Memory for `requirements` in the first allowed type with `flags`.
fn allocate<'d>(
    device: &'d Device,
    gpu: &Gpu,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Outcome<DeviceOwned<'d, vk::DeviceMemory>> {
    let index = gpu.memory_type(requirements.memory_type_bits, flags)?;
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(index);
    // SAFETY: `device` was created on `gpu`, so `index` is one of its memory types.
    let memory = unsafe { device.allocate_memory(&info, None) }.ctx("vkAllocateMemory")?;
    // SAFETY: `memory` was just allocated on `device`; only this owner frees it.
    Ok(unsafe { Owned::new(device, memory, destroy::memory) })
}
