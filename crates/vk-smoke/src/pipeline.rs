//! The graphics pipeline and the `VK_KHR_pipeline_binary` round trip: a pipeline created with
//! binary capture gives its binaries' keys and bytes, and a second pipeline is created from
//! binaries rebuilt from those bytes alone.
//!
//! Batch outputs (pipelines, binaries) are owned entry by entry before the result is checked:
//! a failed batch may still hold created handles.

use crate::failure::{Context, Outcome, fail};
use crate::frame::{EXTENT, FORMAT, VERTEX_FORMAT, VERTEX_STRIDE};
use crate::owned::{DeviceOwned, Owned, destroy};
use crate::shader::Shaders;
use tessera_vk::{Device, vk};

/// What both pipelines are built from: one shader module per stage and an empty pipeline
/// layout. Fields drop in declaration order.
pub struct Recipe<'d> {
    layout: DeviceOwned<'d, vk::PipelineLayout>,
    fragment: DeviceOwned<'d, vk::ShaderModule>,
    vertex: DeviceOwned<'d, vk::ShaderModule>,
}

/// One captured pipeline binary: its key and its bytes.
pub struct Captured {
    key: vk::PipelineBinaryKeyKHR<'static>,
    data: Vec<u8>,
}

impl Captured {
    /// The binary's size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<'d> Recipe<'d> {
    /// The modules of `shaders` and an empty pipeline layout.
    pub fn create(device: &'d Device, shaders: &Shaders) -> Outcome<Recipe<'d>> {
        let vertex = module(device, &shaders.vertex)?;
        let fragment = module(device, &shaders.fragment)?;
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: `device` is live; the layout is empty.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .ctx("vkCreatePipelineLayout")?;
        // SAFETY: `layout` was just created on `device`; only this owner destroys it.
        let layout = unsafe { Owned::new(device, layout, destroy::pipeline_layout) };
        Ok(Recipe {
            layout,
            fragment,
            vertex,
        })
    }
}

/// The shader module of `spirv`.
fn module<'d>(device: &'d Device, spirv: &[u32]) -> Outcome<DeviceOwned<'d, vk::ShaderModule>> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: `device` is live; `spirv` is a whole SPIR-V module (checked by `shader`).
    let module = unsafe { device.create_shader_module(&info, None) }.ctx("vkCreateShaderModule")?;
    // SAFETY: `module` was just created on `device`; only this owner destroys it.
    Ok(unsafe { Owned::new(device, module, destroy::shader_module) })
}

/// Owns every non-null entry of `binaries`, created on `device` by one batch call.
fn own_binaries<'d>(
    device: &'d Device,
    binaries: &[vk::PipelineBinaryKHR],
) -> Vec<DeviceOwned<'d, vk::PipelineBinaryKHR>> {
    binaries
        .iter()
        .filter(|b| !b.is_null())
        .map(|&binary| {
            // SAFETY: the batch call just created `binary` on `device`; only this owner
            // destroys it.
            unsafe { Owned::new(device, binary, destroy::pipeline_binary) }
        })
        .collect()
}

/// Creates the triangle pipeline from `recipe`: capturing binary data when `capture`, or from
/// `binaries` when given. Every other field is the same for both pipelines, as
/// `VkPipelineBinaryInfoKHR` requires.
///
/// # Safety
///
/// `device` created `recipe` and every binary.
unsafe fn create<'d>(
    device: &'d Device,
    recipe: &Recipe<'d>,
    binaries: Option<&[vk::PipelineBinaryKHR]>,
    capture: bool,
) -> Outcome<DeviceOwned<'d, vk::Pipeline>> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(recipe.vertex.handle())
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(recipe.fragment.handle())
            .name(c"main"),
    ];
    // One position per vertex: binding 0, location 0 (`[[vk::location(0)]]`).
    let bindings = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(VERTEX_STRIDE)
        .input_rate(vk::VertexInputRate::VERTEX)];
    let attributes = [vk::VertexInputAttributeDescription::default()
        .location(0)
        .binding(0)
        .format(VERTEX_FORMAT)
        .offset(0)];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [vk::Viewport::default()
        .width(EXTENT.width as f32)
        .height(EXTENT.height as f32)
        .max_depth(1.0)];
    let scissors = [vk::Rect2D::default().extent(EXTENT)];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    // Zeroed `Default`s are invalid here (line width 0, sample count 0) or draw nothing (an
    // empty colour write mask), so each is set.
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::COUNT_1);
    let blend_attachments = [
        vk::PipelineColorBlendAttachmentState::default().color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        ),
    ];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let formats = [FORMAT];
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
    let mut flags = vk::PipelineCreateFlags2CreateInfo::default()
        .flags(vk::PipelineCreateFlags2::CAPTURE_DATA_KHR);
    let mut binary_info =
        vk::PipelineBinaryInfoKHR::default().pipeline_binaries(binaries.unwrap_or_default());
    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(recipe.layout.handle())
        .push_next(&mut rendering);
    if capture {
        info = info.push_next(&mut flags);
    }
    if binaries.is_some() {
        info = info.push_next(&mut binary_info);
    }
    let mut pipelines = [vk::Pipeline::null()];
    // SAFETY: the caller guarantees `device` created `recipe` and the binaries; every pointer
    // in `info` borrows a local that outlives the call; the cache is null, as capture and
    // binaries require.
    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None, &mut pipelines)
    };
    let pipeline = (!pipelines[0].is_null()).then(|| {
        // SAFETY: the call just created `pipelines[0]` on `device`; only this owner destroys
        // it.
        unsafe { Owned::new(device, pipelines[0], destroy::pipeline) }
    });
    let code = result.ctx("vkCreateGraphicsPipelines")?;
    match pipeline {
        Some(pipeline) if code == vk::Result::SUCCESS => Ok(pipeline),
        _ => fail(format!("vkCreateGraphicsPipelines returned {code}")),
    }
}

/// Creates the pipeline with binary capture, fetches each binary's key and bytes, then releases
/// the captured data; the binaries and the pipeline are destroyed on return.
pub fn capture<'d>(device: &'d Device, recipe: &Recipe<'d>) -> Outcome<Vec<Captured>> {
    // SAFETY: `device` created `recipe`; there are no binaries.
    let pipeline = unsafe { create(device, recipe, None, true) }?;
    let info = vk::PipelineBinaryCreateInfoKHR::default().pipeline(pipeline.handle());
    let mut count = vk::PipelineBinaryHandlesInfoKHR::default();
    // SAFETY: `pipeline` was created with capture; a null array asks for the count and creates
    // nothing.
    let code = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut count) }
        .ctx("vkCreatePipelineBinariesKHR (count)")?;
    expect_success(code, "vkCreatePipelineBinariesKHR (count)")?;
    if count.pipeline_binary_count == 0 {
        return fail("the capturing pipeline has no binaries");
    }
    let mut created = vec![vk::PipelineBinaryKHR::null(); count.pipeline_binary_count as usize];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default().pipeline_binaries(&mut created);
    // SAFETY: as above; `handles` has room for the reported count.
    let result = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut handles) };
    let binaries = own_binaries(device, &created);
    let code = result.ctx("vkCreatePipelineBinariesKHR")?;
    expect_success(code, "vkCreatePipelineBinariesKHR")?;

    let mut captured = Vec::with_capacity(binaries.len());
    for binary in &binaries {
        let mut key = vk::PipelineBinaryKeyKHR::default();
        let data_info = vk::PipelineBinaryDataInfoKHR::default().pipeline_binary(binary.handle());
        // SAFETY: `binary` is live on `device`.
        let data = unsafe { device.get_pipeline_binary_data_khr(&data_info, &mut key) }
            .ctx("vkGetPipelineBinaryDataKHR")?;
        captured.push(Captured { key, data });
    }
    let release = vk::ReleaseCapturedPipelineDataInfoKHR::default().pipeline(pipeline.handle());
    // SAFETY: `pipeline` was created with capture and its data has not been released.
    unsafe { device.release_captured_pipeline_data_khr(&release, None) }
        .ctx("vkReleaseCapturedPipelineDataKHR")?;
    Ok(captured)
}

/// Rebuilds the binaries from `captured` (keys and bytes only) and creates the pipeline from
/// them, without capture. The binaries are destroyed on return, after the pipeline exists.
pub fn rebuild<'d>(
    device: &'d Device,
    recipe: &Recipe<'d>,
    captured: &mut [Captured],
) -> Outcome<DeviceOwned<'d, vk::Pipeline>> {
    let keys: Vec<vk::PipelineBinaryKeyKHR<'static>> = captured.iter().map(|c| c.key).collect();
    let data: Vec<vk::PipelineBinaryDataKHR<'_>> = captured
        .iter_mut()
        .map(|c| vk::PipelineBinaryDataKHR::default().data(&mut c.data))
        .collect();
    let keys_and_data =
        vk::PipelineBinaryKeysAndDataKHR::default().pipeline_binary_keys(&keys, &data);
    let info = vk::PipelineBinaryCreateInfoKHR::default().keys_and_data_info(&keys_and_data);
    let mut created = vec![vk::PipelineBinaryKHR::null(); keys.len()];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default().pipeline_binaries(&mut created);
    // SAFETY: `device` is live; every key and byte span came from `vkGetPipelineBinaryDataKHR`
    // on this device and outlives the call; `handles` has room for every binary.
    let result = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut handles) };
    // Held until return, so the binaries outlive the pipeline's creation.
    let _binaries = own_binaries(device, &created);
    let code = result.ctx("vkCreatePipelineBinariesKHR (from data)")?;
    expect_success(code, "vkCreatePipelineBinariesKHR (from data)")?;
    // SAFETY: `device` created `recipe` and the binaries, in the order capture returned them
    // (every entry is non-null after a `SUCCESS`).
    unsafe { create(device, recipe, Some(created.as_slice()), false) }
}

/// `code` must be `SUCCESS`: the other success codes of `vkCreatePipelineBinariesKHR`
/// (`INCOMPLETE`, `PIPELINE_BINARY_MISSING_KHR`) mean the round trip did not happen.
fn expect_success(code: vk::Result, what: &str) -> Outcome<()> {
    if code == vk::Result::SUCCESS {
        Ok(())
    } else {
        fail(format!("{what} returned {code}"))
    }
}
