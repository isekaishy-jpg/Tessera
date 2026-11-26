//! The graphics pipeline and the `VK_KHR_pipeline_binary` round trip: a pipeline created with
//! binary capture gives its binaries' keys and bytes, and a second pipeline is created from
//! binaries rebuilt from those bytes alone.

use crate::failure::{Context, Outcome, fail};
use crate::frame::{EXTENT, FORMAT, VERTEX_FORMAT, VERTEX_STRIDE};
use crate::shader::Shaders;
use tessera_vk::{Device, vk};

/// What both pipelines are built from: one shader module per stage and an empty pipeline
/// layout.
pub struct Recipe {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    layout: vk::PipelineLayout,
}

/// One captured pipeline binary: its key and its bytes.
pub struct Captured {
    key: vk::PipelineBinaryKeyKHR<'static>,
    data: Vec<u8>,
}

impl Captured {
    /// The binary's size in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Recipe {
    /// The modules of `shaders` and an empty pipeline layout.
    pub fn create(device: &Device, shaders: &Shaders) -> Outcome<Recipe> {
        let vertex = module(device, &shaders.vertex)?;
        let fragment = module(device, &shaders.fragment)?;
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        // SAFETY: `device` is live; the layout is empty.
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .ctx("vkCreatePipelineLayout")?;
        Ok(Recipe {
            vertex,
            fragment,
            layout,
        })
    }

    /// Destroys the modules and the layout.
    ///
    /// # Safety
    ///
    /// `device` created this recipe, and no pipeline is being created from it.
    pub unsafe fn destroy(self, device: &Device) {
        // SAFETY: the caller guarantees `device` owns every object and none is in use.
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_shader_module(self.fragment, None);
            device.destroy_shader_module(self.vertex, None);
        }
    }
}

/// The shader module of `spirv`.
fn module(device: &Device, spirv: &[u32]) -> Outcome<vk::ShaderModule> {
    let info = vk::ShaderModuleCreateInfo::default().code(spirv);
    // SAFETY: `device` is live; `spirv` is a whole SPIR-V module (checked by `shader`).
    unsafe { device.create_shader_module(&info, None) }.ctx("vkCreateShaderModule")
}

/// Creates the triangle pipeline from `recipe`: capturing binary data when `capture`, or from
/// `binaries` when given. Every other field is the same for both pipelines, as
/// `VkPipelineBinaryInfoKHR` requires.
///
/// # Safety
///
/// `device` created `recipe` and every binary.
unsafe fn create(
    device: &Device,
    recipe: &Recipe,
    binaries: Option<&[vk::PipelineBinaryKHR]>,
    capture: bool,
) -> Outcome<vk::Pipeline> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(recipe.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(recipe.fragment)
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
        .layout(recipe.layout)
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
    let code = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None, &mut pipelines)
    }
    .ctx("vkCreateGraphicsPipelines")?;
    if code != vk::Result::SUCCESS {
        return fail(format!("vkCreateGraphicsPipelines returned {code}"));
    }
    Ok(pipelines[0])
}

/// Creates the pipeline with binary capture, fetches each binary's key and bytes, then releases
/// the captured data and destroys the binaries and the pipeline.
pub fn capture(device: &Device, recipe: &Recipe) -> Outcome<Vec<Captured>> {
    // SAFETY: `device` created `recipe`; there are no binaries.
    let pipeline = unsafe { create(device, recipe, None, true) }?;
    let info = vk::PipelineBinaryCreateInfoKHR::default().pipeline(pipeline);
    let mut count = vk::PipelineBinaryHandlesInfoKHR::default();
    // SAFETY: `pipeline` was created with capture; a null array asks for the count.
    let code = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut count) }
        .ctx("vkCreatePipelineBinariesKHR (count)")?;
    expect_success(code, "vkCreatePipelineBinariesKHR (count)")?;
    if count.pipeline_binary_count == 0 {
        return fail("the capturing pipeline has no binaries");
    }
    let mut binaries = vec![vk::PipelineBinaryKHR::null(); count.pipeline_binary_count as usize];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default().pipeline_binaries(&mut binaries);
    // SAFETY: as above; `handles` has room for the reported count.
    let code = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut handles) }
        .ctx("vkCreatePipelineBinariesKHR")?;
    expect_success(code, "vkCreatePipelineBinariesKHR")?;

    let mut captured = Vec::with_capacity(binaries.len());
    for &binary in &binaries {
        let mut key = vk::PipelineBinaryKeyKHR::default();
        let data_info = vk::PipelineBinaryDataInfoKHR::default().pipeline_binary(binary);
        // SAFETY: `binary` was just created on `device`.
        let data = unsafe { device.get_pipeline_binary_data_khr(&data_info, &mut key) }
            .ctx("vkGetPipelineBinaryDataKHR")?;
        captured.push(Captured { key, data });
    }
    let release = vk::ReleaseCapturedPipelineDataInfoKHR::default().pipeline(pipeline);
    // SAFETY: `pipeline` was created with capture and its data has not been released; the
    // binaries and the pipeline are not used again.
    unsafe {
        device
            .release_captured_pipeline_data_khr(&release, None)
            .ctx("vkReleaseCapturedPipelineDataKHR")?;
        for &binary in &binaries {
            device.destroy_pipeline_binary_khr(binary, None);
        }
        device.destroy_pipeline(pipeline, None);
    }
    Ok(captured)
}

/// Rebuilds the binaries from `captured` (keys and bytes only) and creates the pipeline from
/// them, without capture. The binaries are destroyed once the pipeline exists.
pub fn rebuild(
    device: &Device,
    recipe: &Recipe,
    captured: &mut [Captured],
) -> Outcome<vk::Pipeline> {
    let keys: Vec<vk::PipelineBinaryKeyKHR<'static>> = captured.iter().map(|c| c.key).collect();
    let data: Vec<vk::PipelineBinaryDataKHR<'_>> = captured
        .iter_mut()
        .map(|c| vk::PipelineBinaryDataKHR::default().data(&mut c.data))
        .collect();
    let keys_and_data =
        vk::PipelineBinaryKeysAndDataKHR::default().pipeline_binary_keys(&keys, &data);
    let info = vk::PipelineBinaryCreateInfoKHR::default().keys_and_data_info(&keys_and_data);
    let mut binaries = vec![vk::PipelineBinaryKHR::null(); keys.len()];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default().pipeline_binaries(&mut binaries);
    // SAFETY: `device` is live; every key and byte span came from `vkGetPipelineBinaryDataKHR`
    // on this device and outlives the call; `handles` has room for every binary.
    let code = unsafe { device.create_pipeline_binaries_khr(&info, None, &mut handles) }
        .ctx("vkCreatePipelineBinariesKHR (from data)")?;
    expect_success(code, "vkCreatePipelineBinariesKHR (from data)")?;
    // SAFETY: `device` created `recipe` and the binaries, in the order capture returned them.
    let pipeline = unsafe { create(device, recipe, Some(&binaries), false) };
    for &binary in &binaries {
        // SAFETY: pipeline creation has returned; the binaries are not used again.
        unsafe { device.destroy_pipeline_binary_khr(binary, None) };
    }
    pipeline
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
