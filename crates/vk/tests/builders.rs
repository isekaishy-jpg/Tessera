//! Struct setters: every setter of a group writes the whole group, so setters compose in any order
//! (groups set together, parallel, alternatives and selected; count-only setters; expression
//! lengths), `bool` for `VkBool32`, strings, byte lengths, `push_next`, packed bitfields and
//! `Default`.

use core::ffi::c_void;
use tessera_vk::vk;

#[test]
fn default_sets_the_structure_type_and_zeroes_the_rest() {
    let info = vk::BufferCreateInfo::default();
    assert_eq!(info.s_type, vk::StructureType::BUFFER_CREATE_INFO);
    assert!(info.p_next.is_null());
    assert_eq!(info.size, 0);
    assert_eq!(
        <vk::BufferCreateInfo as vk::TaggedStructure>::STRUCTURE_TYPE,
        vk::StructureType::BUFFER_CREATE_INFO
    );
}

#[test]
fn value_setters_and_bool32_take_bool() {
    let info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS);
    assert_eq!(info.depth_test_enable, vk::TRUE);
    assert_eq!(info.depth_write_enable, vk::FALSE);
    assert_eq!(info.depth_compare_op, vk::CompareOp::LESS);
}

#[test]
fn a_group_with_required_arrays_is_set_together() {
    let semaphores = [vk::Semaphore::from_raw(1), vk::Semaphore::from_raw(2)];
    let stages = [vk::PipelineStageFlags::TOP_OF_PIPE; 2];
    let info = vk::SubmitInfo::default().wait_semaphores(&semaphores, &stages);
    assert_eq!(info.wait_semaphore_count, 2);
    assert_eq!(info.p_wait_semaphores, semaphores.as_ptr());
    assert_eq!(info.p_wait_dst_stage_mask, stages.as_ptr());
}

#[test]
#[should_panic(expected = "waitSemaphoreCount: arrays differ in length")]
fn arrays_set_together_must_agree() {
    let semaphores = [vk::Semaphore::from_raw(1), vk::Semaphore::from_raw(2)];
    let _ = vk::SubmitInfo::default()
        .wait_semaphores(&semaphores, &[vk::PipelineStageFlags::TOP_OF_PIPE]);
}

#[test]
fn an_optional_array_in_a_group_is_an_option() {
    let color = [vk::AttachmentReference::default()];
    let resolve = [vk::AttachmentReference::default().attachment(3)];
    let with = vk::SubpassDescription::default().color_attachments(&color, Some(&resolve));
    assert_eq!(with.color_attachment_count, 1);
    assert_eq!(with.p_resolve_attachments, resolve.as_ptr());
    let without = vk::SubpassDescription::default().color_attachments(&color, None);
    assert!(without.p_resolve_attachments.is_null());
}

#[test]
fn an_alternative_nulls_its_siblings() {
    let geometries = [vk::AccelerationStructureGeometryKHR::default()];
    let pointers = [core::ptr::from_ref(&geometries[0]); 2];
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .geometries(&geometries)
        .pp_geometries(&pointers);
    assert_eq!(info.geometry_count, 2);
    assert_eq!(info.pp_geometries, pointers.as_ptr());
    assert!(
        info.p_geometries.is_null(),
        "the alternative set first is dropped"
    );
    let counted = info.geometry_count(5);
    assert_eq!(counted.geometry_count, 5);
    assert!(counted.p_geometries.is_null() && counted.pp_geometries.is_null());
}

#[test]
fn write_descriptor_set_arrays_are_alternatives() {
    let images = [vk::DescriptorImageInfo::default(); 2];
    let buffers = [vk::DescriptorBufferInfo::default(); 3];
    let write = vk::WriteDescriptorSet::default()
        .image_info(&images)
        .buffer_info(&buffers);
    assert_eq!(write.descriptor_count, 3);
    assert_eq!(write.p_buffer_info, buffers.as_ptr());
    assert!(write.p_image_info.is_null() && write.p_texel_buffer_view.is_null());
    let inline = vk::WriteDescriptorSet::default()
        .image_info(&images)
        .descriptor_count(16);
    assert_eq!(
        inline.descriptor_count, 16,
        "an inline uniform block sets only the count"
    );
    assert!(
        inline.p_image_info.is_null()
            && inline.p_buffer_info.is_null()
            && inline.p_texel_buffer_view.is_null()
    );
}

#[test]
fn a_count_setter_nulls_its_array() {
    let samplers = [vk::Sampler::from_raw(4); 3];
    let with = vk::DescriptorSetLayoutBinding::default().immutable_samplers(&samplers);
    assert_eq!(with.descriptor_count, 3);
    let without = with.descriptor_count(8);
    assert_eq!(without.descriptor_count, 8);
    assert!(without.p_immutable_samplers.is_null());
}

#[test]
fn parallel_arrays_share_one_count() {
    let offset = [1u64, 2, 3];
    let extent = [4u64, 5, 6];
    let copy = vk::TensorCopyARM::default().src_offset(Some(&offset), None, Some(&extent));
    assert_eq!(copy.dimension_count, 3);
    assert_eq!(copy.p_src_offset, offset.as_ptr());
    assert!(copy.p_dst_offset.is_null());
    assert_eq!(copy.p_extent, extent.as_ptr());
    let none = copy.src_offset(None, None, None);
    assert_eq!(none.dimension_count, 0);
    assert!(none.p_src_offset.is_null() && none.p_extent.is_null());
}

#[test]
#[should_panic(expected = "dimensionCount: arrays differ in length")]
fn parallel_arrays_must_agree() {
    let _ = vk::TensorCopyARM::default().src_offset(Some(&[1, 2]), Some(&[1]), None);
}

#[test]
fn a_sample_mask_comes_with_its_sample_count() {
    let mask = [u32::MAX, 1];
    let ms = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_mask(vk::SampleCountFlags::COUNT_64, &mask);
    assert_eq!(ms.rasterization_samples, vk::SampleCountFlags::COUNT_64);
    assert_eq!(ms.p_sample_mask, mask.as_ptr());
    let fewer = ms.rasterization_samples(vk::SampleCountFlags::COUNT_4);
    assert!(
        fewer.p_sample_mask.is_null(),
        "the mask's length depends on the sample count"
    );
}

#[test]
#[should_panic(expected = "pSampleMask: slice length is not (rasterizationSamples + 31) / 32")]
fn a_sample_mask_of_the_wrong_length_panics() {
    let _ = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_mask(vk::SampleCountFlags::COUNT_64, &[u32::MAX]);
}

#[test]
fn shader_code_sets_its_type_size_and_pointer() {
    let words = [0x0723_0203u32; 3];
    let spirv = vk::ShaderCreateInfoEXT::default().code_spirv(&words);
    assert_eq!(spirv.code_type, vk::ShaderCodeTypeEXT::SPIRV);
    assert_eq!(spirv.code_size, 12);
    assert_eq!(spirv.p_code, words.as_ptr().cast::<c_void>());
    let bytes = tessera_vk::AlignedBytes::from_bytes(&[7; 20]);
    let binary = spirv.code_binary(&bytes);
    assert_eq!(binary.code_type, vk::ShaderCodeTypeEXT::BINARY);
    assert_eq!(binary.code_size, 20);
    assert_eq!(binary.p_code, bytes.as_bytes().as_ptr().cast::<c_void>());
    assert!(binary.p_code.addr().is_multiple_of(16));
}

#[test]
fn layer_setting_values_are_counted_in_elements() {
    let values = [vk::TRUE, vk::FALSE];
    let setting = vk::LayerSettingEXT::default()
        .r#type(vk::LayerSettingTypeEXT::BOOL32)
        .values(&values);
    assert_eq!(
        setting.value_count, 2,
        "valueCount counts values of the setting's type, not bytes"
    );
    assert_eq!(setting.p_values, values.as_ptr().cast::<c_void>());
}

#[test]
fn strings_bytes_and_expression_lengths() {
    let name = c"tessera";
    let app = vk::ApplicationInfo::default().application_name(name);
    assert_eq!(app.p_application_name, name.as_ptr());

    let data = [1u32, 2, 3];
    let spec = vk::SpecializationInfo::default().data(&data);
    assert_eq!(
        spec.data_size, 12,
        "a byte array's count is its size in bytes"
    );
    assert_eq!(spec.p_data, data.as_ptr().cast::<c_void>());

    let code = [0x0723_0203u32; 4];
    let module = vk::ShaderModuleCreateInfo::default().code(&code);
    assert_eq!(
        module.code_size, 16,
        "codeSize / 4 elements: the size in bytes"
    );
    assert_eq!(module.p_code, code.as_ptr());

    let version = [0u8; 2 * vk::UUID_SIZE];
    let info = vk::AccelerationStructureVersionInfoKHR::default().version_data(&version);
    assert_eq!(info.p_version_data, version.as_ptr());
}

#[test]
fn push_next_links_each_struct_right_after_the_head() {
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let a: *mut c_void = core::ptr::from_mut(&mut vulkan12).cast();
    let b: *mut c_void = core::ptr::from_mut(&mut vulkan13).cast();
    let info = vk::DeviceCreateInfo::default()
        .push_next(&mut vulkan12)
        .push_next(&mut vulkan13);
    assert_eq!(info.p_next, b.cast_const(), "the last pushed is first");
    // SAFETY: both structs are alive; `push_next` linked them.
    let second = unsafe { (*b.cast::<vk::PhysicalDeviceVulkan13Features>()).p_next };
    assert_eq!(second, a);
    // SAFETY: as above.
    let end = unsafe { (*a.cast::<vk::PhysicalDeviceVulkan12Features>()).p_next };
    assert!(end.is_null());
}

#[test]
#[should_panic(expected = "the pushed struct already has a pNext chain")]
fn push_next_refuses_a_struct_with_a_chain() {
    let mut inner = vk::PhysicalDeviceVulkan11Features::default();
    let mut outer = vk::PhysicalDeviceFeatures2::default().push_next(&mut inner);
    let _ = vk::DeviceCreateInfo::default().push_next(&mut outer);
}

#[test]
fn packed_bitfields_read_and_write_their_own_bits() {
    let mut instance = vk::AccelerationStructureInstanceKHR::default();
    instance.set_instance_custom_index(0x12_3456);
    instance.set_mask(0xab);
    instance.set_flags(vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE);
    assert_eq!(instance.instance_custom_index(), 0x12_3456);
    assert_eq!(instance.mask(), 0xab);
    assert_eq!(instance.instance_custom_index_bits, 0xab12_3456);
    assert_eq!(
        instance.flags(),
        vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
    );
    assert_eq!(instance.instance_shader_binding_table_record_offset(), 0);
}

#[test]
#[should_panic(expected = "mask does not fit in 8 bits")]
fn a_bitfield_value_too_wide_panics() {
    vk::AccelerationStructureInstanceKHR::default().set_mask(0x100);
}
