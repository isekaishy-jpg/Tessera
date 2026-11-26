//! `Debug` and `Display` of generated types: enum and flag names, unknown values, structs with
//! string fields, handles and `vk::Result`; the text of fixed `char` arrays.

use core::ffi::c_char;
use tessera_vk::vk;

#[test]
fn enums_print_their_name_or_their_raw_value() {
    assert_eq!(
        format!("{:?}", vk::Format::R8G8B8A8_UNORM),
        "R8G8B8A8_UNORM"
    );
    assert_eq!(format!("{:?}", vk::Format::from_raw(-77)), "Format(-77)");
    assert_eq!(vk::PresentModeKHR::MAILBOX.name(), Some("MAILBOX"));
}

#[test]
fn flags_print_known_bits_by_name_and_the_rest_in_hex() {
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::VERTEX_BUFFER;
    assert_eq!(format!("{usage:?}"), "TRANSFER_SRC | VERTEX_BUFFER");
    let unknown = vk::BufferUsageFlags::from_raw(0x1 | 0x4000_0000);
    assert_eq!(format!("{unknown:?}"), "TRANSFER_SRC | 0x40000000");
    assert_eq!(format!("{:?}", vk::BufferUsageFlags::empty()), "empty");
    let stage = vk::PipelineStageFlags2::COPY;
    assert_eq!(format!("{stage:?}"), "COPY", "64-bit flags above bit 31");
}

#[test]
fn flag_operations() {
    let a = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE;
    assert!(a.contains(vk::ImageUsageFlags::SAMPLED));
    assert!(!a.contains(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST));
    assert!(a.intersects(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST));
    assert_eq!(
        a - vk::ImageUsageFlags::STORAGE,
        vk::ImageUsageFlags::SAMPLED
    );
    assert_eq!((a & vk::ImageUsageFlags::STORAGE).as_raw(), 0x8);
    assert!((a ^ a).is_empty());
}

#[test]
fn structs_list_their_fields_with_strings_readable() {
    let extent = vk::Extent2D::default().width(640).height(480);
    assert_eq!(
        format!("{extent:?}"),
        "Extent2D { width: 640, height: 480 }"
    );
    let mut props = vk::ExtensionProperties::default();
    for (dst, src) in props.extension_name.iter_mut().zip(b"VK_KHR_swapchain") {
        *dst = *src as c_char;
    }
    props.spec_version = 70;
    assert_eq!(
        format!("{props:?}"),
        "ExtensionProperties { extension_name: \"VK_KHR_swapchain\", spec_version: 70 }"
    );
}

#[test]
fn fixed_str_reads_up_to_the_first_nul() {
    let chars = |bytes: &[u8]| -> Vec<c_char> { bytes.iter().map(|&b| b as c_char).collect() };
    assert_eq!(tessera_vk::fixed_str(&chars(b"VK_LAYER\0junk")), "VK_LAYER");
    assert_eq!(tessera_vk::fixed_str(&chars(b"no nul")), "no nul");
    assert_eq!(tessera_vk::fixed_str(&chars(b"\0")), "");
    assert_eq!(
        tessera_vk::fixed_str(&chars(b"\xc3\xa9\xff\0")),
        "\u{e9}\u{fffd}",
        "UTF-8 decodes, invalid bytes are replaced"
    );
}

#[test]
fn handles_print_in_hex() {
    assert_eq!(format!("{:?}", vk::Buffer::from_raw(0x2a)), "Buffer(0x2a)");
    assert_eq!(format!("{:?}", vk::Device::null()), "Device(0x0)");
    assert_eq!(
        <vk::Buffer as vk::Handle>::OBJECT_TYPE,
        vk::ObjectType::BUFFER
    );
}

#[test]
fn results_display_their_name_and_description() {
    let text = vk::Result::ERROR_OUT_OF_HOST_MEMORY.to_string();
    assert!(text.starts_with("ERROR_OUT_OF_HOST_MEMORY: "), "{text}");
    assert_eq!(
        vk::Result::from_raw(12345).to_string(),
        "unknown VkResult 12345"
    );
    let error: &dyn std::error::Error = &vk::Result::ERROR_DEVICE_LOST;
    assert!(error.to_string().starts_with("ERROR_DEVICE_LOST"));
}

#[test]
fn api_versions() {
    let v = vk::make_api_version(0, 1, 4, 363);
    assert_eq!(
        (
            vk::api_version_variant(v),
            vk::api_version_major(v),
            vk::api_version_minor(v),
            vk::api_version_patch(v)
        ),
        (0, 1, 4, 363)
    );
    assert_eq!(vk::API_VERSION_1_4, vk::make_api_version(0, 1, 4, 0));
}
