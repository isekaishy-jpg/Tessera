//! The validation layer: the instance enables `VK_LAYER_KHRONOS_validation` with synchronization
//! validation on, and every warning or error message it (or the loader) reports is counted.

use crate::failure::{Context, Outcome, fail};
use crate::text;
use core::ffi::{CStr, c_void};
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use tessera_vk::{Entry, Instance, vk};

/// The Khronos validation layer.
const LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Warning and error messages reported so far.
static MESSAGES: AtomicU32 = AtomicU32::new(0);

/// Warning and error messages reported so far.
pub fn messages() -> u32 {
    MESSAGES.load(Ordering::SeqCst)
}

/// Counts the message and writes it to stderr. It must not panic: a panic cannot unwind out of
/// an `extern "system"` function, so writes ignore their result (`eprintln!` would panic on a
/// failed write).
unsafe extern "system" fn on_message(
    severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    MESSAGES.fetch_add(1, Ordering::SeqCst);
    // SAFETY: the layer passes callback data valid for this call, or null.
    let message = unsafe { data.as_ref() }
        .map(|d| d.p_message)
        .filter(|p| !p.is_null())
        // SAFETY: a non-null `pMessage` is a NUL-terminated string valid for this call.
        .map(|p| unsafe { CStr::from_ptr(p) }.to_string_lossy());
    let text = message.as_deref().unwrap_or("<no message>");
    let _ = writeln!(
        std::io::stderr().lock(),
        "validation {severity:?} {types:?}: {text}"
    );
    vk::FALSE
}

/// The messenger: warnings and errors of every message type, to [`on_message`].
fn messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(on_message))
}

/// Creates the instance (Vulkan 1.4) with the validation layer, synchronization validation on
/// (`validate_sync` through `VK_EXT_layer_settings`) and a messenger. A messenger is also
/// chained into the create info, so instance creation and destruction are counted too.
pub fn create_instance(entry: &Entry) -> Outcome<(Instance, vk::DebugUtilsMessengerEXT)> {
    // SAFETY: global enumerations with no preconditions.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .ctx("vkEnumerateInstanceLayerProperties")?;
    let layer = LAYER.to_string_lossy();
    if !layers.iter().any(|l| text::fixed(&l.layer_name) == layer) {
        return fail(format!("{layer} is not installed: install the Vulkan SDK"));
    }
    // SAFETY: as above; `LAYER` is a listed layer.
    let layer_extensions = unsafe { entry.enumerate_instance_extension_properties(Some(LAYER)) }
        .ctx("vkEnumerateInstanceExtensionProperties")?;
    if !text::has_extension(&layer_extensions, vk::EXT_LAYER_SETTINGS_EXTENSION_NAME) {
        return fail(format!("{layer} does not offer VK_EXT_layer_settings"));
    }

    let app = vk::ApplicationInfo::default()
        .application_name(c"tessera-vk-smoke")
        .api_version(vk::API_VERSION_1_4);
    let layer_names = [LAYER.as_ptr()];
    let extension_names = [
        vk::EXT_DEBUG_UTILS_EXTENSION_NAME.as_ptr(),
        vk::EXT_LAYER_SETTINGS_EXTENSION_NAME.as_ptr(),
    ];
    let validate_sync: [vk::Bool32; 1] = [vk::TRUE];
    let mut setting = vk::LayerSettingEXT::default()
        .layer_name(LAYER)
        .setting_name(c"validate_sync")
        .r#type(vk::LayerSettingTypeEXT::BOOL32)
        .values(&validate_sync);
    // `valueCount` counts values, but `values` sets the byte length (a vkgen defect carried to
    // the sprint review): one `Bool32`.
    setting.value_count = 1;
    let settings = [setting];
    let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    let mut chained = messenger_info();
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&extension_names)
        .push_next(&mut layer_settings)
        .push_next(&mut chained);
    // SAFETY: every pointer in `info` borrows a local that outlives the call; the layer and its
    // extension were checked above, and `VK_EXT_debug_utils` is a loader extension.
    let instance = unsafe { entry.create_instance(&info, None) }.ctx("vkCreateInstance")?;
    // SAFETY: `instance` is live and enables `VK_EXT_debug_utils`.
    let messenger = unsafe { instance.create_debug_utils_messenger_ext(&messenger_info(), None) }
        .ctx("vkCreateDebugUtilsMessengerEXT")?;
    Ok((instance, messenger))
}
