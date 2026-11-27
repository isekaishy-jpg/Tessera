//! The validation layer: the instance enables `VK_LAYER_KHRONOS_validation` and requests
//! synchronization validation; every warning or error message it (or the loader) reports is
//! counted against the gate. [`probe`] proves synchronization validation is live: it records one
//! known hazard and requires the layer to report exactly that.

use crate::failure::{Context, Outcome, fail};
use crate::owned::OwnedInstance;
use crate::text;
use core::ffi::{CStr, c_char, c_void};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tessera_vk::{Entry, vk};

/// The Khronos validation layer.
const LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// The message id of the hazard [`probe`] expects: two writes with no dependency between them.
pub const HAZARD: &CStr = c"SYNC-HAZARD-WRITE-AFTER-WRITE";

/// Messages counted against the gate.
static GATE: AtomicU32 = AtomicU32::new(0);
/// True while [`probe`] runs its recording.
static PROBING: AtomicBool = AtomicBool::new(false);
/// [`HAZARD`] messages seen while probing.
static HAZARDS: AtomicU32 = AtomicU32::new(0);

/// Warning and error messages counted against the gate so far (every message except the one
/// hazard a probe expects).
pub fn messages() -> u32 {
    GATE.load(Ordering::SeqCst)
}

/// Runs `record` with probing on, then requires that it succeeded, that the layer reported
/// [`HAZARD`] exactly once, and that it reported nothing else meanwhile.
pub fn probe(record: impl FnOnce() -> Outcome<()>) -> Outcome<()> {
    HAZARDS.store(0, Ordering::SeqCst);
    let gate = messages();
    PROBING.store(true, Ordering::SeqCst);
    let recorded = record();
    PROBING.store(false, Ordering::SeqCst);
    recorded?;
    let others = messages() - gate;
    if others != 0 {
        return fail(format!("the probe raised {others} other messages"));
    }
    match HAZARDS.load(Ordering::SeqCst) {
        1 => Ok(()),
        hazards => fail(format!(
            "the probe's hazard was reported {hazards} times, expected once: synchronization \
             validation (validate_sync, syncval_record_time_validation) is not live"
        )),
    }
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
    // SAFETY: the layer passes callback data valid for this call, or null.
    let data = unsafe { data.as_ref() };
    let text_of = |p: *const c_char| {
        // SAFETY: a non-null `pMessage` or `pMessageIdName` is a NUL-terminated string valid
        // for this call.
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) })
    };
    let id = data.and_then(|d| text_of(d.p_message_id_name));
    let text = data
        .and_then(|d| text_of(d.p_message))
        .map_or("<no message>".into(), CStr::to_string_lossy);
    let expected = PROBING.load(Ordering::SeqCst) && id == Some(HAZARD);
    let label = if expected {
        HAZARDS.fetch_add(1, Ordering::SeqCst);
        "probe (expected)"
    } else {
        GATE.fetch_add(1, Ordering::SeqCst);
        "validation"
    };
    let _ = writeln!(
        std::io::stderr().lock(),
        "{label} {severity:?} {types:?}: {text}"
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

/// Creates the instance (Vulkan 1.4) with the validation layer and a messenger, requesting
/// synchronization validation (`validate_sync` through `VK_EXT_layer_settings`, the
/// lowest-priority source; [`probe`] proves it is live). A messenger is also chained into the
/// create info, so instance creation and destruction are counted too.
pub fn create_instance(entry: &Entry) -> Outcome<OwnedInstance> {
    // SAFETY: global enumerations with no preconditions.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }
        .ctx("vkEnumerateInstanceLayerProperties")?;
    let layer = LAYER.to_string_lossy();
    if !layers
        .iter()
        .any(|l| tessera_vk::fixed_str(&l.layer_name) == layer)
    {
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
    let settings = [vk::LayerSettingEXT::default()
        .layer_name(LAYER)
        .setting_name(c"validate_sync")
        .r#type(vk::LayerSettingTypeEXT::BOOL32)
        .values(&validate_sync)];
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
    // SAFETY: `instance` was just created, and only this owner destroys it.
    let mut instance = unsafe { OwnedInstance::new(instance) };
    // SAFETY: `instance` is live and enables `VK_EXT_debug_utils`.
    let messenger = unsafe { instance.create_debug_utils_messenger_ext(&messenger_info(), None) }
        .ctx("vkCreateDebugUtilsMessengerEXT")?;
    // SAFETY: `messenger` was just created from `instance`, and only it destroys it.
    unsafe { instance.set_messenger(messenger) };
    Ok(instance)
}
