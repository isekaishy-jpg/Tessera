//! Teardown of the device and the instance: the device is destroyed only after its wait
//! established completion, and the instance (messenger first) only when not abandoned.

use super::mock::{instance, owned_device, unowned_instance, with_device};
use super::{EVENTS, Script, TEARDOWN, events, reset, script};
use tessera_vk::vk;
use tessera_vk_smoke::owned::OwnedInstance;

/// The device-wait results that establish completion.
const SETTLED: [vk::Result; 2] = [vk::Result::SUCCESS, vk::Result::ERROR_DEVICE_LOST];

/// The device-wait results that do not.
const FAILED: [vk::Result; 2] = [
    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
];

/// Scripts `idle` for the device's wait.
fn script_idle(idle: vk::Result) {
    script(Script {
        idle,
        ..Script::SUCCESS
    });
}

/// Closes a mock device with `idle` scripted for its wait, then drops its instance; returns
/// the close's failure, if any, and the events (one device wait: the drop after `close` does
/// not close again).
fn close(idle: vk::Result) -> (Option<String>, Vec<&'static str>) {
    reset();
    let failure = {
        let instance = instance();
        let device = owned_device(&instance);
        script_idle(idle);
        device.close().err().map(|f| f.to_string())
    };
    (failure, events())
}

/// The events of dropping a mock device, unclosed, and its instance with `idle` scripted for
/// the device's wait.
fn drop_unclosed(idle: vk::Result) -> Vec<&'static str> {
    with_device(|_| script_idle(idle));
    events()
}

#[test]
fn a_settled_device_wait_destroys_everything() {
    for idle in SETTLED {
        assert_eq!(close(idle), (None, TEARDOWN.to_vec()), "{idle:?}");
    }
}

#[test]
fn a_failed_device_wait_fails_and_destroys_neither_device_nor_instance() {
    for idle in FAILED {
        let (failure, log) = close(idle);
        let failure = failure.unwrap_or_default();
        assert!(
            failure.ends_with(
                "completion not established: the device and the instance are not destroyed"
            ),
            "{idle:?}: {failure}"
        );
        assert_eq!(log, ["wait idle"], "{idle:?}");
    }
}

#[test]
fn a_dropped_device_is_closed() {
    for idle in SETTLED {
        assert_eq!(drop_unclosed(idle), TEARDOWN, "{idle:?}");
    }
    for idle in FAILED {
        assert_eq!(drop_unclosed(idle), ["wait idle"], "{idle:?}");
    }
}

#[test]
fn owned_instance_destroys_messenger_then_instance() {
    reset();
    // SAFETY: the mock instance and messenger are owned only here.
    unsafe {
        let mut owned = OwnedInstance::new(unowned_instance());
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(5));
    }
    assert_eq!(events(), ["destroy messenger", "destroy instance"]);
    EVENTS.with(|e| e.borrow_mut().clear());
    // SAFETY: as above, with no messenger.
    drop(unsafe { OwnedInstance::new(unowned_instance()) });
    assert_eq!(
        events(),
        ["destroy instance"],
        "no messenger, none destroyed"
    );
}

#[test]
#[should_panic(expected = "the instance already owns a messenger")]
fn a_second_messenger_panics() {
    // SAFETY: the mock instance and messengers are owned only here.
    unsafe {
        let mut owned = OwnedInstance::new(unowned_instance());
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(5));
        owned.set_messenger(vk::DebugUtilsMessengerEXT::from_raw(6));
    }
}
