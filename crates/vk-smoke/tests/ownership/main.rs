//! Ownership on a mock device: every handle created is destroyed exactly once, never while the
//! device may still use it, and nothing is destroyed when completion of a submit cannot be
//! established.
//!
//! - `create`: each fallible call of the creating functions fails in turn (batch creates return
//!   a partial batch), and every handle created is destroyed.
//! - `submit`: `Frame::render` under every scripted submit, fence and queue result.
//! - `teardown`: the device's close and drop under every scripted device-idle result, the
//!   instance's drop, and the panic on a second messenger.
//!
//! The mocks keep their state in thread-locals (tests run in parallel threads) and record faults
//! instead of panicking: a panic cannot unwind out of an `extern "system"` function. Every test
//! starts from [`reset`].

mod create;
mod mock;
mod submit;
mod teardown;

use core::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use tessera_vk::vk;

/// The results the scripted commands return.
#[derive(Clone, Copy)]
struct Script {
    /// `vkQueueSubmit2`.
    submit: vk::Result,
    /// `vkWaitForFences`.
    fence: vk::Result,
    /// `vkQueueWaitIdle`.
    queue: vk::Result,
    /// `vkDeviceWaitIdle`.
    idle: vk::Result,
}

impl Script {
    /// Every scripted command succeeds.
    const SUCCESS: Script = Script {
        submit: vk::Result::SUCCESS,
        fence: vk::Result::SUCCESS,
        queue: vk::Result::SUCCESS,
        idle: vk::Result::SUCCESS,
    };
}

thread_local! {
    /// The fallible call that fails (1-based); 0 for none.
    static FAIL_AT: Cell<u32> = const { Cell::new(0) };
    /// Fallible calls made since the last [`arm`].
    static CALLS: Cell<u32> = const { Cell::new(0) };
    /// The next handle value; never reset, so no two handles of a thread are equal.
    static NEXT: Cell<u64> = const { Cell::new(0x100) };
    /// Live handles and their kinds.
    static LIVE: RefCell<BTreeMap<u64, &'static str>> = const { RefCell::new(BTreeMap::new()) };
    /// Invalid calls: destroys of null, unknown, other-kind or pending handles, and parent
    /// destroys while work is pending.
    static FAULTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// The waits and the device and instance teardown, in order.
    static EVENTS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    /// Each bound buffer or image and its memory.
    static BOUND: RefCell<BTreeMap<u64, u64>> = const { RefCell::new(BTreeMap::new()) };
    /// Each image view and its image.
    static VIEWS: RefCell<BTreeMap<u64, u64>> = const { RefCell::new(BTreeMap::new()) };
    /// Each command buffer and its pool.
    static POOLS: RefCell<BTreeMap<u64, u64>> = const { RefCell::new(BTreeMap::new()) };
    /// Each command buffer and the handles its recorded commands name.
    static RECORDED: RefCell<BTreeMap<u64, BTreeSet<u64>>> =
        const { RefCell::new(BTreeMap::new()) };
    /// Handles a submit may still use.
    static PENDING: RefCell<BTreeSet<u64>> = const { RefCell::new(BTreeSet::new()) };
    /// The scripted results.
    static SCRIPT: Cell<Script> = const { Cell::new(Script::SUCCESS) };
}

/// Clears every mock state, scripts every command to succeed and fails no call.
fn reset() {
    arm(0);
    LIVE.with(|l| l.borrow_mut().clear());
    FAULTS.with(|f| f.borrow_mut().clear());
    EVENTS.with(|e| e.borrow_mut().clear());
    BOUND.with(|b| b.borrow_mut().clear());
    VIEWS.with(|v| v.borrow_mut().clear());
    POOLS.with(|p| p.borrow_mut().clear());
    RECORDED.with(|r| r.borrow_mut().clear());
    PENDING.with(|p| p.borrow_mut().clear());
    SCRIPT.with(|s| s.set(Script::SUCCESS));
}

/// Restarts the call count and fails call `n` (1-based; 0 fails none).
fn arm(n: u32) {
    CALLS.with(|c| c.set(0));
    FAIL_AT.with(|f| f.set(n));
}

/// Fallible calls made since the last [`arm`].
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

/// Sets the scripted results.
fn script(script: Script) {
    SCRIPT.with(|s| s.set(script));
}

/// The scripted results.
fn scripted() -> Script {
    SCRIPT.with(Cell::get)
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

/// Records a fault.
fn fault(text: String) {
    FAULTS.with(|f| f.borrow_mut().push(text));
}

/// Records the destroy of `handle`, which must be a live handle of `kind` that no submit may
/// still use.
fn destroyed(kind: &'static str, handle: u64) {
    if PENDING.with(|p| p.borrow().contains(&handle)) {
        fault(format!("{kind}: {handle:#x} destroyed while pending"));
    }
    match LIVE.with(|l| l.borrow_mut().remove(&handle)) {
        _ if handle == 0 => fault(format!("{kind}: null handle destroyed")),
        Some(k) if k == kind => {}
        Some(k) => fault(format!("{kind}: destroyed a {k} ({handle:#x})")),
        None => fault(format!(
            "{kind}: {handle:#x} is unknown or already destroyed"
        )),
    }
}

/// Logs the destroy of a parent (`event`: the device, the messenger or the instance);
/// destroying one while a submit may still use a child is a fault.
fn parent_destroyed(event: &'static str) {
    if PENDING.with(|p| !p.borrow().is_empty()) {
        fault(format!("{event} while work is pending"));
    }
    EVENTS.with(|e| e.borrow_mut().push(event));
}

/// True when a wait returning `code` ends the submitted work: `SUCCESS`, or `ERROR_DEVICE_LOST`
/// (a lost device completes every command).
fn settles(code: vk::Result) -> bool {
    code == vk::Result::SUCCESS || code == vk::Result::ERROR_DEVICE_LOST
}

/// Logs a wait `event` and returns `code`; a wait that [`settles`] ends every pending use.
fn waited(event: &'static str, code: vk::Result) -> vk::Result {
    EVENTS.with(|e| e.borrow_mut().push(event));
    if settles(code) {
        PENDING.with(|p| p.borrow_mut().clear());
    }
    code
}

/// True when any command was recorded since the last [`reset`].
fn recorded_any() -> bool {
    RECORDED.with(|r| !r.borrow().is_empty())
}

/// The live handles and their kinds.
fn live() -> BTreeMap<u64, &'static str> {
    LIVE.with(|l| l.borrow().clone())
}

/// The faults recorded so far.
fn faults() -> Vec<String> {
    FAULTS.with(|f| f.borrow().clone())
}

/// The events logged so far, in order.
fn events() -> Vec<&'static str> {
    EVENTS.with(|e| e.borrow().clone())
}

/// The full teardown after a settled run: the device's wait, then every destroy, in order.
const TEARDOWN: [&str; 4] = [
    "wait idle",
    "destroy device",
    "destroy messenger",
    "destroy instance",
];
