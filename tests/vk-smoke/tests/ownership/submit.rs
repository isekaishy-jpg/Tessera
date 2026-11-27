//! `Frame::render` under every scripted result: nothing is destroyed until completion is
//! established, and when it cannot be, nothing is destroyed at all.

use super::mock::{instance, owned_device, shaders, with_device};
use super::{Script, TEARDOWN, events, faults, live, recorded_any, reset, script, settles};
use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use tessera_vk::vk;
use tessera_vk_smoke::frame::Frame;
use tessera_vk_smoke::pipeline::Recipe;

/// What a render run left: its result, the handles live right after it, and the handles live
/// once everything was dropped.
struct Run {
    /// `Frame::render`'s failure, if it failed.
    failure: Option<String>,
    /// The handles live right after the render.
    after_render: BTreeMap<u64, &'static str>,
    /// The handles live once the device and instance were dropped.
    after_drop: BTreeMap<u64, &'static str>,
}

/// Builds the frame and the rebuilt pipeline on a mock device, renders once with `results`
/// scripted, then drops everything in scope order.
fn render(results: Script) -> Run {
    let (failure, after_render) = with_device(|device| {
        let frame = Frame::create(device).expect("the frame is created");
        let shaders = shaders();
        let recipe = Recipe::create(device, &shaders).expect("the recipe is created");
        let mut captured = recipe.capture().expect("the binaries are captured");
        let rebuilt = captured.rebuild().expect("the pipeline is rebuilt");
        script(results);
        // SAFETY: `rebuild` builds the pipeline for the frame's format and vertex layout; the
        // frame's command buffer is new; `render` has not run on this frame.
        let rendered = unsafe { frame.render(&rebuilt) };
        (rendered.err().map(|f| f.to_string()), live())
    });
    Run {
        failure,
        after_render,
        after_drop: live(),
    }
}

/// Requires that everything was destroyed after the device's wait, in order.
fn assert_destroyed(run: &Run, case: &str) {
    assert_eq!(run.after_drop, BTreeMap::new(), "{case}: a handle leaked");
    let log = events();
    assert!(log.ends_with(&TEARDOWN), "{case}: {log:?}");
}

/// Requires that nothing was destroyed: every handle live after the render is still live, and
/// neither the device's wait nor any parent destroy ran.
fn assert_abandoned(run: &Run, case: &str) {
    assert_eq!(
        run.after_drop, run.after_render,
        "{case}: a handle was destroyed"
    );
    let log = events();
    assert!(!log.contains(&"wait idle"), "{case}: {log:?}");
    assert!(!destroys_any(&log), "{case}: {log:?}");
    let failure = run.failure.as_deref().unwrap_or_default();
    assert!(
        failure.ends_with("completion not established: nothing is destroyed"),
        "{case}: {failure}"
    );
}

/// The fence results the matrix scripts.
const FENCE: [vk::Result; 5] = [
    vk::Result::SUCCESS,
    vk::Result::TIMEOUT,
    vk::Result::ERROR_DEVICE_LOST,
    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
];

/// The queue results the matrix scripts.
const QUEUE: [vk::Result; 4] = [
    vk::Result::SUCCESS,
    vk::Result::ERROR_DEVICE_LOST,
    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
];

/// True when `events` holds a destroy of the device, the messenger or the instance.
fn destroys_any(events: &[&str]) -> bool {
    events.iter().any(|e| e.starts_with("destroy"))
}

#[test]
fn every_fence_and_queue_result() {
    for fence in FENCE {
        for queue in QUEUE {
            let case = format!("fence {fence:?}, queue {queue:?}");
            let run = render(Script {
                fence,
                queue,
                ..Script::SUCCESS
            });
            assert_eq!(
                run.failure.is_none(),
                fence == vk::Result::SUCCESS,
                "{case}: {:?}",
                run.failure
            );
            let log = events();
            assert_eq!(
                log.contains(&"wait queue"),
                !settles(fence),
                "{case}: {log:?}"
            );
            assert_eq!(faults(), Vec::<String>::new(), "{case}");
            if settles(fence) || settles(queue) {
                assert_destroyed(&run, &case);
            } else {
                assert_abandoned(&run, &case);
            }
        }
    }
}

#[test]
fn a_submit_out_of_memory_leaves_nothing_pending() {
    for submit in [
        vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    ] {
        let case = format!("submit {submit:?}");
        let run = render(Script {
            submit,
            ..Script::SUCCESS
        });
        assert!(run.failure.is_some(), "{case}");
        let log = events();
        assert!(!log.contains(&"wait fence"), "{case}: {log:?}");
        assert!(!log.contains(&"wait queue"), "{case}: {log:?}");
        assert_eq!(faults(), Vec::<String>::new(), "{case}");
        assert_destroyed(&run, &case);
    }
}

#[test]
fn another_submit_failure_is_settled_by_the_queue() {
    let lost = |queue| Script {
        submit: vk::Result::ERROR_DEVICE_LOST,
        queue,
        ..Script::SUCCESS
    };

    let run = render(lost(vk::Result::SUCCESS));
    let log = events();
    assert!(!log.contains(&"wait fence"), "{log:?}");
    assert!(log.contains(&"wait queue"), "{log:?}");
    assert_eq!(faults(), Vec::<String>::new());
    assert_destroyed(&run, "queue SUCCESS");

    let run = render(lost(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    assert_eq!(faults(), Vec::<String>::new());
    assert_abandoned(&run, "queue ERROR_OUT_OF_HOST_MEMORY");
}

#[test]
fn a_pipeline_from_another_device_panics_before_any_call() {
    reset();
    {
        let instance = instance();
        let first = owned_device(&instance);
        let second = owned_device(&instance);
        let frame = Frame::create(&first).expect("the frame is created");
        let shaders = shaders();
        let recipe = Recipe::create(&second, &shaders).expect("the recipe is created");
        let mut captured = recipe.capture().expect("the binaries are captured");
        let rebuilt = captured.rebuild().expect("the pipeline is rebuilt");
        let panicked = panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the pipeline has the frame's format and vertex layout; the frame's
            // command buffer is new; `render` has not run on this frame. Its device differs,
            // which `render` checks before any call.
            let _ = unsafe { frame.render(&rebuilt) };
        }))
        .expect_err("render panics");
        let message = panicked.downcast_ref::<&str>().copied().unwrap_or_default();
        assert_eq!(message, "the pipeline belongs to another device");
        assert!(!recorded_any(), "a command was recorded before the panic");
    }
    let log = events();
    assert!(!log.contains(&"wait fence"), "{log:?}");
    assert_eq!(faults(), Vec::<String>::new());
    assert_eq!(live(), BTreeMap::new(), "a handle leaked");
}
