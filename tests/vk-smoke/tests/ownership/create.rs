//! Creation under failure: `Frame::create`, `Recipe::create`, `Recipe::capture` and
//! `Captured::rebuild` fail one fallible call per run (call 1, then call 2, ...), and every
//! handle created must be destroyed exactly once. Batch creates that fail return a partial
//! batch: a live handle in entry 0, null in the rest.

use super::mock::{shaders, with_device};
use super::{arm, calls, faults, live};
use std::collections::BTreeMap;
use tessera_vk_smoke::frame::Frame;
use tessera_vk_smoke::pipeline::Recipe;

/// More fallible calls than any function under test makes.
const MAX_CALLS: u32 = 100;

/// Runs `attempt` once per fallible call it makes, failing call 1, then call 2, ...; after
/// each run, the handles live are exactly `before` and no fault was recorded. Every run with a
/// failure must fail; the first run that makes fewer calls than the one armed (no failure)
/// must succeed. Returns the number of failures injected.
fn every_failure_cleans_up(
    before: &BTreeMap<u64, &'static str>,
    mut attempt: impl FnMut() -> bool,
) -> u32 {
    for n in 1..=MAX_CALLS {
        arm(n);
        let succeeded = attempt();
        let injected = calls() >= n;
        arm(0);
        assert_eq!(&live(), before, "call {n} failing: a handle leaked");
        assert_eq!(faults(), Vec::<String>::new(), "call {n} failing");
        if !injected {
            assert!(succeeded, "no call failed, yet the run failed");
            return n - 1;
        }
        assert!(!succeeded, "call {n} failed, yet the run succeeded");
    }
    panic!("more than {MAX_CALLS} fallible calls");
}

#[test]
fn frame_create_cleans_up_on_every_failure() {
    with_device(|device| {
        let failures = every_failure_cleans_up(&BTreeMap::new(), || Frame::create(device).is_ok());
        assert_ne!(failures, 0);
    });
}

#[test]
fn recipe_create_cleans_up_on_every_failure() {
    with_device(|device| {
        let shaders = shaders();
        let failures = every_failure_cleans_up(&BTreeMap::new(), || {
            Recipe::create(device, &shaders).is_ok()
        });
        assert_ne!(failures, 0);
    });
}

#[test]
fn capture_cleans_up_on_every_failure() {
    with_device(|device| {
        let shaders = shaders();
        let recipe = Recipe::create(device, &shaders).expect("the recipe is created");
        let before = live();
        let failures = every_failure_cleans_up(&before, || recipe.capture().is_ok());
        assert_ne!(failures, 0);
    });
}

#[test]
fn rebuild_cleans_up_on_every_failure() {
    with_device(|device| {
        let shaders = shaders();
        let recipe = Recipe::create(device, &shaders).expect("the recipe is created");
        let mut captured = recipe.capture().expect("the binaries are captured");
        let before = live();
        let failures = every_failure_cleans_up(&before, || captured.rebuild().is_ok());
        assert_ne!(failures, 0);
    });
}
