//! Runs `tessera-vk-smoke` and checks its report: it succeeds, prints every step in order, and
//! reports no validation message. A step that stops running fails here even if the program
//! exits 0.

use std::process::Command;

/// The start of each `ok` line, in run order.
const STEPS: [&str; 11] = [
    "ok shader compiled",
    "ok instance with validation (sync on)",
    "ok gpu ",
    "ok device",
    "ok fence TIMEOUT and NOT_READY",
    "ok staged vertex buffer (map_memory2)",
    "ok pipeline binaries captured: ",
    "ok pipeline rebuilt from binaries",
    "ok rendered and read back",
    "ok pixels checked",
    "ok destroyed",
];

#[test]
fn runs_clean_under_validation() {
    let out = Command::new(env!("CARGO_BIN_EXE_tessera-vk-smoke"))
        .output()
        .expect("tessera-vk-smoke runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let report = format!(
        "stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "tessera-vk-smoke failed\n{report}");
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        STEPS.len() + 1,
        "one line per step plus the count\n{report}"
    );
    for (line, step) in lines.iter().zip(STEPS) {
        assert!(
            line.starts_with(step),
            "expected `{step}...`, got `{line}`\n{report}"
        );
    }
    assert_eq!(lines[STEPS.len()], "validation messages: 0", "{report}");
}
