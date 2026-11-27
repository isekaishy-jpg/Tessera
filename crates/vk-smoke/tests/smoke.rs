//! Runs `tessera-vk-smoke` and checks its report: it succeeds, prints every step in order, and
//! reports no validation message. A step that stops running fails here even if the program
//! exits 0.
//!
//! The program runs in a fresh empty directory with every `VK_*` variable removed and
//! `VK_LAYER_SETTINGS_PATH` naming a settings file written here, so no environment variable,
//! settings file or working-directory file from outside the test reaches the layer. What the
//! test cannot remove (per-user configuration) is caught by the program's probe step.

use std::path::PathBuf;
use std::process::{self, Command};
use std::{env, fs};

/// The start of each `ok` line, in run order.
const STEPS: [&str; 12] = [
    "ok shader compiled",
    "ok instance with validation",
    "ok gpu ",
    "ok device",
    "ok sync validation live (probe hazard reported)",
    "ok fence TIMEOUT and NOT_READY",
    "ok staged vertex buffer (map_memory2)",
    "ok pipeline binaries captured: ",
    "ok pipeline rebuilt from binaries",
    "ok rendered and read back",
    "ok pixels checked",
    "ok destroyed",
];

/// The layer settings file the program runs with: synchronization validation on.
const SETTINGS: &str = "khronos_validation.validate_sync = true\n";

/// A temporary directory, removed when dropped (also when the test panics).
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn runs_clean_under_validation() {
    let dir = TempDir(env::temp_dir().join(format!("tessera-vk-smoke-launch-{}", process::id())));
    let _ = fs::remove_dir_all(&dir.0);
    fs::create_dir_all(&dir.0).expect("the launch directory is created");
    let settings = dir.0.join("vk_layer_settings.txt");
    fs::write(&settings, SETTINGS).expect("the settings file is written");

    let mut command = Command::new(env!("CARGO_BIN_EXE_tessera-vk-smoke"));
    command.current_dir(&dir.0);
    for (name, _) in env::vars_os() {
        // Windows variable names are case-insensitive.
        if name
            .to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("VK_")
        {
            command.env_remove(&name);
        }
    }
    command.env("VK_LAYER_SETTINGS_PATH", &settings);
    let out = command.output();
    drop(dir);
    let out = out.expect("tessera-vk-smoke runs");

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
