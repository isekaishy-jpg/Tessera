//! The layout check: the generated C program (`layout/layout.c`, compiled with MSVC against the
//! pinned SDK headers) and the generated Rust program (`layout/rust_layout.rs`, compiled with
//! the bindings) must print the same lines: the header version, every struct and union size and
//! alignment, every member offset and every packed bitfield's bit pattern.
//!
//! It needs the Vulkan SDK (`VULKAN_SDK`) and MSVC (found through `vswhere.exe`), and fails
//! naming whichever is missing.

#[path = "layout/rust_layout.rs"]
mod rust_layout;

use std::path::{Path, PathBuf};
use std::process::Command;

/// The SDK's include directory.
fn sdk_include() -> PathBuf {
    let sdk = std::env::var_os("VULKAN_SDK")
        .filter(|s| !s.is_empty())
        .expect("VULKAN_SDK is not set: the layout check compiles against the SDK's headers");
    let include = PathBuf::from(sdk).join("Include");
    assert!(
        include.join("vulkan").join("vulkan_core.h").is_file(),
        "no vulkan/vulkan_core.h under {}",
        include.display()
    );
    include
}

/// `vcvars64.bat` of the newest Visual Studio with the C++ tools.
fn vcvars() -> PathBuf {
    let program_files = std::env::var_os("ProgramFiles(x86)")
        .expect("ProgramFiles(x86) is not set: the layout check needs Windows and MSVC");
    let vswhere = PathBuf::from(program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    assert!(
        vswhere.is_file(),
        "{} not found: install Visual Studio with the C++ tools",
        vswhere.display()
    );
    let out = Command::new(&vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .expect("vswhere.exe did not run");
    let install = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        !install.is_empty(),
        "vswhere found no Visual Studio with the C++ x64 tools"
    );
    let bat = Path::new(&install)
        .join("VC")
        .join("Auxiliary")
        .join("Build")
        .join("vcvars64.bat");
    assert!(bat.is_file(), "{} not found", bat.display());
    bat
}

/// Compiles `layout.c` with `cl` in `dir` and returns the program's output lines.
fn c_lines(dir: &Path) -> Vec<String> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("layout")
        .join("layout.c");
    let exe = dir.join("layout.exe");
    let script = dir.join("build.bat");
    std::fs::write(
        &script,
        format!(
            "@call \"{}\" >nul\r\n\
             @cl /nologo /std:c11 /W3 /I\"{}\" /Fo\"{}\\\\\" /Fe\"{}\" \"{}\"\r\n\
             @exit /b %errorlevel%\r\n",
            vcvars().display(),
            sdk_include().display(),
            dir.display(),
            exe.display(),
            source.display()
        ),
    )
    .expect("write build.bat");
    let build = Command::new("cmd")
        .arg("/C")
        .arg(&script)
        .output()
        .expect("cmd did not run");
    assert!(
        build.status.success(),
        "cl failed:\n{}{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().expect("layout.exe did not run");
    assert!(run.status.success(), "layout.exe failed");
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

/// A temporary directory, removed when dropped (also when the test panics).
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn layout_matches_msvc() {
    let dir =
        TempDir(std::env::temp_dir().join(format!("tessera-vk-layout-{}", std::process::id())));
    std::fs::create_dir_all(&dir.0).expect("create the build directory");
    let c = c_lines(&dir.0);
    drop(dir);
    let rust = rust_layout::lines();
    let differences: Vec<String> = c
        .iter()
        .zip(&rust)
        .filter(|(a, b)| a != b)
        .take(20)
        .map(|(a, b)| format!("C:    {a}\nRust: {b}"))
        .collect();
    assert!(
        differences.is_empty(),
        "layouts differ:\n{}",
        differences.join("\n")
    );
    assert_eq!(
        c.len(),
        rust.len(),
        "the programs print different numbers of lines"
    );
    assert!(
        c[0].starts_with("V "),
        "the first line is the header version"
    );
}
