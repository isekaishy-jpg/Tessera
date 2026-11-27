//! `system_library` loads bare file names from System32 only. A copy of this test program runs
//! beside planted libraries, in a working directory that holds them too, and must never select
//! them: loading such a garbage file would fail with `ERROR_BAD_EXE_FORMAT` (193) instead of
//! finding the system's library or reporting `ERROR_MOD_NOT_FOUND` (126).

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::{env, fs};
use tessera_vk::{Entry, LoadError, system_library};

/// Set in the child's environment: it runs beside planted libraries.
const CHILD: &str = "TESSERA_VK_LOADER_CHILD";
/// A library name System32 does not hold.
const ABSENT: &str = "tessera-absent-1.dll";
/// `ERROR_MOD_NOT_FOUND`.
const MOD_NOT_FOUND: u32 = 126;
/// `ERROR_BAD_EXE_FORMAT`: what loading a planted garbage file reports.
const BAD_EXE_FORMAT: u32 = 193;
/// The libraries planted beside the child and in its working directory.
const PLANTED: [&str; 2] = ["vulkan-1.dll", ABSENT];

#[test]
fn only_bare_file_names_are_loaded() {
    for name in [
        "",
        ".",
        "..",
        "C:\\Windows\\System32\\kernel32.dll",
        "C:kernel32.dll",
        "sub\\kernel32.dll",
        "sub/kernel32.dll",
        "kernel32.dll\0x",
    ] {
        // SAFETY: a refused name loads nothing.
        let loaded = unsafe { system_library(name) };
        assert_eq!(loaded.map(|_| ()), Err(LoadError::Name), "{name:?}");
    }
}

#[test]
fn a_system_library_loads_and_resolves_symbols() {
    // SAFETY: kernel32 is loaded in every process already; nothing new initialises.
    let kernel32 = unsafe { system_library("kernel32.dll") }.expect("System32 holds kernel32");
    assert!(kernel32.symbol(c"GetProcAddress").is_some());
    assert!(kernel32.symbol(c"TesseraNoSuchExport").is_none());
}

/// The child's checks, which also hold in this process: an absent library is not found (126),
/// and `Entry::load` never selects a planted garbage file (193). Under [`CHILD`] it first
/// checks that the planted files are where the search would find them.
#[test]
fn planted_libraries_are_never_loaded() {
    if env::var_os(CHILD).is_some() {
        let exe = env::current_exe().expect("the child knows its path");
        let beside = exe.parent().expect("the child is in a directory");
        let cwd = env::current_dir().expect("the child has a working directory");
        assert_ne!(beside, cwd, "the two search locations are apart");
        for dir in [beside, cwd.as_path()] {
            for name in PLANTED {
                assert!(
                    dir.join(name).is_file(),
                    "{name} is planted in {}",
                    dir.display()
                );
            }
        }
    }
    // SAFETY: System32 holds no such library, so nothing loads.
    let absent = unsafe { system_library(ABSENT) };
    assert_eq!(
        absent.map(|_| ()),
        Err(LoadError::Library {
            code: MOD_NOT_FOUND
        })
    );
    // SAFETY: only the system's Vulkan runtime may initialise, which the test accepts.
    if let Err(e) = unsafe { Entry::load() } {
        assert_ne!(
            e,
            LoadError::Library {
                code: BAD_EXE_FORMAT
            },
            "a planted vulkan-1.dll was selected"
        );
    }
}

/// A temporary directory, removed when dropped (also when the test panics).
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Writes every planted garbage library into `dir`.
fn plant(dir: &Path) {
    fs::create_dir_all(dir).expect("the directory is created");
    for name in PLANTED {
        fs::write(dir.join(name), b"not a library").expect("the planted file is written");
    }
}

#[test]
fn a_child_beside_planted_libraries_ignores_them() {
    let root = TempDir(env::temp_dir().join(format!("tessera-vk-loader-{}", process::id())));
    let bin = root.0.join("bin");
    let work = root.0.join("work");
    plant(&bin);
    plant(&work);
    let child = bin.join("loader-child.exe");
    fs::copy(env::current_exe().expect("the test knows its path"), &child)
        .expect("the test program is copied");
    let out = Command::new(&child)
        .args([
            "--exact",
            "planted_libraries_are_never_loaded",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .current_dir(&work)
        .output()
        .expect("the child runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "the child failed:\n{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("1 passed"),
        "the child ran its one test:\n{stdout}"
    );
}
