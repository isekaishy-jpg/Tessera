//! The shaders: `smoke.slang`, compiled with the SDK's `slangc` when the program runs (no SPIR-V
//! is committed, and building the workspace does not need the SDK).
//!
//! Each stage is its own module with one entry point named `main`. One module holding both
//! entry points is valid SPIR-V, but the NVIDIA driver then refuses to create a pipeline from
//! the binaries it captured (`PIPELINE_COMPILE_REQUIRED`).

use crate::failure::{Context, Failure, Outcome, fail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The Slang source: a vertex entry `vertexMain` and a fragment entry `fragmentMain`.
const SOURCE: &str = include_str!("smoke.slang");

/// The first word of every SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The compiled stages, each a SPIR-V module whose entry point is `main`.
pub struct Shaders {
    /// `vertexMain`.
    pub vertex: Vec<u32>,
    /// `fragmentMain`.
    pub fragment: Vec<u32>,
}

/// Compiles both stages of [`SOURCE`] in a fresh temp directory that is removed afterwards
/// (also on failure).
pub fn compile() -> Outcome<Shaders> {
    let slangc = slangc()?;
    let dir = std::env::temp_dir().join(format!("tessera-vk-smoke-{}", std::process::id()));
    let result = compile_in(&slangc, &dir);
    let _ = fs::remove_dir_all(&dir);
    result
}

/// `$VULKAN_SDK/Bin/slangc`.
fn slangc() -> Outcome<PathBuf> {
    let sdk = std::env::var_os("VULKAN_SDK")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            Failure(
                "VULKAN_SDK is not set: slangc and the validation layer come from the SDK".into(),
            )
        })?;
    let exe = format!("slangc{}", std::env::consts::EXE_SUFFIX);
    let path = PathBuf::from(sdk).join("Bin").join(exe);
    if !path.is_file() {
        return fail(format!(
            "{} not found: install the Vulkan SDK",
            path.display()
        ));
    }
    Ok(path)
}

fn compile_in(slangc: &Path, dir: &Path) -> Outcome<Shaders> {
    fs::create_dir_all(dir).ctx(&dir.display().to_string())?;
    let source = dir.join("smoke.slang");
    fs::write(&source, SOURCE).ctx(&source.display().to_string())?;
    Ok(Shaders {
        vertex: stage(slangc, &source, "vertexMain", "vertex")?,
        fragment: stage(slangc, &source, "fragmentMain", "fragment")?,
    })
}

/// Compiles `entry` of `source` as `stage` into a module beside it, and reads its words.
fn stage(slangc: &Path, source: &Path, entry: &str, stage: &str) -> Outcome<Vec<u32>> {
    let output = source.with_file_name(format!("{stage}.spv"));
    let run = Command::new(slangc)
        .arg(source)
        .args(["-target", "spirv", "-entry", entry, "-stage", stage, "-o"])
        .arg(&output)
        .output()
        .ctx(&slangc.display().to_string())?;
    if !run.status.success() {
        return fail(format!(
            "slangc ({entry}) failed:\n{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    words(&fs::read(&output).ctx(&output.display().to_string())?)
}

/// `bytes` as little-endian SPIR-V words, checked for length and magic.
fn words(bytes: &[u8]) -> Outcome<Vec<u32>> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return fail(format!(
            "slangc wrote {} bytes, not whole SPIR-V words",
            bytes.len()
        ));
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    if words[0] != SPIRV_MAGIC {
        return fail(format!(
            "slangc output starts with {:#010x}, not SPIR-V",
            words[0]
        ));
    }
    Ok(words)
}
