//! The shaders: `smoke.slang`, compiled with the SDK's `slangc` when the program runs (no SPIR-V
//! is committed, and building the workspace does not need the SDK).
//!
//! Each stage is its own module with one entry point named `main`. One module holding both
//! entry points is valid SPIR-V, but the NVIDIA driver then refuses to create a pipeline from
//! the binaries it captured (`PIPELINE_COMPILE_REQUIRED`).

use crate::failure::{Context, Outcome, fail};
use std::env::consts::EXE_SUFFIX;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::{env, fs};

/// The Slang source: a vertex entry `vertexMain` and a fragment entry `fragmentMain`.
const SOURCE: &str = include_str!("smoke.slang");

/// The first word of every SPIR-V module.
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// The two stages, each a SPIR-V module whose entry point is `main`. They come from [`compile`]
/// or the vouching [`Shaders::from_spirv`].
pub struct Shaders {
    /// The vertex stage (`vertexMain`).
    vertex: Vec<u32>,
    /// The fragment stage (`fragmentMain`).
    fragment: Vec<u32>,
}

impl Shaders {
    /// Stages given by the caller.
    ///
    /// # Safety
    ///
    /// Every device [`crate::pipeline::Recipe::create`] passes them to accepts each as its
    /// stage's module with entry point `main`. A real device requires valid SPIR-V
    /// (VUID-VkShaderModuleCreateInfo-pCode-08737); a mock that never reads them qualifies.
    pub unsafe fn from_spirv(vertex: Vec<u32>, fragment: Vec<u32>) -> Shaders {
        Shaders { vertex, fragment }
    }

    /// The vertex stage's module.
    pub fn vertex(&self) -> &[u32] {
        &self.vertex
    }

    /// The fragment stage's module.
    pub fn fragment(&self) -> &[u32] {
        &self.fragment
    }
}

/// Compiles both stages of `smoke.slang` in a fresh temp directory that is removed afterwards
/// (also on failure).
pub fn compile() -> Outcome<Shaders> {
    let slangc = slangc()?;
    let dir = TempDir(env::temp_dir().join(format!("tessera-vk-smoke-{}", process::id())));
    compile_in(&slangc, &dir.0)
}

/// A temporary directory, removed when dropped.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `$VULKAN_SDK/Bin/slangc`.
fn slangc() -> Outcome<PathBuf> {
    let Some(sdk) = env::var_os("VULKAN_SDK").filter(|s| !s.is_empty()) else {
        return fail("VULKAN_SDK is not set: slangc and the validation layer come from the SDK");
    };
    let exe = format!("slangc{EXE_SUFFIX}");
    let path = PathBuf::from(sdk).join("Bin").join(exe);
    if !path.is_file() {
        return fail(format!(
            "{} not found: install the Vulkan SDK",
            path.display()
        ));
    }
    Ok(path)
}

/// Writes the source into `dir` and compiles both stages there with `slangc`, whose output is
/// trusted to be valid SPIR-V (only its length and magic are checked).
fn compile_in(slangc: &Path, dir: &Path) -> Outcome<Shaders> {
    fs::create_dir_all(dir).ctx(dir.display())?;
    let source = dir.join("smoke.slang");
    fs::write(&source, SOURCE).ctx(source.display())?;
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
        .ctx(slangc.display())?;
    if !run.status.success() {
        return fail(format!(
            "slangc ({entry}) failed:\n{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    words(&fs::read(&output).ctx(output.display())?)
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
