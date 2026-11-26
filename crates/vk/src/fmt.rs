//! `Debug` helpers for generated types, and the `spec!` doc-link macro.

use core::ffi::c_char;
use core::fmt;

/// The Markdown link to a registry name's reference page, for `#[doc = spec!("VkName")]`.
macro_rules! spec {
    ($name:literal) => {
        concat!(
            "[`",
            $name,
            "`](https://registry.khronos.org/vulkan/specs/latest/man/html/",
            $name,
            ".html) in the Vulkan specification."
        )
    };
}

/// Writes flag bits as `TRANSFER_SRC | VERTEX_BUFFER | 0x100`: known single bits by name, any
/// rest in hex; `empty` for none.
pub(crate) fn flags(f: &mut fmt::Formatter<'_>, bits: u64, known: &[(u64, &str)]) -> fmt::Result {
    if bits == 0 {
        return f.write_str("empty");
    }
    let mut rest = bits;
    let mut first = true;
    let mut sep = |f: &mut fmt::Formatter<'_>| {
        let s = if first { "" } else { " | " };
        first = false;
        f.write_str(s)
    };
    for &(bit, name) in known {
        if bits & bit == bit {
            sep(f)?;
            f.write_str(name)?;
            rest &= !bit;
        }
    }
    if rest != 0 {
        sep(f)?;
        write!(f, "{rest:#x}")?;
    }
    Ok(())
}

/// Writes a struct as `Name { a: .., b: .. }` from local bindings named like its fields, each
/// named once (some field names are too long to write twice on one line).
macro_rules! debug_fields {
    ($f:expr, $name:literal $(, $field:ident)* $(,)?) => {
        $f.debug_struct($name)$(.field(stringify!($field), &$field))*.finish()
    };
}

/// Writes an enum value: its name, or `Type(raw)` for a value these bindings do not know.
pub(crate) fn enum_value(
    f: &mut fmt::Formatter<'_>,
    ty: &str,
    name: Option<&str>,
    raw: i64,
) -> fmt::Result {
    match name {
        Some(n) => f.write_str(n),
        None => write!(f, "{ty}({raw})"),
    }
}

/// A fixed `char` array, printed as the string up to its first NUL.
pub(crate) struct CharArray<'a>(pub(crate) &'a [c_char]);

impl fmt::Debug for CharArray<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes: Vec<u8> = self
            .0
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c.to_ne_bytes()[0])
            .collect();
        fmt::Debug::fmt(&*String::from_utf8_lossy(&bytes), f)
    }
}
