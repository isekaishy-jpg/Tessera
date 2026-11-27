//! `Debug` helpers for generated types, and the doc-link macros `spec!` (a reference page) and
//! `section!` (a section of the specification).

use crate::text;
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

/// The Markdown link reference definition of a section of the specification, for
/// `#[doc = section!("anchor")]`: `[anchor]` or `[text][anchor]` in the item's docs links to it.
#[allow(
    unused_macros,
    reason = "generated docs use it only when a registry comment links a section"
)]
macro_rules! section {
    ($anchor:literal) => {
        concat!(
            "[",
            $anchor,
            "]: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#",
            $anchor
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

/// A fixed `char` array, printed as its text ([`text::fixed_str`]).
pub(crate) struct CharArray<'a>(pub(crate) &'a [c_char]);

impl fmt::Debug for CharArray<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&text::fixed_str(self.0), f)
    }
}
