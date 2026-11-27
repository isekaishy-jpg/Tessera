//! The `flags!` macro: the operations every generated flags type has.

/// Implements the flag operations for `$name`, a `#[repr(transparent)]` newtype over `$raw`:
/// `empty`, `from_raw`, `as_raw`, `is_empty`, `contains`, `intersects`, and `| & ^ ! -` with
/// their assign forms (`-` is the difference).
macro_rules! flags {
    ($name:ident, $raw:ty) => {
        impl $name {
            /// No flags.
            #[inline]
            pub const fn empty() -> Self {
                Self(0)
            }

            /// The flags with raw bits `bits`; bits these bindings do not know are kept.
            #[inline]
            pub const fn from_raw(bits: $raw) -> Self {
                Self(bits)
            }

            /// The raw bits.
            #[inline]
            pub const fn as_raw(self) -> $raw {
                self.0
            }

            /// True when no flag is set.
            #[inline]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// True when every flag of `other` is set.
            #[inline]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// True when any flag of `other` is set.
            #[inline]
            pub const fn intersects(self, other: Self) -> bool {
                self.0 & other.0 != 0
            }
        }

        impl core::ops::BitOr for $name {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl core::ops::BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl core::ops::BitAnd for $name {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl core::ops::BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl core::ops::BitXor for $name {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl core::ops::BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl core::ops::Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self {
                Self(!self.0)
            }
        }

        impl core::ops::Sub for $name {
            type Output = Self;

            /// The flags of `self` that are not in `rhs`.
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 & !rhs.0)
            }
        }

        impl core::ops::SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 &= !rhs.0;
            }
        }
    };
}
