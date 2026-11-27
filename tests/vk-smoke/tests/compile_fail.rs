//! Compile-time checks of the ownership boundaries: a device cannot outlive its instance, a
//! `Gpu` (which vouches for a physical device) is built only in `unsafe` code, and `Shaders`
//! (which vouch for SPIR-V) are not built from their fields. Each case in `compile_fail/` must
//! fail to compile with the error recorded next to it.

#[test]
fn wrong_code_does_not_compile() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile_fail/*.rs");
}
