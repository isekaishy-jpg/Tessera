//! Compile-time checks: `push_next` accepts only structs the registry lets extend the head
//! (`structextends`), `returnedonly` structs have no setters, and byte parameters and setters
//! accept only `Plain` types. Each case in `compile_fail/` must fail to compile with the error
//! recorded next to it.

#[test]
fn wrong_code_does_not_compile() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile_fail/*.rs");
}
