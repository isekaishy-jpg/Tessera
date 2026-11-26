//! Compile-time checks: `push_next` accepts only structs the registry lets extend the head
//! (`structextends`), and `returnedonly` structs have no setters. Each case in `compile_fail/`
//! must fail to compile with the error recorded next to it.

#[test]
fn wrong_chains_and_setters_do_not_compile() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile_fail/*.rs");
}
