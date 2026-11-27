# apps

The programs, such as `player`, `editor`, `studio` and `launcher`.

- **Depends on:** any layer except `tests/`. `player` never reaches `authoring/`, `pipeline/` or a nested role crate; `editor` and `studio` never depend on each other.
- **`unsafe`:** forbidden (`#![forbid(unsafe_code)]`).
- **Not here:** library code another program could share (it goes in the layer it belongs to).
- **Crates:** none yet.
