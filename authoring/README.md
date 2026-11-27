# authoring

The tool framework the Editor and Studio share, such as `toolkit`, `graph`, `history`, `inspector`, `gizmo` and `viewport`.

- **Depends on:** `pipeline/`, `modules/`, `scene/`, `servers/`, `platform/`, `core/`. `apps/player` never reaches this layer.
- **`unsafe`:** forbidden (`#![forbid(unsafe_code)]`).
- **Not here:** one feature's editor (it goes in that feature's role crate, `modules/<feature>/authoring`); anything the player needs.
- **Crates:** none yet.
