# modules

Engine features, such as `ui`, `animation`, `particles` and `net`. A feature's editor and pipeline halves are nested role crates beside its runtime crate (`particles/authoring`, `particles/pipeline`); they count as `authoring/` and `pipeline/` and follow those layers' rules.

- **Depends on:** `scene/`, `servers/`, `platform/`, `core/`. A role crate depends as its role's layer does.
- **`unsafe`:** forbidden (`#![forbid(unsafe_code)]`).
- **Not here:** editor or import code inside a runtime crate (it goes in the feature's role crate).
- **Crates:** none yet.
