# pipeline

Turning source content into shipped data: the asset database, import, cooking and shader compilation, such as `asset-db`, `import`, `cook` and `shader-compile`.

- **Depends on:** `modules/`, `scene/`, `servers/`, `platform/`, `core/`. `apps/player` never reaches this layer.
- **`unsafe`:** forbidden (`#![forbid(unsafe_code)]`).
- **Not here:** anything the player needs at run time.
- **Crates:** none yet.
