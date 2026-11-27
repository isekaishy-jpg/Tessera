# servers

Engine services, each over one platform backend, such as `render`, `physics`, `audio`, `nav` and `text`.

- **Depends on:** `platform/`, `core/`.
- **`unsafe`:** forbidden (`#![forbid(unsafe_code)]`).
- **Not here:** the world that uses the services (`scene/`); FFI (`platform/`).
- **Crates:** none yet.
