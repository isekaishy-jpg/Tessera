# core

Foundations with no OS or GPU dependency, such as `ecs`, `reflect`, `math`, `containers`, `jobs`, `log` and `serial`.

- **Depends on:** nothing in the repo (third-party crates only).
- **`unsafe`:** allowed (ECS storage, containers); every `unsafe` block has a `// SAFETY:` comment.
- **Not here:** anything that calls the OS, a driver or a native library (`platform/`).
- **Crates:** none yet.
