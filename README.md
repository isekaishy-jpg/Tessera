# Tessera

A 3D game engine and editor in one tool, written in Rust with Vulkan rendering.

## Status

Alpha. The first engine crate is `platform/vk` (`tessera-vk`): Vulkan bindings generated from the pinned Vulkan SDK (see `platform/vk/README.md`).

## Layout

One Cargo workspace. Each layer folder holds crates one level deep (`<layer>/<crate>/`), except the nested role crates (see Rules), and a `README.md` with its rules. A crate depends only on its own layer and the layers below it in this table:

| Layer | Holds |
|---|---|
| `apps/` | the programs: player, Editor, Studio, launcher |
| `authoring/` | the tool framework the Editor and Studio share |
| `pipeline/` | asset database, import, cooking, shader compilation |
| `modules/` | engine features: UI, animation, particles, networking |
| `scene/` | the world: placement, streaming, terrain, foliage, prefabs, assets, script API |
| `servers/` | engine services: rendering, physics, audio, navigation, text |
| `platform/` | FFI and single-API wrappers: Vulkan, windowing, input, files, native libraries |
| `core/` | foundations with no OS or GPU: ECS, reflection, math, containers, jobs |

Outside the chain:
- `tests/`: cross-crate system tests and on-hardware checks. It may depend on any layer; nothing depends on it.
- `thirdparty/`: pinned third-party sources with their records; its own workspace, never a member.
- `tools/`: internal tools (`wg`, `vkgen`), not tracked.

Rules:
- A feature's editor and pipeline halves are nested role crates (`modules/<feature>/authoring`, `modules/<feature>/pipeline`) and count as `authoring/` and `pipeline/`.
- `apps/player` never reaches `authoring/`, `pipeline/` or a role crate. `apps/editor` and `apps/studio` never depend on each other.
- `unsafe` only in `core/`, `platform/` and in `tests/` crates that drive their unsafe APIs; crates in `servers/` and above use `#![forbid(unsafe_code)]`.
- No tool checks these rules yet; a wg layer check is planned.

Adding a crate:
- Folder `<layer>/<name>/`, package `tessera-<name>`, library `tessera_<name>`. Folder names are unique across layers, except a role crate's: its folder is its role (`modules/<feature>/authoring`), its package `tessera-<feature>-<role>` and its library `tessera_<feature>_<role>`.
- Add the layer's `"<layer>/*"` glob to `members` in `Cargo.toml` with its first crate. A glob over a missing folder, or one matching a sub-folder without a `Cargo.toml`, breaks every Cargo command. Role crates get explicit globs (`"modules/*/authoring"`).
- List the crate once in `[workspace.dependencies]`; dependents write `<package>.workspace = true`.

## Version

See `VERSION`: `MAJOR.MINOR.SPRINT` plus a postfix (`a` alpha, `b` beta, `rc` release candidate, `s` stable) and a build counter.
