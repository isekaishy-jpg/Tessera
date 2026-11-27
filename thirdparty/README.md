# thirdparty

Pinned third-party sources, one folder per library, each with its record: `<lib>/README.md` (source, version or commit, licence, local patches), `<lib>/LICENSE`, `<lib>/patches/` and `<lib>/src/`. Build output is never tracked: `thirdparty/target/` (the recipes' workspace) and `<lib>/build/` (a library's recipe), both denied in `.gitignore`.

- **Depends on:** nothing. When the first library lands, its build recipes form their own Cargo workspace here, never a member of the engine workspace; `platform/*-sys` crates link what it builds.
- **`unsafe`:** not applicable (third-party code).
- **Not here:** our own code.
- **Crates:** none yet.
