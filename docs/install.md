# Install

This covers getting from a fresh clone to an installed `membrane-run`
binary, and removing it again. For CLI flag semantics see `membrane-run
--help` and the README; for the KV precision/placement mechanism see
`docs/live-runtime.md` and `docs/kv-residency.md`.

Linux only — this has not been verified on macOS or Windows.

Every step below (CPU and Vulkan configure/build/install/run-outside-
the-build-tree/uninstall) is exercised on every push and PR by
`.github/workflows/ci.yml`'s `packaging-smoke` job — a continuously
re-verified check, rather than a point-in-time log that could go stale
as the codebase changes.

## Prerequisites

- A C11/C++17 compiler (GCC or Clang) and CMake ≥ 3.16.
- `git`, including submodule support.
- For a Vulkan build only: Vulkan development headers, the `glslc`
  shader compiler, and SPIR-V headers already installed. On Ubuntu/
  Pop!_OS (24.04, the only distribution this was verified on):

  ```bash
  sudo apt install libvulkan-dev glslc vulkan-tools spirv-headers
  ```

  A CPU-only build needs none of this — no Vulkan SDK, no GPU, no
  GPU driver.

## Clone and initialize the submodule

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane
```

If you already cloned without `--recursive`:

```bash
git submodule update --init third_party/llama.cpp
```

`third_party/llama.cpp` carries two small MEMBRANE patches (KV type
override, KV device override — see `docs/live-runtime.md` and
`docs/kv-residency.md`). These are applied automatically, idempotently,
the first time you configure with `-DMEMBRANE_ENABLE_LLAMA=ON` — CMake
checks each patch's marker in `third_party/llama.cpp/include/llama.h`
before applying it, so reconfiguring never double-applies, and it fails
loudly (`FATAL_ERROR`) rather than silently if `git apply` itself
fails. This only ever adds to the submodule's working tree — it never
runs `git reset`, `git checkout`, or `git submodule update` on your
behalf, and it never touches the submodule's own commit pin. If you see
an unexpected patch-related configure error, the submodule is very
likely already dirty in some other way (e.g. a manual edit); restore it
with `git -C third_party/llama.cpp checkout -- .` before reconfiguring.

## Build

CPU-only:

```bash
cmake -S . -B build -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target membrane-run
```

Vulkan:

```bash
cmake -S . -B build-vulkan \
  -DMEMBRANE_ENABLE_LLAMA=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target membrane-run
```

If `-DGGML_VULKAN=ON` is given without the Vulkan development files
present, configure fails immediately with `Vulkan backend requested
(-DGGML_VULKAN=ON) but Vulkan development files were not found.` —
before descending into `third_party/llama.cpp`'s own build, not as a
compiler error partway through the build.

The resulting binary also runs directly from the build tree without
installing anything:

```bash
./build-vulkan/tools/membrane-run/membrane-run --help
```

## Install

```bash
cmake --install build-vulkan --prefix "$HOME/.local"
```

or system-wide:

```bash
sudo cmake --install build-vulkan
```

(with no `--prefix`, the default is `/usr/local`). Either way this
installs:

- `<prefix>/bin/membrane-run`
- `<prefix>/<libdir>/libllama.so*`, `libggml*.so*` (`<libdir>` is
  `lib` on most distributions, but GNUInstallDirs can resolve it to
  `lib64` on some — check `install_manifest.txt`, described below, for
  the exact paths on your system). `membrane_core` is always linked
  statically into `membrane-run` and is never part of this list — see
  "Why install shared libraries" below.
- `<prefix>/share/membrane/LICENSE.txt`

If you used `--prefix "$HOME/.local"`, make sure `~/.local/bin` is on
your `PATH` (most distributions already add it for you; if not, add
`export PATH="$HOME/.local/bin:$PATH"` to your shell profile).

### Verifying the install

```bash
membrane-run --help
```

run from any directory, not just the repo. If this is the first time
you've installed to this prefix, open a new shell (or re-source your
profile) so `PATH` picks up the change.

CMake also writes `install_manifest.txt` to the build directory
(`build-vulkan/install_manifest.txt`) listing every file the install
step wrote — useful if you want to inspect the install footprint
directly rather than trusting a summary.

### Why install shared libraries

`llama.cpp`/`ggml` default to building as shared libraries
(`BUILD_SHARED_LIBS=ON`) even without passing that flag explicitly, so
`membrane-run` is dynamically linked against them by default. An
install step that only copied the executable would produce a binary
that fails immediately outside the build tree with `error while
loading shared libraries: ... cannot open shared object file`. Installed
libraries carry an `$ORIGIN`-relative RPATH so the installed binary
finds them at their installed location — no `LD_LIBRARY_PATH` needed,
and no dependency on the build directory continuing to exist.

## Uninstall

```bash
cmake --build build-vulkan --target uninstall
```

Removes exactly the files recorded in that build directory's own
`install_manifest.txt` — nothing else. It never deletes the prefix
directory itself (`bin/`, `lib/`, `share/membrane/` are left in place,
even if now empty), since a shared prefix like `~/.local` or
`/usr/local` may hold files from other software. Running it again
after everything is already removed is safe (reports what was already
absent, doesn't error). Running it against a build directory that was
never installed (no `install_manifest.txt` yet) fails clearly instead
of guessing.

## Troubleshooting

- **`llama.cpp submodule missing; run: git submodule update --init
  third_party/llama.cpp`** — exactly what it says; the submodule
  wasn't cloned.
- **`Vulkan backend requested (-DGGML_VULKAN=ON) but Vulkan development
  files were not found`** — install the packages listed under
  Prerequisites, or drop `-DGGML_VULKAN=ON` for a CPU-only build.
- **`error while loading shared libraries` after installing** — this
  was a real bug found and fixed during Phase 17 packaging work (the
  Vulkan backend's shared library wasn't in the install list). If you
  hit a variant of this, please file an issue — it means some
  dependency still isn't covered by `tools/membrane-run/CMakeLists.txt`'s
  install rules.
- **Configure re-applies or complains about a patch** — see "Clone and
  initialize the submodule" above; the submodule is in an unexpected
  state and needs a clean checkout first.
