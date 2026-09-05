# Install

This covers getting from zero to an installed `membrane-run` binary, and
removing it again. For CLI flag semantics see `membrane-run --help` and
the README; for the KV precision/placement mechanism see
`docs/live-runtime.md` and `docs/kv-residency.md`.

Linux only — this has not been verified on macOS or Windows.
Debian-family only (Ubuntu / Pop!_OS / Debian) for the `.deb` path below
— no Fedora/RPM, Arch, Flatpak, Snap, or AppImage packaging exists.

Three ways to get `membrane-run` installed, easiest first:

- **Option A** — install a `.deb` package (Ubuntu/Debian/Pop!_OS, no
  manual CMake).
- **Option B** — build from source and `cmake --install` to a system
  prefix.
- **Option C** — build from source and install to a user-local prefix,
  no root needed.

All three produce the same binary behavior; they differ only in how the
files get onto disk (and, for Option C, whether root is needed at all).

## Option A — install a `.deb` package

**Package policy** (decided Phase 30, `results/release-artifacts/manifest.json`):
the ONE official package is `membrane_<version>_amd64.deb` — a
Vulkan-enabled build. That may look surprising for a "CPU-first" tool,
but it is the simplest honest choice: this exact build runs correctly
CPU-only, handles a host with `libvulkan1` present but zero visible GPU
devices cleanly (`--doctor` reports it accurately, `--list-devices`
shows CPU only, generation still works), and never touches a GPU unless
you ask for one (`--auto` or an explicit `--gpu-layers`/`--kv-placement`
flag) — verified with real container tests, see
`results/distribution/validation.json` and
`results/first-run/validation.json`. Its only extra cost over a
CPU-only build is one additional real runtime dependency
(`libvulkan1`, resolved automatically by `apt`). A separate
`membrane-cpu_<version>_amd64.deb` also exists (built and validated
every CI run) but is a CI-validation/build-your-own artifact, not the
official release asset — use it only if you specifically want to
avoid the Vulkan/`libvulkan1` dependency.

Release packages are attached to this project's own
[GitHub releases](https://github.com/kadireren7/membrane/releases)
(each with a `SHA256SUMS` file) — download the latest, or build one
yourself with this project's own build pipeline (see "Building a
`.deb` package yourself" below). External multi-host validation beyond
the maintainer's own development hardware is still limited, disclosed
in the README rather than hidden. Once you have a
`membrane_<version>_amd64.deb` file (downloaded, or built yourself):

```bash
sudo apt install ./membrane_0.4.0_amd64.deb
```

(`apt install ./file.deb`, not `dpkg -i`, so apt resolves the package's
declared runtime dependencies — `libvulkan1` on the official package,
nothing beyond the standard C/C++/OpenMP runtime on `membrane-cpu` —
instead of leaving them "half-installed" for you to fix by hand.)

Verify:

```bash
membrane-run --version
membrane-run --doctor
membrane-run --list-devices
```

Inspect a model before running it (no generation, cheap):

```bash
membrane-run --model model.gguf --inspect-model
```

Uninstall:

```bash
sudo apt remove membrane
```

(`sudo apt remove membrane-cpu` for that variant) removes exactly the
files the package installed (`/usr/bin/membrane-run`, its bundled
`libllama`/`libggml*` shared libraries, and
`/usr/share/membrane/LICENSE.txt`) — nothing else. It never touches your
model files, config, or unrelated system libraries. The two packages
use different `Package:` names specifically so they can never silently
overwrite or shadow each other.

### Building a `.deb` package yourself

Needs `dpkg-dev` in addition to this project's normal build
prerequisites (below):

```bash
sudo apt install dpkg-dev
```

Official (Vulkan-enabled) package — needs the same Vulkan development
files as Option B/C's Vulkan build below:

```bash
cmake -S . -B build-package \
  -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-package -j2 --target package
```

produces `membrane_<version>_amd64.deb`. CPU-only variant (no Vulkan
development files needed):

```bash
cmake -S . -B build-package-cpu -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-package-cpu -j2 --target package
```

produces `membrane-cpu_<version>_amd64.deb`. Same underlying CPack
definition either way — which package name/filename you get is purely
the build-time `-DGGML_VULKAN=ON` choice (`CMakeLists.txt`), reflected
honestly in each package's own declared dependencies
(`dpkg-deb -I membrane_*.deb` shows `libvulkan1` only on the official
build).

The `--target package` build compiles every target in the tree (CMake's
own default: the `package` target depends on `all`), not just
`membrane-run` — simple and matches the commands above exactly, but
slower than it needs to be. If you already have `membrane-run` built
(e.g. from Option B below) and just want to re-package it without
rebuilding everything else, run `cpack -G DEB` directly from that build
directory instead — it only re-runs the install step and packaging, not
a build.

Version mapping: `MEMBRANE_VERSION` (`tools/membrane-run/product_cli.h`)
is the one source of truth `membrane-run --version` and the package
version both derive from — never hand-edited in more than that one
place. A prerelease like `0.3.0-rc3` is mapped to `0.3.0~rc3` for the
package's own `Version:` field only (`membrane-run --version` still
prints the unmapped `0.3.0-rc3`): Debian's version-ordering rules treat
a bare hyphen as the upstream/revision separator and would otherwise
sort `0.3.0-rc3` as newer than the real `0.3.0` release, making an
`apt upgrade` from the RC to the final release look like a downgrade
and get refused. `~` is the documented Debian convention for exactly
this (Debian Policy §5.6.12) — it sorts before everything, including no
suffix at all, so `0.3.0~rc3` correctly orders before `0.3.0`. A stable
version with no hyphen (e.g. `0.2.0`) passes through unchanged.

## Option B — build from source, system install

### Prerequisites

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

### Clone and initialize the submodule

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

### Build

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

### Install

```bash
sudo cmake --install build-vulkan
```

(with no `--prefix`, the default is `/usr/local`) installs:

- `<prefix>/bin/membrane-run`
- `<prefix>/<libdir>/libllama.so*`, `libggml*.so*` (`<libdir>` is
  `lib` on most distributions, but GNUInstallDirs can resolve it to
  `lib64` on some — check `install_manifest.txt`, described below, for
  the exact paths on your system). `membrane_core` is always linked
  statically into `membrane-run` and is never part of this list — see
  "Why install shared libraries" below.
- `<prefix>/share/membrane/LICENSE.txt`

### Verifying the install

```bash
membrane-run --help
```

run from any directory, not just the repo. If this is the first time
you've installed to this prefix, open a new shell (or re-source your
profile) so `PATH` picks up the change.

`membrane-run --doctor` runs a handful of cheap, non-destructive checks
(GPU backend/device visibility, host RAM detection) and prints [OK]/
[WARN] for each — useful right after install to confirm the binary sees
what you expect before your first real run. `membrane-run --list-devices`
lists every backend device it can see (no model needed).

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

### Uninstall

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

## Option C — build from source, user-local install (no root)

Same Prerequisites/Clone/Build steps as Option B above, then install to
a prefix inside your own home directory instead of a system path — no
`sudo` anywhere in this path:

```bash
cmake --install build-vulkan --prefix "$HOME/.local"
```

Make sure `~/.local/bin` is on your `PATH` (most distributions already
add it for you; if not, add `export PATH="$HOME/.local/bin:$PATH"` to
your shell profile yourself — nothing in this project edits shell
profile files on your behalf). Verifying and uninstalling work exactly
as described in Option B above, just without `sudo`:

```bash
membrane-run --help
cmake --build build-vulkan --target uninstall
```

## Troubleshooting

- **`llama.cpp submodule missing; run: git submodule update --init
  third_party/llama.cpp`** — exactly what it says; the submodule
  wasn't cloned.
- **`Vulkan backend requested (-DGGML_VULKAN=ON) but Vulkan development
  files were not found`** — install the packages listed under Option
  B's Prerequisites, or drop `-DGGML_VULKAN=ON` for a CPU-only build.
- **`error while loading shared libraries` after installing** — this
  was a real bug found and fixed during Phase 17 packaging work (the
  Vulkan backend's shared library wasn't in the install list). If you
  hit a variant of this, please file an issue — it means some
  dependency still isn't covered by `tools/membrane-run/CMakeLists.txt`'s
  install rules.
- **Configure re-applies or complains about a patch** — see "Clone and
  initialize the submodule" above; the submodule is in an unexpected
  state and needs a clean checkout first.
- **`cpack`/`cmake --build --target package` fails or warns about
  `dpkg-shlibdeps`** — install `dpkg-dev` (`sudo apt install
  dpkg-dev`); it provides `dpkg-shlibdeps`, which CPack uses to detect
  the package's real runtime dependencies from the actual built binary
  rather than a hand-maintained guess.
- **`apt install ./membrane_*.deb` refuses with a dependency error**
  — this means your system genuinely doesn't have a required runtime
  library (e.g. no `libvulkan1` available for a Vulkan-enabled
  package on a non-Debian-family system) — install it via your
  distribution's normal package manager, or use the CPU-only package/
  build instead.
