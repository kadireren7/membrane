# Validating MEMBRANE v0.3.0-rc2

Thank you for trying `v0.3.0-rc2` before it becomes a stable release.
This should take about 10–15 minutes. Everything below is copy-paste
friendly. **Nothing here uploads anything automatically** — at the end
you paste a small filled-in report into a GitHub issue or PR yourself,
by hand.

You'll need: a C11/C++17 compiler, CMake ≥ 3.16, `git`. A GPU/Vulkan
SDK is optional — the CPU path alone is a complete, useful report.

## Current status (environment matrix)

Updated as real reports land in `results/rc2-user-validation/summary.json`
— never fabricated ahead of an actual report.

| ID | OS / backend | Status |
|---|---|---|
| ENV-01-cpu | Pop!_OS 24.04, CPU-only (AMD Ryzen 5 5600H) | VALIDATED |
| ENV-01-vulkan-nvidia | Pop!_OS 24.04, Vulkan / NVIDIA GTX 1650 | VALIDATED |
| ENV-01-vulkan-amd | Pop!_OS 24.04, Vulkan / AMD Radeon (RADV, integrated) | VALIDATED |
| ENV-02 | different distro, CPU or Vulkan | NOT_YET_VALIDATED |
| ENV-03 | different distro, Vulkan | NOT_YET_VALIDATED |
| ENV-04 | Intel Vulkan | NOT_YET_VALIDATED |
| ENV-05 | non-Linux (macOS/Windows) | NOT_YET_VALIDATED — this project has never claimed non-Linux support |

**Note:** all three VALIDATED rows above are the same physical host
(one developer's machine) exercising three different backends, not
three independent machines/distros/users — real GPU-vendor diversity
(NVIDIA + AMD), but not yet OS/distro/user diversity. Independent
external reports on a genuinely different machine are still the goal
of this document.

## 1. Clone the RC2 tag

```bash
git clone --recursive --branch v0.3.0-rc2 https://github.com/kadireren7/membrane
cd membrane
git rev-parse HEAD   # save this — you'll need it for the report
```

If you already have a clone, check out the tag explicitly instead of
`main`:

```bash
git fetch --tags
git checkout v0.3.0-rc2
git submodule update --init --recursive
```

## 2. CPU build and install

```bash
cmake -S . -B build -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target membrane-run
cmake --install build --prefix ./install-cpu
```

## 3. Vulkan build and install (skip if you have no GPU/Vulkan SDK)

Install prerequisites first (package names vary by distro; this is
confirmed for Ubuntu/Pop!_OS):

```bash
sudo apt install libvulkan-dev glslc vulkan-tools spirv-headers
```

Then:

```bash
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j2 --target membrane-run
cmake --install build-vulkan --prefix ./install-vulkan
```

## 4. Version and help, outside the build tree

```bash
cd /tmp   # anywhere outside the membrane/ directory
/path/to/install-cpu/bin/membrane-run --version
/path/to/install-cpu/bin/membrane-run --help | head -5
```

Both should print without error. `--version` should say
`MEMBRANE 0.3.0-rc2`.

## 5. Plan-only (no generation, no model weights touched beyond loading)

Get any small GGUF model (a few hundred MB, e.g. a "1B" or smaller
instruct model). Then:

```bash
/path/to/install-cpu/bin/membrane-run \
  --model /path/to/your-model.gguf \
  --prompt "Hi" --ctx 256 --auto --plan-only --json
```

This should print exactly one JSON object and exit 0.

## 6. One small real generation

CPU:

```bash
/path/to/install-cpu/bin/membrane-run \
  --model /path/to/your-model.gguf \
  --prompt "The capital of France is" --ctx 256 --gen-tokens 16 --auto --json --quiet
```

Vulkan, if you built it (add `--device NAME` if you have more than one
GPU and want to pick a specific one — omit it to let MEMBRANE choose):

```bash
/path/to/install-vulkan/bin/membrane-run \
  --model /path/to/your-model.gguf \
  --prompt "The capital of France is" --ctx 256 --gen-tokens 16 --auto --json --quiet
```

**Please don't run large contexts or long generations for this** — a
few hundred tokens of context and 16–32 generated tokens is enough to
validate the path.

## 7. Uninstall

```bash
cd /path/to/membrane
xargs rm -f < build/install_manifest.txt
xargs rm -f < build-vulkan/install_manifest.txt   # if you built Vulkan
```

Confirm nothing is left under your `install-cpu`/`install-vulkan`
prefixes.

## 8. Report your result

Copy `results/rc2-user-validation/schema.json`'s shape (or just look
at the worked example in `results/rc2-user-validation/summary.json`)
and fill in one entry per backend you tested. **Before submitting,
check the privacy list below.**

Open a GitHub issue on `kadireren7/membrane` titled `RC2 validation:
<your OS>` with your filled-in JSON entry (or a PR adding it directly
to `results/rc2-user-validation/summary.json` — either is fine).

### Privacy — please sanitize before submitting

Do **not** include:

- your username or hostname
- your home directory path (use `~` or `/path/to/...` in the
  `command` field)
- your machine's serial number or any hardware ID
- your IP address
- the real path to any private/local model file (just the filename)
- any environment secrets

If something failed and you want to share a log excerpt, please
re-read it once for any of the above before pasting it.

## What counts as a release blocker vs. not

If something in the sections above **crashes, corrupts JSON output,
silently does something different from what you asked for (e.g.
silently uses a different KV precision or placement than you
requested), or reproducibly fails install/build on a normal, supported
setup** — that's exactly the kind of thing we need to know before
shipping stable `v0.3.0`. Please report it even if you're not sure
it's a real bug.

Things that are useful to know but won't block the release by
themselves: unclear wording, a distro/toolchain combination we haven't
tested yet, hardware we haven't validated yet (e.g. AMD/Intel GPUs are
still light on real-world coverage).

Thank you.
