#!/usr/bin/env bash
# Thin convenience wrapper around the documented build/install commands
# (docs/install.md, Option C by default) -- configures, builds, and
# installs membrane-run. Does NOT: download or curl anything, run sudo
# on your behalf, edit shell profile files, download models, or change
# GPU drivers. Every command it runs is echoed before it runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$HOME/.local"
BUILD_DIR="$REPO_ROOT/build-install"
# Phase 30: deliberately NOT nproc -- llama.cpp/ggml's largest
# translation units are memory-hungry enough that unbounded parallelism
# on a CPU-count-high/RAM-low host (confirmed directly: this project's
# own CI runners, 2 vCPU/~7 GiB, OOM-kill at unbounded -j; a real local
# dev machine with 12 CPUs but only ~5.6 GiB RAM hits the same failure
# mode) can OOM-kill the build outright. 2 is the same conservative
# default this project's own CI already uses everywhere it compiles
# llama.cpp/ggml. Pass --jobs N yourself if your machine has RAM to
# spare.
JOBS=2
VULKAN=0

usage() {
	cat <<'EOF'
Usage: scripts/install.sh [--prefix DIR] [--vulkan] [--jobs N] [--build-dir DIR]

Configures, builds, and installs membrane-run via the same CMake steps
docs/install.md documents by hand -- this script adds no behavior of
its own beyond running them for you.

  --prefix DIR      install prefix (default: $HOME/.local -- no root
                     needed; pass e.g. --prefix /usr/local yourself if
                     you want a system install, in which case YOU run
                     this script under sudo -- it never elevates
                     itself)
  --vulkan          build with Vulkan GPU offload support (needs the
                     Vulkan development headers/glslc/SPIR-V headers
                     already installed -- see docs/install.md's Option
                     B prerequisites; NOT auto-detected, since silently
                     changing build flags based on a guess about your
                     system is exactly the kind of hidden behavior this
                     script avoids)
  --jobs N           parallel build jobs (default: 2 -- deliberately
                     conservative, not nproc; llama.cpp/ggml's largest
                     translation units can OOM a build on a high-CPU-
                     count/low-RAM host at unbounded parallelism.
                     Raise it yourself if your machine has RAM to
                     spare)
  --build-dir DIR    CMake build directory (default: build-install)
  -h, --help         this message

After installing, add the prefix's bin/ directory to PATH yourself if
it isn't already (this script never edits shell profile files):
  export PATH="<prefix>/bin:$PATH"
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--prefix) PREFIX="$2"; shift 2 ;;
	--vulkan) VULKAN=1; shift ;;
	--jobs) JOBS="$2"; shift 2 ;;
	--build-dir) BUILD_DIR="$2"; shift 2 ;;
	-h | --help) usage; exit 0 ;;
	*) echo "install.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

if [ ! -f "$REPO_ROOT/third_party/llama.cpp/CMakeLists.txt" ]; then
	echo "install.sh: third_party/llama.cpp submodule missing." >&2
	echo "Run this first (no other network access happens here):" >&2
	echo "  git -C '$REPO_ROOT' submodule update --init third_party/llama.cpp" >&2
	exit 1
fi

CMAKE_ARGS=(-S "$REPO_ROOT" -B "$BUILD_DIR" -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release)
if [ "$VULKAN" = "1" ]; then
	CMAKE_ARGS+=(-DGGML_VULKAN=ON)
fi

set -x
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$JOBS" --target membrane-run
cmake --install "$BUILD_DIR" --prefix "$PREFIX"
set +x

echo
echo "Installed to: $PREFIX/bin/membrane-run"
case ":$PATH:" in
*":$PREFIX/bin:"*) ;;
*)
	echo "NOTE: $PREFIX/bin is not on your PATH yet. Add it yourself, e.g.:"
	echo "  export PATH=\"$PREFIX/bin:\$PATH\""
	;;
esac
echo "Verify: $PREFIX/bin/membrane-run --doctor"
echo "Uninstall: cmake --build '$BUILD_DIR' --target uninstall"
