#!/usr/bin/env bash
#
# scripts/build-release.sh -- builds real, reproducible MEMBRANE release
# artifacts (Mega Phase C, PR C2). Never touches git in any way: no tag,
# no push, no `gh release create`. Tagging and publishing a real GitHub
# release stay a separate, explicit, primary-agent-performed step (see
# docs/release-supply-chain.md) -- this script's only job is to turn an
# already-committed source tree into a set of .deb + SHA256SUMS + SBOM
# files on disk.
#
# What it does, per requested backend:
#   1. Configure + build (Release) into its own build directory
#      (build-release-cpu / build-release-vulkan -- never reuses/pollutes
#      a developer's existing build-* directories).
#   2. Run `cpack -G DEB` with SOURCE_DATE_EPOCH pinned to the current
#      HEAD commit's own author timestamp -- a real, verified fix (this
#      PR's own reproducible-build investigation): dpkg-deb >= 1.18.19
#      honors this standard reproducible-builds variable and normalizes
#      every embedded timestamp to it, making two builds from the same
#      commit byte-for-byte identical (confirmed directly: without it,
#      two consecutive `cpack -G DEB` runs from an unchanged tree produce
#      different gzip headers and therefore different .deb files).
#   3. Copy the resulting .deb into --out-dir.
# Then, once per requested backend's own real .deb:
#   4. scripts/generate-sbom.py --deb <path> -- the real Depends: line
#      this exact build produced, not a hand-maintained guess.
# Finally: a SHA256SUMS file over every .deb in --out-dir.
#
# Usage: scripts/build-release.sh [--backend cpu|vulkan|both] [--out-dir DIR]
#   --backend   which variant(s) to build (default: cpu -- the cheap,
#               always-available-in-CI variant; pass vulkan or both for
#               a real release build on a host with the Vulkan SDK/
#               loader installed, e.g. at actual v0.4.0 tagging time).
#   --out-dir   where to place the final artifacts (default: dist/).
#   --jobs      parallel build jobs (default: 2 -- this project's dev
#               host is real-memory-constrained; override on a more
#               capable machine, e.g. --jobs "$(nproc)").
#
# Exit code: 0 on success, non-zero (with the failing step named) on any
# configure/build/cpack/SBOM failure.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BACKEND="cpu"
OUT_DIR="$REPO_ROOT/dist"
JOBS=2
while [ $# -gt 0 ]; do
	case "$1" in
	--backend)
		BACKEND="$2"
		shift 2
		;;
	--out-dir)
		OUT_DIR="$2"
		shift 2
		;;
	--jobs)
		JOBS="$2"
		shift 2
		;;
	*)
		echo "build-release.sh: unknown argument: $1" >&2
		exit 2
		;;
	esac
done

case "$BACKEND" in
cpu | vulkan | both) ;;
*)
	echo "build-release.sh: --backend must be cpu, vulkan, or both (got: $BACKEND)" >&2
	exit 2
	;;
esac

mkdir -p "$OUT_DIR"

SOURCE_DATE_EPOCH="$(git log -1 --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
GIT_COMMIT="$(git rev-parse HEAD)"
echo "build-release.sh: HEAD=$GIT_COMMIT SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"

DEB_PATHS=()

build_one() {
	local variant="$1"
	local vulkan_flag="$2"
	local build_dir="$REPO_ROOT/build-release-$variant"

	echo
	echo "== building $variant variant in $build_dir =="
	if ! cmake -S "$REPO_ROOT" -B "$build_dir" \
		-DCMAKE_BUILD_TYPE=Release \
		-DMEMBRANE_ENABLE_LLAMA=ON \
		-DGGML_VULKAN="$vulkan_flag" \
		>"/tmp/build-release-$variant-configure.log" 2>&1; then
		echo "build-release.sh: configure failed for $variant, see /tmp/build-release-$variant-configure.log" >&2
		return 1
	fi
	if ! cmake --build "$build_dir" -j"$JOBS" \
		>"/tmp/build-release-$variant-build.log" 2>&1; then
		echo "build-release.sh: build failed for $variant, see /tmp/build-release-$variant-build.log" >&2
		return 1
	fi
	(
		cd "$build_dir" || exit 1
		rm -f ./*.deb
		cpack -G DEB
	) >"/tmp/build-release-$variant-cpack.log" 2>&1
	local deb
	deb="$(find "$build_dir" -maxdepth 1 -name "*.deb" | head -1)"
	if [ -z "$deb" ]; then
		echo "build-release.sh: cpack produced no .deb for $variant, see /tmp/build-release-$variant-cpack.log" >&2
		return 1
	fi
	# Sanity check (a real bug this PR's own testing found: an
	# unconfigured -DMEMBRANE_ENABLE_LLAMA silently produces a "valid"
	# but near-empty .deb with no membrane/membrane-run binaries at
	# all) -- never ship a package that doesn't actually contain the
	# two real executables it claims to provide.
	local listing
	listing="$(dpkg-deb -c "$deb")"
	if ! grep -q "usr/bin/membrane-run" <<<"$listing" \
		|| ! grep -q "usr/bin/membrane$" <<<"$listing"; then
		echo "build-release.sh: $deb is missing membrane/membrane-run -- refusing to publish an empty package" >&2
		return 1
	fi
	cp "$deb" "$OUT_DIR/"
	local copied="$OUT_DIR/$(basename "$deb")"
	echo "build-release.sh: $variant -> $copied"
	DEB_PATHS+=("$copied")
}

FAILED=0
if [ "$BACKEND" = "cpu" ] || [ "$BACKEND" = "both" ]; then
	build_one "cpu" "OFF" || FAILED=1
fi
if [ "$BACKEND" = "vulkan" ] || [ "$BACKEND" = "both" ]; then
	build_one "vulkan" "ON" || FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then
	echo "build-release.sh: one or more variants failed to build" >&2
	exit 1
fi

echo
echo "== generating SBOMs =="
for deb in "${DEB_PATHS[@]}"; do
	sbom_path="${deb%.deb}.sbom.json"
	if ! python3 "$SCRIPT_DIR/generate-sbom.py" --deb "$deb" --out "$sbom_path"; then
		echo "build-release.sh: SBOM generation failed for $deb" >&2
		exit 1
	fi
done

echo
echo "== SHA256SUMS =="
(
	cd "$OUT_DIR" || exit 1
	sha256sum ./*.deb >SHA256SUMS
	cat SHA256SUMS
)

echo
echo "build-release.sh: done. Artifacts in $OUT_DIR (commit $GIT_COMMIT)."
