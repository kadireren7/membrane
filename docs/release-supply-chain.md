# Release supply chain

Mega Phase C, PR C2: packaging content, upgrade behavior, reproducible
builds, SBOM, and release provenance/signing policy. See
`docs/schema-versioning.md` for the config/registry migration side of
"upgrade behavior," and `docs/service.md` for the systemd-unit side
(both pre-date this PR and are unchanged by it, other than the registry
now sharing the exact same fail-closed guarantee the config already had).

## Package content audit (real, this PR)

A real `.deb` (CPU-only variant, built from this tree) was inspected
directly with `dpkg -I`/`dpkg -c`/`dpkg-deb -e`:

- Contents: `usr/bin/membrane`, `usr/bin/membrane-run` (both stripped,
  confirmed via `file`), the three real shared libraries
  (`libggml*.so*`, `libllama.so*`) with correct SONAME symlinks,
  `usr/share/membrane/LICENSE.txt`. No headers, no source, no static
  libraries, no docs/markdown, no model files, nothing outside `/usr`.
- `postinst`/`postrm`: seven lines each, `ldconfig` only on
  configure/remove. No service management, no touching of
  `~/.config`/`~/.local/share` from either script -- confirmed by
  reading their real content, not assumed.
- Runtime `Depends:` is entirely `dpkg-shlibdeps`-computed from the
  real linked binaries (`libc6`, `libgcc-s1`, `libgomp1`, `libstdc++6`
  for the CPU variant; the Vulkan variant additionally picks up
  `libvulkan1` the same way) -- never a hand-maintained list that could
  drift from what the binary actually needs.

No content changes were needed -- the existing CPack configuration
(Phase 29) was already minimal and correct.

## Service-upgrade / install-remove data-preservation audit (real, this PR)

A real, memory-bounded (`--memory=512m`) `ubuntu:24.04` container ran
the full lifecycle end to end:

1. `apt-get install ./membrane-cpu_0.3.0_amd64.deb`
2. Real registry (`models.json`) and config (`server.json`) state
   created, SHA256 recorded.
3. `apt-get install --reinstall` (simulates a version upgrade replacing
   the binaries) -- SHA256 of both files confirmed **byte-identical**
   afterward; `membrane model list` still works.
4. `apt-get remove` (not purge) -- `membrane` binary confirmed gone;
   both user-data files confirmed still present, unmodified.
5. Reinstall -- registry/config still there and functional.

This confirms directly, not just by reading `postinst`/`postrm`, that a
package upgrade or removal never touches user data, matching the
policy already documented in `docs/service.md`.

## Reproducible-build investigation (real finding + real fix, this PR)

**Finding**: two back-to-back `cpack -G DEB` runs from an unchanged
build tree produce two *different* `.deb` files. Root cause isolated by
extracting both and comparing: every real file inside (`membrane`,
`membrane-run`, every `.so`, `LICENSE.txt`) is byte-for-byte identical
between the two runs -- the difference is entirely in the `.deb`
container itself. The embedded `gzip` header for `data.tar.gz`/
`control.tar.gz` carries the real wall-clock build timestamp (bytes
4-7 of the gzip header), which differs run to run and changes the
compressed size by a few bytes even though the underlying tar content
is identical.

**Fix**: setting the standard reproducible-builds environment variable
`SOURCE_DATE_EPOCH` before invoking `cpack -G DEB` makes the two runs
produce a **bit-for-bit identical** `.deb` (confirmed directly: same
SHA256). This is handled entirely by `dpkg-deb` itself (>= 1.18.19,
present on this host) -- no CMakeLists.txt change was needed.
`scripts/build-release.sh` sets `SOURCE_DATE_EPOCH` to the release
commit's own `git log -1 --format=%ct` timestamp, so a release built
twice from the same tagged commit is reproducible.

This was investigated, not assumed: reproducibility was verified to
fail without the fix and succeed with it, on this exact host.

## SBOM

`scripts/generate-sbom.py` emits a CycloneDX 1.5-shaped JSON, hand-
written (no equivalent third-party C++/vendored-header SBOM generator
was worth adding) but populated entirely from real, current repo state:
the llama.cpp submodule's actual pinned commit, the real applied patch
files, the real vendored `nlohmann::json`/`cpp-httplib` versions read
directly from their own version macros, `ggml`'s own version, and (when
given `--deb`) the real `Depends:` line a specific built package
produced. `scripts/build-release.sh` always generates one SBOM per
built `.deb`, from that exact `.deb`.

## Release provenance

`scripts/build-release.sh` produces, per invocation:
- one `.deb` per requested backend (`cpu`, `vulkan`, or `both`)
- one `<package>.sbom.json` per `.deb`
- one `SHA256SUMS` covering every `.deb` in the output directory

It prints the exact commit (`git rev-parse HEAD`) and the
`SOURCE_DATE_EPOCH` used. It never touches git, never creates a tag,
never pushes, and never calls `gh release create` -- artifact
*building* and release *publishing* are deliberately kept as two
separate steps, the latter always a manual, explicit, primary-agent-
performed action (see the git-safety rules this whole mega-phase
operates under).

Release asset hashes/SBOMs are not required to live in the tagged
source tree itself -- they ship as `SHA256SUMS`/GitHub release assets
(or a later evidence-only PR), matching this project's own established
convention.

## Signed-tag policy

This project **never auto-generates a signing key**. Before tagging
any release, the build host is checked for a real, already-configured
signing identity (`git config --get user.signingkey`, `gpg
--list-secret-keys`, `git config --get tag.gpgSign`). As of this PR, no
such identity is configured on this project's own dev host -- so
release tags are created as real **annotated but unsigned** tags, and
this fact is disclosed plainly in the release notes rather than
silently omitted or faked with an ad hoc generated key. If a real
signing identity is ever configured on a release host, the same
checklist should be re-run and tags switched to `git tag -s`.

## CI release-candidate smoke

A `release-candidate-smoke` CI job runs `scripts/build-release.sh
--backend cpu --out-dir dist` on every push/PR -- proving the release
automation script itself keeps working (real build, real SBOM, real
checksums) without ever calling `gh release create` or otherwise
publishing anything. It is a smoke check on the *mechanism*, not a
real release -- an actual v0.4.0 release build (potentially including
the Vulkan variant) happens later, manually, at C4's tagging step.
