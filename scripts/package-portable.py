#!/usr/bin/env python3
"""Build a portable MEMBRANE release archive from a `cmake --install` prefix.

  macOS    membrane_<version>_macos_<arch>.tar.gz
  Windows  membrane_<version>_windows_<arch>.zip

The archive extracts to one directory, membrane_<version>_<os>_<arch>/,
holding only what a user needs: bin/ (membrane, membrane-run and, on Windows,
the DLLs they load), lib/ (macOS shared libraries), LICENSE, third-party
license text, a short README.txt and PACKAGE-INFO.json (version, commit,
OS, architecture, build date, signing status). No build tree, tests, results
or sources are included.

Deterministic: entries are sorted and carry a fixed timestamp
(SOURCE_DATE_EPOCH, else the commit time), root ownership and normalized
modes, so the same tree packs to the same bytes. Never touches git state.

Usage: package-portable.py --os macos|windows --arch ARCH --version V
           --prefix INSTALL_PREFIX --commit SHA --out-dir DIR
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

README = """MEMBRANE {version} ({os_label}, {arch})
Hardware-aware memory intelligence and control for local LLM inference.

This is an UNSIGNED portable build. {signing_note}

Quick start
-----------
{quickstart}

Then:
  membrane doctor
  membrane models available
  membrane use
  membrane chat

MEMBRANE keeps its models and settings under your user profile, not in this
folder; deleting this folder removes the program and leaves your models.
The local server listens on 127.0.0.1 only. Nothing here changes firewall
settings or opens a network port by itself.

Docs: https://github.com/kadireren7/membrane  (docs/install.md)
License: Apache-2.0 (LICENSE). Third-party licenses: licenses/.
Commit: {commit}
"""

QUICK = {
    "macos": """  tar -xzf membrane_{version}_macos_{arch}.tar.gz
  cd membrane_{version}_macos_{arch}
  xattr -dr com.apple.quarantine .      # only if Gatekeeper blocks the unsigned binary
  ./bin/membrane --version
  # optional: put it on your PATH
  export PATH="$PWD/bin:$PATH\"""",
    "windows": """  Expand-Archive membrane_{version}_windows_{arch}.zip -DestinationPath .
  cd membrane_{version}_windows_{arch}
  .\\bin\\membrane.exe --version
  # optional: put it on your PATH for this session
  $env:Path = "$PWD\\bin;$env:Path\"""",
}
SIGNING = {
    "macos": "It is not code-signed or notarized, so macOS Gatekeeper may "
             "warn the first time; see docs/install.md.",
    "windows": "It is not code-signed, so Windows SmartScreen may warn the "
               "first time; see docs/install.md.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--os", required=True, choices=["macos", "windows"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    prefix = Path(a.prefix)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0")) or 315532800
    top = f"membrane_{a.version}_{a.os}_{a.arch}"
    exe = ".exe" if a.os == "windows" else ""
    for need in (f"bin/membrane{exe}", f"bin/membrane-run{exe}"):
        if not (prefix / need).is_file():
            sys.exit(f"package-portable: {need} missing from {prefix}")

    links = {}
    files = {}   # archive path (relative) -> (bytes, is_exec)
    for sub in ("bin", "lib"):
        base = prefix / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_symlink() and a.os == "macos":
                # versioned dylib names (libllama.0.dylib -> libllama.X.dylib)
                links[f"{top}/{p.relative_to(prefix).as_posix()}"] = \
                    os.readlink(p)
                continue
            if p.is_file():
                rel = p.relative_to(prefix).as_posix()
                if p.suffix in (".h", ".hpp", ".a", ".lib", ".cmake", ".pc"):
                    continue
                is_exec = sub == "bin" and a.os == "macos" or \
                    p.suffix in (".exe", ".dll", ".dylib")
                files[f"{top}/{rel}"] = (p.read_bytes(), is_exec)
    lic = REPO / "LICENSE"
    files[f"{top}/LICENSE"] = (lic.read_bytes(), False)
    ll = REPO / "third_party" / "llama.cpp" / "LICENSE"
    if ll.is_file():
        files[f"{top}/licenses/llama.cpp-LICENSE"] = (ll.read_bytes(), False)
    info = {"product": "MEMBRANE", "version": a.version, "os": a.os,
            "arch": a.arch, "commit": a.commit,
            "source_date_epoch": epoch, "signed": False, "notarized": False}
    files[f"{top}/PACKAGE-INFO.json"] = (
        (json.dumps(info, indent=2, sort_keys=True) + "\n").encode(), False)
    quick = QUICK[a.os].format(version=a.version, arch=a.arch)
    files[f"{top}/README.txt"] = (README.format(
        version=a.version, arch=a.arch, commit=a.commit,
        os_label="macOS" if a.os == "macos" else "Windows",
        signing_note=SIGNING[a.os], quickstart=quick).encode(), False)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.os == "windows":
        path = out / f"{top}.zip"
        t = (1980, 1, 1, 0, 0, 0)
        import time
        tm = time.gmtime(max(epoch, 315532800))
        t = (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
             tm.tm_sec - tm.tm_sec % 2)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                zi = zipfile.ZipInfo(name, date_time=t)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                z.writestr(zi, files[name][0])
    else:
        path = out / f"{top}.tar.gz"
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for name in sorted(links):
                ti = tarfile.TarInfo(name)
                ti.type = tarfile.SYMTYPE
                ti.linkname = links[name]
                ti.mtime = epoch
                ti.mode = 0o755
                tf.addfile(ti)
            for name in sorted(files):
                data, is_exec = files[name]
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                ti.mtime = epoch
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o755 if is_exec else 0o644
                tf.addfile(ti, io.BytesIO(data))
        with open(path, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0,
                               filename="") as g:
                g.write(raw.getvalue())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{digest}  {path.name}")


if __name__ == "__main__":
    main()
