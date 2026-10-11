# Upgrading from v1.1.0 to v1.2.0

**Linux:** install the new package over the old one
(`sudo apt install ./membrane_1.2.0_amd64.deb`). Nothing is migrated: the
registry (`~/.local/share/membrane/models.json`), `server.json`
(`~/.config/membrane`) and the audit log keep their locations and are read as
they are.

**macOS and Windows:** v1.1.0 had no official package, so there is nothing to
upgrade from a package. If you built v1.1.0 from source, your registry lives in
the older location (`~/.local/share/membrane`, which on Windows meant
`%USERPROFILE%\.local\share\membrane`). v1.2.0 keeps using that directory when
it exists and the new native directory does not. Nothing is moved, copied or
deleted. New installs use `~/Library/Application Support/membrane` (macOS) and
`%LOCALAPPDATA%\membrane` (Windows). `membrane doctor` prints the exact
directories in its `platform` check. To switch to the native directory, move
the folder yourself and run `membrane doctor`; or set `XDG_DATA_HOME`, which
wins on every OS.

## What behaves differently

| Before | Now |
|---|---|
| checksums of downloaded models were computed by running `sha256sum` (absent on stock macOS and on Windows outside Git Bash, in which case verification was skipped with a notice) | computed by MEMBRANE itself on every OS, always |
| `membrane doctor` had no platform section | a new `platform` check (OS, architecture, data/config/state directories, service manager, evidence label); the JSON gains one check, existing checks are unchanged |
| `doctor` looked for `membrane` on PATH using `:` and no `.exe` | uses `;` and `.exe` on Windows |
| the installed macOS `membrane` binary could not locate its libraries outside the build tree | fixed (`@loader_path`) |
| Windows consoles could print raw ANSI escapes | color is enabled only after the console accepts virtual-terminal processing |

The OpenAI-compatible API v1, every `--json` shape (apart from the added
`platform` doctor check) and `server.json` are unchanged.

## Downgrading

Install the v1.1.0 package (Linux). On macOS and Windows, delete the extracted
folder; models and settings stay where they are.
