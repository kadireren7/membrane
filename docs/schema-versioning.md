# Schema versioning policy

Mega Phase C, PR C2. This project has exactly two pieces of on-disk,
user-owned state with an explicit `schema_version` field:

- `~/.config/membrane/server.json` (`server_config.h`,
  `MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION`, == 1 today)
- `~/.local/share/membrane/models.json` (`registry_core.h`,
  `MEMBRANE_REGISTRY_SCHEMA_VERSION`, == 1 today)

Both use the same fail-closed policy, and until this PR only the config
file actually enforced it — the registry's own loader checked that a
`schema_version` key was *present* but never that its *value* was one
this build understands, so a hypothetical future registry format bump
would have been silently misparsed rather than rejected. Fixed here to
match `server_config.h`'s existing, already-tested behavior exactly.

## The policy

1. **A missing/malformed file is not an error.** No config yet, or no
   registry yet, both resolve to sensible empty defaults — a fresh
   install must never require pre-creating either file.
2. **A file that exists but doesn't parse, or is missing required keys,
   fails closed** with a clear `PARSE_ERROR`/similar code — never
   silently discarded or reinterpreted as empty.
3. **A file with a `schema_version` this build does not recognize fails
   closed** with `UNSUPPORTED_SCHEMA` and a message naming both the
   version found and the version this build understands. This is the
   one case a future incompatible format change is expected to trigger.
4. **No automatic migration exists, and none is planned for v0.4.**
   Both schemas have been at version 1 since they were introduced —
   there has never been a real migration to design against, and writing
   one now would be speculative. If a future phase needs an
   incompatible shape change, the intended path is:
   - Bump the `*_SCHEMA_VERSION` constant so old files are rejected
     with a clear, actionable message rather than misread.
   - Decide, at that time, whether the change is worth a real migration
     path (read-old-write-new) or a documented manual regeneration step
     — both files are cheap, single-user, and trivially reconstructible
     (`server.json` has three meaningful fields with sensible defaults;
     `models.json` entries are re-added in seconds via `membrane model
     add`), so a hard break with a clear upgrade note has, so far, been
     judged an acceptable and honest tradeoff over building migration
     machinery for a version bump that has never actually happened.
5. **A package upgrade never touches either file, or the generated
   systemd unit.** All three are user data/state living outside the
   package's own install tree — `membrane service install` only seeds a
   config if none exists yet, and never rewrites an existing one, on
   any version. See `docs/service.md`'s own "Service upgrade / config
   versioning" section for the config side of this (established PR B4);
   this PR extends the identical guarantee/rationale to the registry.

## Test coverage

- `test_server_config`'s `test_load_unsupported_schema_version_fails_closed`
  (pre-existing, PR B4).
- `test_registry_core`'s `test_load_unsupported_schema_version_fails_closed`
  (added this PR).

Both are real, deterministic unit tests (a hand-written file with
`"schema_version": 999`), no container or real upgrade needed to
exercise this specific guarantee.
