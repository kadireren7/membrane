# Support

MEMBRANE is a single-maintainer research project. There is no formal
support SLA — support is best-effort.

## Before asking

1. Check [README.md](README.md) — especially
   [Technical limitations](README.md#technical-limitations), which lists
   what this project explicitly does and does not claim.
2. Check [docs/reproduction.md](docs/reproduction.md) if you're trying to
   reproduce a specific result — it lists working directory,
   dependencies, expected time/output, and success signal per level.
3. Check the relevant `docs/phase*.md` document for the specific result
   or component you have a question about; each documents its own
   methodology and disclosed limitations in detail.
4. Run `scripts/verify-results.py` if you suspect a number in the docs
   doesn't match the underlying artifact — it will tell you exactly
   which check failed and why.

## Getting help

- **Bugs / reproducibility problems**: open a GitHub issue with the
  exact command you ran, the expected vs. actual output, and your
  environment (`uname -a`, compiler version — `scripts/demo.sh` writes
  this automatically to `demo-output/demo-results.json` if you're
  reporting a demo failure).
- **Questions about a specific finding**: open a GitHub issue referencing
  the specific `docs/phase*.md` section — that document is the source of
  truth, and general questions are easier to answer with a concrete
  section to anchor to.
- **Security-relevant issues**: see [SECURITY.md](SECURITY.md) instead —
  do not open a public issue for those.

## What is out of scope for support

- Requests to run the full research reproduction (Level 3 in
  [docs/reproduction.md](docs/reproduction.md)) on someone else's behalf
  — it is multi-hour and resource-intensive; the point of the checkpoint/
  resume design and the committed artifacts is that you can run it
  yourself, or trust the committed, SHA-256-verified results without
  re-running it.
- Requests to extend claims beyond what's measured (e.g., real CXL
  hardware numbers) — see
  [README.md's Technical limitations](README.md#technical-limitations).
