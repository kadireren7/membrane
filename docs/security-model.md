# Security model

What MEMBRANE does and does not protect, in one page. (Vulnerability reports:
[../SECURITY.md](../SECURITY.md).)

## The boundary

* **Loopback only.** `membrane serve` binds `127.0.0.1` by default. Binding
  elsewhere needs an explicit `--allow-non-loopback`, which warns loudly.
* **No authentication.** The inference API and the admin routes have none; the
  loopback boundary is the protection. Anyone who can reach the port can use
  the model and change what is loaded.
* **No CORS.** No `Access-Control-*` headers are sent, so other web origins
  cannot read responses. No wildcard origin is, or may be, added. A browser
  integration would need an origin allowlist **and** a token, explicitly
  opted into.
* **No cloud, no credentials.** MEMBRANE never sends prompts or data to a
  remote service and stores no secrets. Downloads are HTTPS-only with SHA-256
  verification against the catalog.

## Admin (mutating) surfaces

The loopback-only `/membrane/v1/...` routes (activate, pin, unpin, unload;
read-only adaptive status) are not part of the OpenAI surface. Mutations are
reached through `membrane apply`, which is **dry-run by default**, needs
`--yes` or a terminal confirmation, re-checks live state before changing
anything, rolls back on failure and writes a local audit log. `apply` refuses
to act on a non-loopback `listen_address` or a non-loopback Ollama endpoint.

## External runtimes

Ollama and vLLM are contacted only when a command asks, on loopback by
default, with GET-only allowlists (plus Ollama's single documented
`keep_alive=0` unload used by an explicit `membrane apply --unload`). Cloud
model references are refused.

## Adaptive control

Off by default. `recommend` never mutates. `apply` mode needs explicit
consent, acts only within the classes you allowed, and has no mutation path of
its own (it uses `membrane apply`'s pipeline). It records no prompts or
generated text -- the controller never sees them.

## What is logged

`~/.local/state/membrane/actions.jsonl` (mode 0600, rotated at 1 MiB) records
applied changes: time, runtime, action, before/after, outcome, reason codes.
Never prompt text, generated content or secrets.
