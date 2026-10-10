# Integrating an application

MEMBRANE's application interface is deliberately **generic**: the
OpenAI-compatible `POST /v1/chat/completions` (and `GET /v1/models`) on
`http://127.0.0.1:8642`. There are no application-specific endpoints and there
will not be: a translator, a coding assistant and a chatbot all call the same
route with a different prompt.

```
your app  --HTTP (loopback)-->  membrane serve  -->  llama.cpp
```

* Use any OpenAI-compatible SDK with `base_url = http://127.0.0.1:8642/v1` and
  a placeholder API key. Streaming (`stream: true`), `stop` sequences and
  switching `"model"` per request work -- see
  [client-compatibility.md](client-compatibility.md) for exactly what was run.
* Not supported, and rejected explicitly: tool calling and `response_format`
  beyond plain text (no constrained-decoding path exists, and none is faked).
* Build your task-specific behavior (prompts, output parsing, retries) in your
  application, not in MEMBRANE.

## From a web page

The server sets **no CORS headers** and listens on loopback with no
authentication, so a browser page on another origin cannot call it. That is
intentional. If browser access is ever added it must be an explicit opt-in
with an origin allowlist and a token -- never a wildcard
([security-model.md](security-model.md)). Today, put a small backend in
between.

## Reference integrations

None is shipped or promised here. A third-party tool (for example a
translation library) integrating with MEMBRANE is external work that targets
the generic API above; it needs an agreed contract on the tool's side, not new
MEMBRANE endpoints.
