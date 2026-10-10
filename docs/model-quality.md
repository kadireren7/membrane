# Model quality vs. runtime health

MEMBRANE manages **inference and memory**: it picks a model file that fits,
loads it, plans context / GPU layers / KV cache, serves it, and tells you what
memory is doing. It does **not** make a model smarter, and it does not judge
answers.

So when a reply is poor, ask first whether it is the model:

| What you see | Usually | Not |
|---|---|---|
| The answer echoes your prompt, repeats itself, or rambles | a tiny model (SmolLM2-135M/360M are smoke-test models) | a runtime failure |
| Broken or off-topic answers in a language other than English | the model's training | MEMBRANE |
| Confident wrong facts | every small model hallucinates | a bug in the server |
| A crash, an error object, `MODEL_LOAD_FAILED`, no tokens at all | the runtime or memory | the model |

Rules of thumb:

* If `membrane chat` produces text -- even bad text -- the runtime worked.
  `membrane doctor` is the check for "is the runtime healthy".
* Judge quality with a model sized for the task. The catalog
  ([model-catalog.md](model-catalog.md)) labels tiny models as tiny, says what
  each model is for, and says how much MEMBRANE actually knows about it
  (`validated` / `smoke-tested` / `metadata-only`). A label never claims
  quality.
* `membrane models recommend` suggests models that **fit your memory**; it
  does not rank their intelligence.
