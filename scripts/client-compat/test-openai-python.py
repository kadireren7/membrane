#!/usr/bin/env python3
"""Real Python `openai` SDK client harness for Mega Phase D, PR D7.

Drives the real, official Python `openai` package (not a hand-rolled
HTTP client) against a real, already-running `membrane serve` instance.
Requires:
  - `pip install openai` (not a MEMBRANE runtime dependency -- this
    script is a manual/CI validation tool, never imported by any
    product code).
  - a real `membrane serve` instance already running and reachable.
  - a real model already installed and registered under the name this
    script is given (see --model), reachable via that running server.

Usage:
  python3 scripts/client-compat/test-openai-python.py \
      --base-url http://127.0.0.1:8642/v1 --model smollm2-135m-instruct

Exit code: 0 if every check passes, 1 otherwise.
"""
import argparse
import sys

import openai


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-url", default="http://127.0.0.1:8642/v1")
	parser.add_argument("--model", default="smollm2-135m-instruct")
	args = parser.parse_args()

	client = openai.OpenAI(base_url=args.base_url,
		api_key="sk-local-dummy-not-checked")
	model = args.model
	failures = []

	def check(name, ok, detail=""):
		print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
		if not ok:
			failures.append(name)

	print(f"openai python SDK version: {openai.__version__}")

	try:
		models = client.models.list()
		ids = [m.id for m in models.data]
		check("models.list()", model in ids, f"ids={ids}")
	except Exception as e:  # noqa: BLE001
		check("models.list()", False, f"raised {type(e).__name__}: {e}")

	try:
		resp = client.chat.completions.create(model=model,
			messages=[{"role": "user", "content": "Say the word: hello"}],
			max_tokens=16, stream=False)
		content = resp.choices[0].message.content
		finish_reason = resp.choices[0].finish_reason
		check("non-streaming chat", len(content) > 0,
			f"finish_reason={finish_reason}")
		check("non-stream usage present",
			resp.usage is not None and resp.usage.total_tokens > 0,
			str(resp.usage))
	except Exception as e:  # noqa: BLE001
		check("non-streaming chat", False, f"raised {type(e).__name__}: {e}")

	try:
		stream = client.chat.completions.create(model=model,
			messages=[{"role": "user", "content": "Count from one to three."}],
			max_tokens=32, stream=True)
		chunks = list(stream)
		text = "".join(c.choices[0].delta.content or "" for c in chunks
			if c.choices and c.choices[0].delta)
		check("streaming chat", len(chunks) > 0 and len(text) > 0,
			f"chunks={len(chunks)}")
	except Exception as e:  # noqa: BLE001
		check("streaming chat", False, f"raised {type(e).__name__}: {e}")

	try:
		client.chat.completions.create(model="this-model-does-not-exist",
			messages=[{"role": "user", "content": "hi"}])
		check("unknown model rejected", False, "did not raise")
	except openai.NotFoundError as e:
		check("unknown model rejected", True, str(e))
	except Exception as e:  # noqa: BLE001
		check("unknown model rejected", False, f"raised {type(e).__name__}: {e}")

	try:
		client.chat.completions.create(model=model,
			messages=[{"role": "user", "content": "weather?"}],
			tools=[{"type": "function", "function": {"name": "get_weather",
				"parameters": {"type": "object"}}}])
		check("tools explicitly rejected", False, "did not raise")
	except openai.BadRequestError as e:
		code = e.response.json().get("error", {}).get("code")
		check("tools explicitly rejected", code == "UNSUPPORTED_TOOL_CALLING",
			f"code={code}")
	except Exception as e:  # noqa: BLE001
		check("tools explicitly rejected", False, f"raised {type(e).__name__}: {e}")

	try:
		resp = client.chat.completions.create(model=model,
			messages=[{"role": "user", "content": "Say hello."}],
			max_tokens=60, stop=["<|im_end|>"])
		content = resp.choices[0].message.content
		check("stop sequence real early termination",
			"<|im_end|>" not in content
			and resp.choices[0].finish_reason == "stop"
			and resp.usage.completion_tokens < 60,
			f"completion_tokens={resp.usage.completion_tokens}")
	except Exception as e:  # noqa: BLE001
		check("stop sequence real early termination", False,
			f"raised {type(e).__name__}: {e}")

	try:
		resp = client.chat.completions.create(model=model,
			messages=[{"role": "user", "content": "hi"}], max_tokens=8, seed=42)
		check("seed field accepted without error", True, "")
	except Exception as e:  # noqa: BLE001
		check("seed field accepted without error", False,
			f"raised {type(e).__name__}: {e}")

	try:
		resp = client.chat.completions.create(model=model,
			messages=[
				{"role": "system", "content": "You are terse."},
				{"role": "user", "content": "My name is Kadir."},
				{"role": "assistant", "content": "Nice to meet you, Kadir."},
				{"role": "user", "content": "What is my name?"},
			], max_tokens=16)
		check("multi-turn chat", len(resp.choices[0].message.content) > 0, "")
	except Exception as e:  # noqa: BLE001
		check("multi-turn chat", False, f"raised {type(e).__name__}: {e}")

	try:
		resp = client.chat.completions.create(model=model,
			messages=[{"role": "user",
				"content": "Merhaba! Bu bir test \U0001F600 çğıöşü"}],
			max_tokens=16)
		resp.choices[0].message.content.encode("utf-8")
		check("UTF-8 request/response round trip", True, "")
	except Exception as e:  # noqa: BLE001
		check("UTF-8 request/response round trip", False,
			f"raised {type(e).__name__}: {e}")

	print(f"\n{len(failures)} failure(s)")
	for f in failures:
		print(f"  - {f}")
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
