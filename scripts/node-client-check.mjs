#!/usr/bin/env node
// Real second-client protocol validation for `membrane serve` (Mega
// Phase C, PR C3, "at least one additional real client validation
// beyond Python SDK"). Uses only Node's built-in `fetch` (Node >= 18)
// -- no npm install, keeping this lightweight/safe on a real memory-
// constrained dev host. Exercises the real, running server's own
// protocol compliance from a second, independent language ecosystem:
// GET /health, GET /v1/models (OpenAI `list` shape), and the real 404
// MODEL_NOT_FOUND error-response shape -- deliberately not attempting
// real generation here (kept separate from, and independent of, host
// memory availability at run time), matching the existing Python SDK
// evidence's own split between protocol-shape checks and real
// generation checks.
//
// Usage: node scripts/node-client-check.mjs http://127.0.0.1:PORT

const base = process.argv[2];
if (!base) {
	console.error("usage: node scripts/node-client-check.mjs <base_url>");
	process.exit(2);
}

let failures = 0;

function check(name, ok, detail) {
	console.log(`[${ok ? "PASS" : "FAIL"}] ${name}: ${detail}`);
	if (!ok) failures++;
}

const health = await fetch(`${base}/health`);
const healthBody = await health.json();
check("GET /health returns 200 with status ok", health.status === 200
	&& healthBody.status === "ok", JSON.stringify(healthBody));

const models = await fetch(`${base}/v1/models`);
const modelsBody = await models.json();
check("GET /v1/models returns 200 with OpenAI list shape",
	models.status === 200 && modelsBody.object === "list"
	&& Array.isArray(modelsBody.data), JSON.stringify(modelsBody));

const chat = await fetch(`${base}/v1/chat/completions`, {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({
		model: "node-client-check-nonexistent-model",
		messages: [{ role: "user", content: "hi" }],
	}),
});
const chatBody = await chat.json();
check("POST /v1/chat/completions with an unregistered model returns "
	+ "real 404 MODEL_NOT_FOUND", chat.status === 404
	&& chatBody?.error?.code === "MODEL_NOT_FOUND", JSON.stringify(chatBody));

// Mega Phase D, PR D7: the following four checks all happen BEFORE
// model resolution in handle_chat_completions() (server.cpp), so they
// are exercisable with no real model here too, matching the checks
// above -- a real second-language proof that these D7 fields are
// rejected/tolerated correctly, not just the C++ ctest suite's own
// coverage (test_server.cpp) or the heavier, real-SDK/real-model
// scripts/client-compat/ scripts.
const toolsRes = await fetch(`${base}/v1/chat/completions`, {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({
		model: "node-client-check-nonexistent-model",
		messages: [{ role: "user", content: "hi" }],
		tools: [{ type: "function", function: { name: "f", parameters: {} } }],
	}),
});
const toolsBody = await toolsRes.json();
check("POST with \"tools\" is rejected with UNSUPPORTED_TOOL_CALLING",
	toolsRes.status === 400
	&& toolsBody?.error?.code === "UNSUPPORTED_TOOL_CALLING",
	JSON.stringify(toolsBody));

const rfRes = await fetch(`${base}/v1/chat/completions`, {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({
		model: "node-client-check-nonexistent-model",
		messages: [{ role: "user", content: "hi" }],
		response_format: { type: "json_object" },
	}),
});
const rfBody = await rfRes.json();
check("POST with response_format \"json_object\" is rejected with "
	+ "UNSUPPORTED_RESPONSE_FORMAT", rfRes.status === 400
	&& rfBody?.error?.code === "UNSUPPORTED_RESPONSE_FORMAT",
	JSON.stringify(rfBody));

const stopRes = await fetch(`${base}/v1/chat/completions`, {
	method: "POST",
	headers: { "Content-Type": "application/json" },
	body: JSON.stringify({
		model: "node-client-check-nonexistent-model",
		messages: [{ role: "user", content: "hi" }],
		stop: ["a", "b", "c", "d", "e"],
	}),
});
const stopBody = await stopRes.json();
check("POST with more than 4 \"stop\" strings is rejected with "
	+ "INVALID_REQUEST", stopRes.status === 400
	&& stopBody?.error?.code === "INVALID_REQUEST", JSON.stringify(stopBody));

const authRes = await fetch(`${base}/v1/chat/completions`, {
	method: "POST",
	headers: {
		"Content-Type": "application/json",
		"Authorization": "Bearer sk-local-dummy-not-checked",
	},
	body: JSON.stringify({
		model: "node-client-check-nonexistent-model",
		messages: [{ role: "user", content: "hi" }],
	}),
});
const authBody = await authRes.json();
check("a real Authorization: Bearer header is tolerated (still a "
	+ "normal 404 MODEL_NOT_FOUND, never 401/403)", authRes.status === 404
	&& authBody?.error?.code === "MODEL_NOT_FOUND", JSON.stringify(authBody));

console.log(`\n${failures === 0 ? "node-client-check.mjs: all checks passed"
	: `node-client-check.mjs: ${failures} check(s) failed`}`);
process.exit(failures === 0 ? 0 : 1);
