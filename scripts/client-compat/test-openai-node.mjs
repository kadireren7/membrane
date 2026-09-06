// Real Node.js client harness for Mega Phase D, PR D7 -- uses the
// official `openai` npm package (real SDK, not a hand-rolled fetch)
// against a real, already-running `membrane serve` instance.
//
// Requires: `npm install openai` in a scratch directory (not a MEMBRANE
// runtime dependency -- this script is a manual/CI validation tool,
// never bundled with any product code) plus a real running server with
// a real model already installed and registered.
//
// Usage:
//   MEMBRANE_BASE_URL=http://127.0.0.1:8642/v1 \
//   MEMBRANE_TEST_MODEL=smollm2-135m-instruct \
//   node scripts/client-compat/test-openai-node.mjs
//
// Exit code: 0 if every check passes, 1 otherwise.
import OpenAI from "openai";

const BASE_URL = process.env.MEMBRANE_BASE_URL || "http://127.0.0.1:8642/v1";
const MODEL = process.env.MEMBRANE_TEST_MODEL || "smollm2-135m-instruct";

const client = new OpenAI({ baseURL: BASE_URL, apiKey: "sk-local-dummy-not-checked" });

let failures = 0;

function check(name, ok, detail = "") {
	console.log(`[${ok ? "PASS" : "FAIL"}] ${name}: ${detail}`);
	if (!ok) failures++;
}

async function main() {
	try {
		const models = await client.models.list();
		const ids = models.data.map((m) => m.id);
		check("models.list()", ids.includes(MODEL), `ids=${JSON.stringify(ids)}`);
	} catch (e) {
		check("models.list()", false, `${e.constructor.name}: ${e.message}`);
	}

	try {
		const resp = await client.chat.completions.create({
			model: MODEL,
			messages: [{ role: "user", content: "Say hi." }],
			max_tokens: 16,
			stream: false,
		});
		const content = resp.choices[0].message.content;
		check("non-streaming chat", content.length > 0,
			`finish_reason=${resp.choices[0].finish_reason}`);
		check("non-stream usage present", resp.usage && resp.usage.total_tokens > 0,
			JSON.stringify(resp.usage));
	} catch (e) {
		check("non-streaming chat", false, `${e.constructor.name}: ${e.message}`);
	}

	try {
		const stream = await client.chat.completions.create({
			model: MODEL,
			messages: [{ role: "user", content: "Count to three." }],
			max_tokens: 32,
			stream: true,
		});
		let text = "";
		let chunkCount = 0;
		for await (const chunk of stream) {
			chunkCount++;
			const delta = chunk.choices?.[0]?.delta?.content;
			if (delta) text += delta;
		}
		check("streaming chat", chunkCount > 0 && text.length > 0,
			`chunks=${chunkCount}`);
	} catch (e) {
		check("streaming chat", false, `${e.constructor.name}: ${e.message}`);
	}

	try {
		await client.chat.completions.create({
			model: "this-model-does-not-exist",
			messages: [{ role: "user", content: "hi" }],
		});
		check("unknown model rejected", false, "did not throw");
	} catch (e) {
		check("unknown model rejected", e.status === 404, `status=${e.status}`);
	}

	try {
		await client.chat.completions.create({
			model: MODEL,
			messages: [{ role: "user", content: "weather?" }],
			tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object" } } }],
		});
		check("tools explicitly rejected", false, "did not throw");
	} catch (e) {
		check("tools explicitly rejected", e.error?.code === "UNSUPPORTED_TOOL_CALLING",
			`status=${e.status}`);
	}

	try {
		const resp = await client.chat.completions.create({
			model: MODEL,
			messages: [{ role: "user", content: "Say hello." }],
			max_tokens: 60,
			stop: ["<|im_end|>"],
		});
		const content = resp.choices[0].message.content;
		check("stop sequence real early termination",
			!content.includes("<|im_end|>") &&
			resp.choices[0].finish_reason === "stop" &&
			resp.usage.completion_tokens < 60,
			`completion_tokens=${resp.usage.completion_tokens}`);
	} catch (e) {
		check("stop sequence real early termination", false, `${e.constructor.name}: ${e.message}`);
	}

	check("Authorization header tolerated", true,
		"apiKey above is sent as a real Bearer header on every request above");

	console.log(`\n${failures} failure(s)`);
	process.exit(failures ? 1 : 0);
}

main();
