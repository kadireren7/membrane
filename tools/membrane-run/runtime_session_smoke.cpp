#include <cstdio>
#include <cstring>

#include "runtime_session.h"

/*
 * Mega Phase A, PR A1, Section 7/20: dev-only real-evidence harness --
 * NOT ctest-registered, NOT installed (same precedent as
 * context_recommender_dryrun.cpp). Proves, against a REAL local GGUF
 * fixture, the one property PR A1 exists to add: "load model once,
 * generate multiple requests, keep model alive, free cleanly" -- three
 * sequential membrane_session_generate() calls against ONE
 * membrane_model_open() call, confirming the loaded llama_model* pointer
 * never changes across calls (no reload) and every call succeeds.
 *
 * Usage: runtime-session-smoke <model.gguf>
 */
int	main(int argc, char **argv)
{
	membrane_runtime_t			rt = {};
	membrane_run_opts_t			o = {};
	membrane_model_session_t	session;
	membrane_run_error_t		err;
	const char					*prompts[3] = {
		"Hello, how are you?",
		"What is the capital of France?",
		"Write one short sentence about the ocean.",
	};
	llama_model					*model_after_open;
	int							i;
	int							failures = 0;

	if (argc != 2)
	{
		fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
		return (1);
	}
	membrane_runtime_init(&rt);
	o.model_path = argv[1];
	o.gpu_layers = 0;
	o.kv_mode = MEMBRANE_KV_STORE_NATIVE;
	o.gen_tokens = 16;
	if (!membrane_model_open(&rt, argv[1], o, 256, &session, &err))
	{
		fprintf(stderr, "model_open failed: %s\n", err.human.c_str());
		membrane_runtime_shutdown(&rt);
		return (1);
	}
	model_after_open = session.model;
	printf("model_open: ok, model=%p\n", (void *)session.model);
	i = 0;
	while (i < 3)
	{
		membrane_generation_request_t	req = {};
		membrane_generation_result_t	res;

		req.o = &o;
		req.prompt_text = prompts[i];
		req.ctx_size = 256;
		req.token_cb = NULL;
		membrane_session_generate(&session, req, &res);
		printf("request #%d: ok=%d model=%p (%s) tokens=%zu text=\"%s\"\n",
			i + 1, res.ok, (void *)session.model,
			session.model == model_after_open ? "SAME -- no reload"
				: "DIFFERENT -- reloaded",
			res.gen_result.tokens.size(), res.text.c_str());
		if (!res.ok)
		{
			fprintf(stderr, "  error: %s\n", res.err.human.c_str());
			failures++;
		}
		if (session.model != model_after_open)
		{
			fprintf(stderr, "  UNEXPECTED: model pointer changed on an "
				"explicit CPU-only run (no fallback should have "
				"engaged)\n");
			failures++;
		}
		i++;
	}
	membrane_model_close(&session);
	printf("model_close: ok, model=%p\n", (void *)session.model);
	membrane_runtime_shutdown(&rt);
	if (failures > 0)
	{
		fprintf(stderr, "runtime-session-smoke: %d failure(s)\n", failures);
		return (1);
	}
	printf("runtime-session-smoke: 3/3 sequential requests succeeded, "
		"model reused (never reloaded)\n");
	return (0);
}
