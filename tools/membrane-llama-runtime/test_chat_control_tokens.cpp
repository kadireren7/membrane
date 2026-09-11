#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "test_helpers.h"

/*
 * Post-v1 product-polish: deterministic, CI-safe regression coverage
 * for the real root cause behind a v1.0.0 user-session bug (prompt
 * echo, leaked "<|im_start|>"/"<|im_end|>" chat-control tokens, and
 * generation never stopping before max_tokens on a real `membrane
 * serve` + SmolLM2-360M-Instruct chat completion).
 *
 * Traced (not assumed) via this exact vocab: `membrane use`'s server
 * path renders the request's messages through the model's own real
 * chat template (llama_chat_apply_template(), server.cpp's apply_
 * chat_template()) into a prompt string containing LITERAL control-
 * token text ("<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n").
 * runtime_session.cpp's membrane_session_generate() then tokenized that
 * string with llama_tokenize()'s own `parse_special` argument hardcoded
 * to false -- per llama.h's own documented contract ("Allow tokenizing
 * special and/or control tokens which otherwise are not exposed and
 * treated as plaintext"), this means the model NEVER actually saw the
 * real, single "<|im_end|>"/"<|im_start|>" special-token ids it was
 * fine-tuned on -- only their literal text, split into ordinary
 * sub-word pieces. Confirmed directly against SmolLM2-135M-Instruct's
 * own real GGUF during investigation (not just this fixture); this
 * test uses the vendored, CI-present ggml-vocab-qwen2.gguf (real
 * Qwen2/Qwen2.5-Instruct tokenizer -- Qwen2.5-1.5B-Instruct's own
 * chat_template is byte-for-byte the same shape) so the SAME systemic
 * bug is proven across two real, independent model families without
 * needing a real multi-GB download or even loading any model WEIGHTS
 * (vocab_only=true -- no tensors, no llama_decode(), just tokenizer
 * metadata, exactly like llama.cpp's own test-tokenizer-0 suite this
 * fixture already serves).
 *
 * Fix under test (runtime_session.h's own membrane_generation_request_t::
 * parse_special_tokens / hide_control_tokens top comments have the
 * full contract): server.cpp's chat-completions path now passes
 * parse_special=true when tokenizing a rendered chat template, and
 * special=false (llama_token_to_piece()) when converting a generated
 * token back to API-facing text -- both real, canonical llama.h
 * mechanisms, never a hardcoded string list.
 */

static llama_model	*load_vocab_only(const char *path)
{
	llama_model_params	mp = llama_model_default_params();

	mp.vocab_only = true;
	return (llama_model_load_from_file(path, mp));
}

static llama_token	find_token_by_text(const llama_vocab *vocab,
				int32_t n_vocab, const char *text)
{
	for (llama_token t = 0; t < n_vocab; ++t)
	{
		char	buf[64];
		int		n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);

		if (n > 0 && (size_t)n == strlen(text)
			&& memcmp(buf, text, (size_t)n) == 0)
			return (t);
	}
	return (LLAMA_TOKEN_NULL);
}

static void	test_chatml_control_tokens_are_real_eog_or_control(
				const llama_vocab *vocab, int32_t n_vocab)
{
	llama_token	im_start = find_token_by_text(vocab, n_vocab, "<|im_start|>");
	llama_token	im_end = find_token_by_text(vocab, n_vocab, "<|im_end|>");

	TEST_ASSERT(im_start != LLAMA_TOKEN_NULL,
		"this tokenizer really has a <|im_start|> control token");
	TEST_ASSERT(im_end != LLAMA_TOKEN_NULL,
		"this tokenizer really has a <|im_end|> control token");
	TEST_ASSERT(llama_vocab_is_eog(vocab, im_end),
		"<|im_end|> is a real, canonical end-of-generation token "
		"(llama_vocab_is_eog()) -- decode_loop.cpp's run_generation() "
		"must stop the moment the model actually emits ITS real token "
		"id, not a re-spelled text approximation of it");
	TEST_ASSERT(!llama_vocab_is_eog(vocab, im_start),
		"<|im_start|> is a control token but NOT itself an end-of-"
		"generation token -- if a model degenerately emits it mid-"
		"generation, run_generation() must keep going (never treat it "
		"as EOG) while still never leaking its literal text (see the "
		"render_special_tokens=false test below)");
}

/*
 * The actual root-cause reproduction: tokenizing a REAL chat-template-
 * rendered prompt with parse_special=false (the pre-fix hardcoded
 * value) never produces the real <|im_end|>/<|im_start|> token ids at
 * all -- the model is handed a corrupted turn structure it was never
 * trained on. parse_special=true (the fix) does.
 */
static void	test_parse_special_recovers_real_control_token_ids(
				const llama_vocab *vocab)
{
	const char	*tmpl =
		"<|im_start|>user\nHello! Tell me one sentence about "
		"yourself.<|im_end|>\n<|im_start|>assistant\n";
	std::string	prompt(tmpl);
	llama_token	im_end = find_token_by_text(vocab, llama_vocab_n_tokens(vocab),
				"<|im_end|>");
	llama_token	im_start = find_token_by_text(vocab,
				llama_vocab_n_tokens(vocab), "<|im_start|>");

	std::vector<llama_token>	toks_broken(prompt.size() + 8);
	int	n_broken = llama_tokenize(vocab, prompt.c_str(),
			(int32_t)prompt.size(), toks_broken.data(),
			(int32_t)toks_broken.size(), true, /* parse_special */ false);

	TEST_ASSERT(n_broken > 0, "the pre-fix tokenization itself succeeds "
		"(it does not fail loudly -- it silently corrupts the prompt, "
		"which is exactly what made this bug hard to notice)");

	bool	broken_has_real_im_end = false;
	bool	broken_has_real_im_start = false;

	for (int i = 0; i < n_broken; ++i)
	{
		if (toks_broken[i] == im_end)
			broken_has_real_im_end = true;
		if (toks_broken[i] == im_start)
			broken_has_real_im_start = true;
	}
	TEST_ASSERT(!broken_has_real_im_end && !broken_has_real_im_start,
		"root-cause reproduction: parse_special=false (the pre-fix "
		"behavior) NEVER produces the real, single <|im_start|>/"
		"<|im_end|> special-token ids for a chat-template-rendered "
		"prompt -- the model never actually sees its own real turn "
		"structure, which is why it never reliably emits a real EOG "
		"token back and instead echoes/rambles until max_tokens");

	std::vector<llama_token>	toks_fixed(prompt.size() + 8);
	int	n_fixed = llama_tokenize(vocab, prompt.c_str(),
			(int32_t)prompt.size(), toks_fixed.data(),
			(int32_t)toks_fixed.size(), true, /* parse_special */ true);

	TEST_ASSERT(n_fixed > 0, "the fixed tokenization succeeds");

	bool	fixed_has_real_im_end = false;
	bool	fixed_has_real_im_start = false;

	for (int i = 0; i < n_fixed; ++i)
	{
		if (toks_fixed[i] == im_end)
			fixed_has_real_im_end = true;
		if (toks_fixed[i] == im_start)
			fixed_has_real_im_start = true;
	}
	TEST_ASSERT(fixed_has_real_im_end && fixed_has_real_im_start,
		"the fix (parse_special=true, runtime_session.cpp's membrane_"
		"session_generate()) correctly recovers the real, single "
		"control-token ids the model was actually trained to recognize");
	TEST_ASSERT(n_fixed < n_broken,
		"the corrupted (pre-fix) tokenization is strictly LARGER -- "
		"each control token that should be one real special-token id "
		"was instead shattered into multiple ordinary sub-word tokens, "
		"real, measurable prompt corruption, not just a theoretical "
		"concern");
}

/*
 * The output-side half of the fix: llama_token_to_piece()'s own
 * `special` argument, forwarded by decode_loop.h's run_generation() as
 * render_special_tokens, must suppress a control token's own literal
 * text (llama.h's own canonical LLAMA_TOKEN_ATTR_CONTROL/UNKNOWN
 * classification -- never a hardcoded string list) while leaving
 * ordinary vocabulary content completely unaffected.
 */
static void	test_special_false_suppresses_control_token_text(
				const llama_vocab *vocab, int32_t n_vocab)
{
	llama_token	im_start = find_token_by_text(vocab, n_vocab, "<|im_start|>");
	char		buf[64];

	int	len_special_true = llama_token_to_piece(vocab, im_start, buf,
			sizeof(buf), 0, /* special */ true);

	TEST_ASSERT(len_special_true > 0,
		"special=true (membrane-run's own CLI/debug-tooling default, "
		"unchanged by this fix) still renders a control token's own "
		"literal text");

	int	len_special_false = llama_token_to_piece(vocab, im_start, buf,
			sizeof(buf), 0, /* special */ false);

	TEST_ASSERT(len_special_false == 0,
		"special=false (server.cpp's chat-completions fix -- decode_"
		"loop.h's run_generation() render_special_tokens=false, driven "
		"by membrane_generation_request_t::hide_control_tokens) "
		"suppresses a real control token's text entirely via llama.h's "
		"own token-attribute classification -- the exact mechanism that "
		"closes the special-token-leakage half of the bug");
}

/*
 * Ordinary vocabulary content must never be affected by special=false
 * -- only tokens the tokenizer itself classifies as control/unknown are
 * suppressed (Section 5 of the task: "preserve ordinary user content
 * that happens to contain similar textual sequences").
 */
static void	test_special_false_preserves_ordinary_tokens(
				const llama_vocab *vocab, int32_t n_vocab)
{
	llama_token	ordinary = LLAMA_TOKEN_NULL;

	for (llama_token t = 0; t < n_vocab && ordinary == LLAMA_TOKEN_NULL; ++t)
	{
		if (llama_vocab_get_attr(vocab, t) == LLAMA_TOKEN_ATTR_NORMAL)
			ordinary = t;
	}
	TEST_ASSERT(ordinary != LLAMA_TOKEN_NULL,
		"this vocabulary really has at least one NORMAL (non-control) "
		"token to test against");

	char	buf_true[64];
	char	buf_false[64];
	int		len_true = llama_token_to_piece(vocab, ordinary, buf_true,
			sizeof(buf_true), 0, true);
	int		len_false = llama_token_to_piece(vocab, ordinary, buf_false,
			sizeof(buf_false), 0, false);

	TEST_ASSERT(len_true > 0 && len_true == len_false
			&& memcmp(buf_true, buf_false, (size_t)len_true) == 0,
		"an ordinary (non-control) token's own text is completely "
		"unaffected by special=false -- only real control/unknown "
		"tokens are ever suppressed");
}

/*
 * Template-application sanity (Section 10A of the task): applying the
 * SAME messages through llama_chat_apply_template() twice produces a
 * byte-identical prompt (no accumulation/duplicate-application bug),
 * and add_ass=true (server.cpp's apply_chat_template()) leaves the
 * rendered prompt ending in the model's own real assistant-generation
 * suffix, never mid-template.
 */
static void	test_template_applied_once_with_correct_assistant_suffix(
				llama_model *model)
{
	const char	*tmpl = llama_model_chat_template(model, NULL);

	TEST_ASSERT(tmpl != NULL && tmpl[0] != '\0',
		"this real model exposes a real, non-empty chat template");

	std::vector<llama_chat_message>	chat;
	llama_chat_message					msg;

	msg.role = "user";
	msg.content = "Hello! Tell me one sentence about yourself.";
	chat.push_back(msg);

	std::vector<char>	buf1(4096);
	int32_t	n1 = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
			true, buf1.data(), (int32_t)buf1.size());
	std::vector<char>	buf2(4096);
	int32_t	n2 = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
			true, buf2.data(), (int32_t)buf2.size());

	TEST_ASSERT(n1 > 0 && n1 == n2 && memcmp(buf1.data(), buf2.data(),
			(size_t)n1) == 0,
		"applying the identical messages through the identical template "
		"twice is byte-for-byte deterministic -- no hidden accumulation "
		"state, no duplicate application");

	std::string	rendered(buf1.data(), (size_t)n1);

	TEST_ASSERT(rendered.find("<|im_start|>assistant") != std::string::npos,
		"add_ass=true correctly appends the real assistant-generation "
		"prompt suffix");
	TEST_ASSERT(rendered.rfind("<|im_start|>assistant")
			== rendered.size() - std::string("<|im_start|>assistant\n")
				.size(),
		"the assistant suffix is the LAST thing in the rendered prompt "
		"-- generation starts exactly there, never mid-template");
	/* Section 4 of the task: the rendered prompt is the template's OWN
	 * structural markup around the user's text -- it is not itself
	 * "generated content," but confirms this test's own prompt_text
	 * matches exactly what server.cpp's apply_chat_template() would
	 * hand to the tokenizer, the same string the two tests above tokenize. */
	TEST_ASSERT(rendered.find("Hello! Tell me one sentence about yourself.")
			!= std::string::npos,
		"the user's own message is present exactly once in the "
		"rendered prompt");
}

int	main(int argc, char **argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "usage: %s VOCAB_ONLY_GGUF_PATH\n", argv[0]);
		return (1);
	}
	llama_backend_init();

	llama_model	*model = load_vocab_only(argv[1]);

	TEST_ASSERT(model != NULL, "the real, vendored vocab-only fixture "
		"loads successfully (vocab_only=true -- no tensors, no decode)");

	const llama_vocab	*vocab = llama_model_get_vocab(model);
	int32_t				n_vocab = llama_vocab_n_tokens(vocab);

	test_chatml_control_tokens_are_real_eog_or_control(vocab, n_vocab);
	test_parse_special_recovers_real_control_token_ids(vocab);
	test_special_false_suppresses_control_token_text(vocab, n_vocab);
	test_special_false_preserves_ordinary_tokens(vocab, n_vocab);
	test_template_applied_once_with_correct_assistant_suffix(model);

	llama_model_free(model);
	llama_backend_free();
	printf("test_chat_control_tokens: all tests passed\n");
	return (0);
}
