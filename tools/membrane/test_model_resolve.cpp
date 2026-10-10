#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "model_resolve.h"
#include "test_helpers.h"

/* P3/P4: the confirmation policy around the pure matcher. */

static std::vector<membrane_model_cand_t>	cands(void)
{
	std::vector<membrane_model_cand_t>	v;
	membrane_model_cand_t				c;

	c.id = "qwen2.5-1.5b-instruct";
	c.display = "Qwen2.5-1.5B-Instruct";
	c.catalog = true;
	v.push_back(c);
	c = membrane_model_cand_t();
	c.id = "qwen2.5-coder-1.5b-instruct";
	c.display = "Qwen2.5-Coder-1.5B-Instruct";
	c.catalog = true;
	v.push_back(c);
	c = membrane_model_cand_t();
	c.id = "smollm2-360m-instruct";
	c.display = "SmolLM2-360M-Instruct";
	c.installed = true;
	c.catalog = true;
	v.push_back(c);
	return (v);
}

static membrane_resolve_result_t	run(const std::string &q, bool tty,
										const std::string &input,
										std::string *out)
{
	std::istringstream			in(input);
	std::ostringstream			os;
	membrane_resolve_io_t		io;

	io.interactive = tty;
	io.in = &in;
	io.out = &os;
	auto						r = membrane_model_resolve(q, cands(),
			"membrane use", io);
	*out = os.str() + r.message;
	return (r);
}

static void	test_exact_never_asks(void)
{
	std::string	out;
	auto		r = run("SmolLM2-360M-Instruct", false, "", &out);

	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_EXACT && r.id
		== "smollm2-360m-instruct" && r.installed, "exact, even with no TTY");
	TEST_ASSERT(out.empty(), "no question was asked");
}

static void	test_single_suggestion(void)
{
	std::string	out;
	auto		r = run("smollm", true, "\n", &out);

	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_CONFIRMED && r.id
		== "smollm2-360m-instruct", "Enter accepts the default (yes)");
	TEST_ASSERT(out.find("Did you mean") != std::string::npos
		&& out.find("SmolLM2-360M-Instruct") != std::string::npos
		&& out.find("[Y/n]") != std::string::npos, "the question is shown");
	r = run("smollm", true, "n\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_DECLINED, "no");
	r = run("smollm", true, "", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_DECLINED, "EOF is not consent");
	r = run("smolm2-360m-instruct", true, "y\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_CONFIRMED && r.kind
		== MEMBRANE_MATCH_FUZZY, "a typo is suggested, not applied");
}

static void	test_non_interactive_never_chooses(void)
{
	std::string	out;
	auto		r = run("smollm", false, "y\n", &out);

	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_NEEDS_CHOICE
		&& r.id.empty(), "no TTY => nothing chosen, even if stdin says yes");
	TEST_ASSERT(out.find("smollm2-360m-instruct") != std::string::npos
		&& out.find("`membrane use smollm2-360m-instruct`")
			!= std::string::npos, "candidates + the exact command to run");
	r = run("qwen", false, "", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_NEEDS_CHOICE
		&& out.find("Multiple models match \"qwen\"") != std::string::npos
		&& out.find("qwen2.5-1.5b-instruct") != std::string::npos
		&& out.find("qwen2.5-coder-1.5b-instruct") != std::string::npos,
		"ambiguity is listed");
	r = run("smolm2-360m-instruct", false, "", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_NEEDS_CHOICE,
		"fuzzy is never auto-applied");
}

static void	test_ambiguous_interactive_uses_picker(void)
{
	std::string	out;
	auto		r = run("qwen", true, "2\n", &out);

	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_CONFIRMED && r.id
		== "qwen2.5-coder-1.5b-instruct", "picked number 2");
	TEST_ASSERT(out.find("Models matching \"qwen\"") != std::string::npos
		&& out.find("[available to install]") != std::string::npos,
		"picker with install state");
	r = run("qwen", true, "q\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_DECLINED, "quit");
}

static void	test_not_found(void)
{
	std::string	out;
	auto		r = run("llama-405b-turbo", true, "", &out);

	TEST_ASSERT(r.status == MEMBRANE_RESOLVE_NOT_FOUND
		&& r.message.find("no model matches") != std::string::npos,
		"nothing close");
}

int	main(void)
{
	test_exact_never_asks();
	test_single_suggestion();
	test_non_interactive_never_chooses();
	test_ambiguous_interactive_uses_picker();
	test_not_found();
	printf("test_model_resolve: all tests passed\n");
	return (0);
}
