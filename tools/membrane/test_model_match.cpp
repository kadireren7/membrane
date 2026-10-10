#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include "model_match.h"
#include "picker.h"
#include "test_helpers.h"

/* P3/P4: pure matcher and picker. */

static std::vector<membrane_model_cand_t>	cands(void)
{
	std::vector<membrane_model_cand_t>	v;
	membrane_model_cand_t				c;

	c.id = "qwen2.5-1.5b-instruct";
	c.display = "Qwen2.5-1.5B-Instruct";
	c.aliases = {"qwen2.5:1.5b", "qwen2.5-1.5b"};
	c.catalog = true;
	v.push_back(c);
	c = membrane_model_cand_t();
	c.id = "smollm2-135m-instruct";
	c.display = "SmolLM2-135M-Instruct";
	c.aliases = {"smollm2:135m"};
	c.catalog = true;
	v.push_back(c);
	c = membrane_model_cand_t();
	c.id = "smollm2-360m-instruct";
	c.display = "SmolLM2-360M-Instruct";
	c.aliases = {"smollm2:360m"};
	c.installed = true;
	c.catalog = true;
	v.push_back(c);
	c = membrane_model_cand_t();
	c.id = "qwen2.5-coder-1.5b-instruct";
	c.display = "Qwen2.5-Coder-1.5B-Instruct";
	c.catalog = true;
	v.push_back(c);
	return (v);
}

static void	test_exact(void)
{
	auto	c = cands();
	auto	m = membrane_model_match("QWEN2.5-1.5B-INSTRUCT", c);

	TEST_ASSERT(m.kind == MEMBRANE_MATCH_EXACT && m.hits.size() == 1
		&& m.hits[0] == 0, "case-insensitive exact id");
	m = membrane_model_match("qwen2.5:1.5b", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_EXACT && m.hits[0] == 0, "alias");
	m = membrane_model_match("Qwen2.5 1.5B Instruct", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_EXACT && m.hits[0] == 0,
		"display name with spaces normalizes to the id");
	m = membrane_model_match("smollm2_135m_instruct", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_EXACT && m.hits[0] == 1,
		"separator differences");
}

static void	test_prefix_substring(void)
{
	auto	c = cands();
	auto	m = membrane_model_match("qwen", c);

	TEST_ASSERT(m.kind == MEMBRANE_MATCH_PREFIX && m.hits.size() == 2,
		"'qwen' is a prefix of two models: ambiguous, never silently first");
	m = membrane_model_match("smollm2-135", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_PREFIX && m.hits.size() == 1
		&& m.hits[0] == 1, "unique prefix");
	m = membrane_model_match("360m", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_SUBSTRING && m.hits.size() == 1
		&& m.hits[0] == 2, "substring");
	m = membrane_model_match("coder 1.5b", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_SUBSTRING && m.hits.size() == 1
		&& m.hits[0] == 3, "every token must appear");
	m = membrane_model_match("smollm2", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_PREFIX && m.hits.size() == 2
		&& m.hits[0] == 2 && m.hits[1] == 1, "installed model is listed "
			"first, then by id");
}

static void	test_fuzzy_only_after_exact_forms_fail(void)
{
	auto	c = cands();
	auto	m = membrane_model_match("qwne2.5-1.5b-instruct", c);

	TEST_ASSERT(m.kind == MEMBRANE_MATCH_FUZZY && m.hits.size() == 1
		&& m.hits[0] == 0 && m.distance == 2, "transposition typo");
	m = membrane_model_match("smolm2-360m-instruct", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_FUZZY && m.hits[0] == 2
		&& m.distance == 1, "one missing letter");
	m = membrane_model_match("zzzzzzzz", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_NONE && m.hits.empty(), "no match");
	m = membrane_model_match("xyz", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_NONE, "queries under 4 characters "
		"are never fuzzy-matched");
	m = membrane_model_match("", c);
	TEST_ASSERT(m.kind == MEMBRANE_MATCH_NONE, "empty query");
	m = membrane_model_match("qwen", c);
	TEST_ASSERT(m.kind != MEMBRANE_MATCH_FUZZY, "a prefix hit never "
		"degrades to fuzzy");
}

static void	test_deterministic_and_order_independent(void)
{
	auto	c = cands();
	auto	r = c;

	std::reverse(r.begin(), r.end());
	auto	a = membrane_model_match("smollm2", c);
	auto	b = membrane_model_match("smollm2", r);

	TEST_ASSERT(a.kind == b.kind && a.hits.size() == b.hits.size()
		&& c[a.hits[0]].id == r[b.hits[0]].id
		&& c[a.hits[1]].id == r[b.hits[1]].id, "input order is irrelevant");
	TEST_ASSERT(membrane_model_normalize("Llama3:latest")
		== membrane_model_normalize("llama3"), "normalize");
}

/* --- picker ------------------------------------------------------------ */

static std::vector<membrane_picker_item_t>	items(size_t n)
{
	std::vector<membrane_picker_item_t>	v;

	for (size_t i = 0; i < n; ++i)
		v.push_back({"Model " + std::to_string(i + 1),
			i % 2 ? "tiny . fast" : "general . 1.1 GB"});
	return (v);
}

static membrane_picker_result_t	pick(const std::vector<membrane_picker_item_t> &v,
									const std::string &input, std::string *out,
									size_t page_size = 8)
{
	std::istringstream			in(input);
	std::ostringstream			os;
	membrane_picker_opts_t		o;

	o.title = "Installed models";
	o.page_size = page_size;
	auto	r = membrane_picker_run(v, o, in, os);

	*out = os.str();
	return (r);
}

static void	test_picker_select_and_pages(void)
{
	std::string	out;
	auto		v = items(11);
	auto		r = pick(v, "2\n", &out);

	TEST_ASSERT(r.status == MEMBRANE_PICK_SELECTED && r.index == 1, "select 2");
	TEST_ASSERT(out.find("Installed models -- Page 1/2") != std::string::npos
		&& out.find("1. Model 1") != std::string::npos
		&& out.find("[1-11] select   n next   / search   q quit")
			!= std::string::npos && out.find("p prev") == std::string::npos,
		"title, numbering, first-page hints");
	r = pick(v, "n\n9\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_SELECTED && r.index == 8,
		"numbers are global, so page 2 can pick 9");
	TEST_ASSERT(out.find("Page 2/2") != std::string::npos
		&& out.find("p prev") != std::string::npos
		&& out.find("n next") == out.rfind("n next") , "page 2 hints");
	r = pick(v, "n\nn\np\n1\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_SELECTED && r.index == 0,
		"next past the end stays; prev goes back");
}

static void	test_picker_search_quit_invalid(void)
{
	std::string	out;
	auto		v = items(11);
	auto		r = pick(v, "/tiny\n1\n", &out);

	TEST_ASSERT(r.status == MEMBRANE_PICK_SELECTED && r.index == 1
		&& out.find("(filter: \"tiny\")") != std::string::npos,
		"search filters; the first match is number 1");
	r = pick(v, "/zzz\n1\n/\n3\n", &out);
	TEST_ASSERT(out.find("(nothing matches)") != std::string::npos
		&& r.status == MEMBRANE_PICK_SELECTED && r.index == 2,
		"empty result is shown; '/' clears the filter");
	r = pick(v, "q\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_QUIT, "quit");
	r = pick(v, "", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_QUIT, "EOF quits");
	r = pick(v, "99\nabc\n0\n-1\nx\n \n3\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_QUIT && out.find("Please enter a "
			"number") != std::string::npos, "too many bad inputs gives up "
				"instead of looping");
	r = pick(v, "99\n2\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_SELECTED && r.index == 1,
		"one bad input is forgiven");
	r = pick(std::vector<membrane_picker_item_t>(), "1\nq\n", &out);
	TEST_ASSERT(r.status == MEMBRANE_PICK_QUIT, "empty list");
}

int	main(void)
{
	test_exact();
	test_prefix_substring();
	test_fuzzy_only_after_exact_forms_fail();
	test_deterministic_and_order_independent();
	test_picker_select_and_pages();
	test_picker_search_quit_invalid();
	printf("test_model_match: all tests passed\n");
	return (0);
}
