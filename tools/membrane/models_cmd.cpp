#include "models_cmd.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <nlohmann/json.hpp>

#include "model_catalog.h"
#include "model_views.h"
#include "product_cli.h"
#include "runtime_session.h"
#include "ui.h"
#include "variant_selector.h"

using json = nlohmann::json;

/* See models_cmd.h's top comment. */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
		printf("%s\n", json{{"ok", false}, {"error", {{"code", code},
			{"message", message}}}}.dump().c_str());
	else
		fprintf(stderr, "membrane models: %s\n", message.c_str());
}

static std::string	join_tags(const std::vector<std::string> &t)
{
	std::string	s;

	for (size_t i = 0; i < t.size(); ++i)
		s += (i ? ", " : "") + t[i];
	return (s);
}

static json	row_json(const membrane_model_row_t &r)
{
	return (json{{"id", r.id}, {"display_name", r.display},
		{"runtime", r.runtime}, {"installed", r.installed},
		{"size_bytes", r.size_bytes ? json(r.size_bytes) : json(nullptr)},
		{"quant", r.quant.empty() ? json(nullptr) : json(r.quant)},
		{"parameters", r.params.empty() ? json(nullptr) : json(r.params)},
		{"architecture", r.arch.empty() ? json(nullptr) : json(r.arch)},
		{"tags", r.tags}, {"purpose", r.purpose.empty() ? json(nullptr)
			: json(r.purpose)}, {"tested_status", r.tested_status.empty()
			? json(nullptr) : json(r.tested_status)},
		{"license", r.license.empty() ? json(nullptr) : json(r.license)},
		{"loaded", r.resident_known ? json(r.resident) : json(nullptr)},
		{"default", r.is_default}});
}

static std::string	pad(const std::string &s, size_t w)
{
	/* width counts bytes; the names are ASCII */
	if (s.size() >= w)
		return (s);
	return (s + std::string(w - s.size(), ' '));
}

static int	cmd_installed(bool want_json)
{
	std::vector<membrane_model_row_t>	rows;

	membrane_model_rows_installed(&rows, true);
	if (want_json)
	{
		json	arr = json::array();

		for (const auto &r : rows)
			arr.push_back(row_json(r));
		printf("%s\n", json{{"schema_version", 1}, {"mode", "models"},
			{"models", arr}}.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (rows.empty())
	{
		printf("No models are installed yet.\n\nNext:\n  membrane models "
			"recommend     see what suits this computer\n  membrane use"
			"                 pick one to install\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	size_t	wn = 5;
	size_t	wi = 2;

	for (const auto &r : rows)
	{
		wn = std::max(wn, r.display.size());
		wi = std::max(wi, r.id.size());
	}
	printf("Installed models\n\n");
	printf("  %s  %s  %-8s %-8s %s\n", pad("MODEL", wn).c_str(),
		pad("ID", wi).c_str(), "SIZE", "QUANT", "STATE");
	for (const auto &r : rows)
	{
		std::string	state = r.resident ? membrane_ui_paint(MEMBRANE_UI_OK,
			"loaded") : membrane_ui_paint(MEMBRANE_UI_DIM, "installed");

		if (r.is_default)
			state += std::string("  ") + membrane_ui_paint(MEMBRANE_UI_INFO,
				"default");
		printf("  %s  %s  %-8s %-8s %s\n", pad(r.display, wn).c_str(),
			pad(r.id, wi).c_str(), membrane_human_size(r.size_bytes).c_str(),
			r.quant.empty() ? "?" : r.quant.c_str(), state.c_str());
	}
	printf("\nNext:\n  membrane chat                 chat with the default "
		"model\n  membrane use                  choose another\n"
		"  membrane models available     browse more\n");
	return (MEMBRANE_EXIT_SUCCESS);
}

static std::string	tested_label(const std::string &t)
{
	if (t == "validated")
		return (membrane_ui_paint(MEMBRANE_UI_OK, "validated"));
	if (t == "smoke-tested")
		return (membrane_ui_paint(MEMBRANE_UI_INFO, "smoke-tested"));
	return (membrane_ui_paint(MEMBRANE_UI_DIM, "metadata-only"));
}

static int	cmd_available(bool want_json)
{
	std::vector<membrane_model_row_t>	rows;

	membrane_model_rows_catalog(&rows);
	if (want_json)
	{
		json	arr = json::array();

		for (const auto &r : rows)
			arr.push_back(row_json(r));
		printf("%s\n", json{{"schema_version", 1}, {"mode",
			"models_available"}, {"models", arr}}.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("Models MEMBRANE can install\n\n");
	for (const auto &r : rows)
	{
		printf("  %s %s  (%s, from %s)%s\n", r.installed
			? membrane_ui_sym_ok().c_str() : membrane_ui_sym_dot().c_str(),
			r.display.c_str(), r.params.c_str(),
			membrane_human_size(r.size_bytes).c_str(),
			r.installed ? "  installed" : "");
		printf("      %s\n", join_tags(r.tags).c_str());
		printf("      %s\n", r.purpose.c_str());
		printf("      license: %s   tested: %s   id: %s\n", r.license.c_str(),
			tested_label(r.tested_status).c_str(), r.id.c_str());
	}
	printf("\n\"validated\" = MEMBRANE has run it and recorded the result; "
		"\"smoke-tested\" = loaded and answered once on a real host;\n"
		"\"metadata-only\" = listed from upstream metadata, never run by "
		"MEMBRANE. MEMBRANE manages the runtime, not model quality.\n");
	printf("\nNext:\n  membrane models recommend     what suits this "
		"computer\n  membrane use NAME             install and select "
		"one\n");
	return (MEMBRANE_EXIT_SUCCESS);
}

/* ---- recommend ------------------------------------------------------- */

typedef struct s_rec
{
	membrane_model_row_t		row;
	const membrane_catalog_family_t	*fam;
	const membrane_catalog_variant_t	*variant;
	std::string					fit;			/* good|tight|marginal */
	double						ratio;
	int							tested_rank;
	std::string					category;
}	rec_t;

static int	tested_rank(const std::string &t)
{
	return (t == "validated" ? 3 : t == "smoke-tested" ? 2 : 1);
}

static bool	has_tag(const membrane_model_row_t &r, const char *t)
{
	return (std::find(r.tags.begin(), r.tags.end(), t) != r.tags.end());
}

static int	cmd_recommend(bool want_json)
{
	membrane_catalog_t					cat = membrane_catalog_load();
	std::vector<membrane_model_row_t>	rows;
	membrane_host_meminfo_t				mem;
	membrane_variant_selector_input_t	hw;
	std::vector<rec_t>					fits;

	membrane_model_rows_catalog(&rows);
	membrane_read_host_meminfo(&mem);
	hw.host_total_bytes = mem.total_bytes;
	hw.host_available_bytes = mem.available_bytes;
	hw.host_available_known = mem.ok;
	for (const auto &r : rows)
	{
		const membrane_catalog_family_t	*fam = NULL;

		for (const auto &f : cat.families)
			if (f.name == r.id)
				fam = &f;
		if (fam == NULL)
			continue ;
		std::vector<membrane_variant_fit_t>	considered;
		const membrane_catalog_variant_t	*v = membrane_select_variant(*fam,
				hw, &considered);

		if (v == NULL)
			continue ;			/* nothing fits: not recommended */
		rec_t	x;

		x.row = r;
		x.fam = fam;
		x.variant = v;
		x.ratio = mem.ok && mem.available_bytes
			? (double)v->size_bytes / (double)mem.available_bytes : 1.0;
		x.fit = x.ratio <= 0.40 ? "good" : x.ratio <= 0.75 ? "tight"
			: "marginal";
		x.tested_rank = tested_rank(r.tested_status);
		fits.push_back(x);
	}
	/* one suggestion per category, chosen by a stated rule: validated before
	 * smoke-tested before metadata-only, then the LARGEST variant that still
	 * fits comfortably, then the id. No quality claim is made. */
	static const struct { const char *tag; const char *title; }	cats[] = {
		{"general", "General chat"}, {"multilingual", "Multilingual"},
		{"coding", "Coding"}, {"reasoning", "Reasoning"},
		{"long-context", "Long context"}, {"tiny", "Tiny / smoke test"}};
	std::vector<rec_t>	picked;
	std::vector<std::string>	titles;

	for (const auto &c : cats)
	{
		const rec_t	*best = NULL;

		for (const auto &x : fits)
		{
			if (!has_tag(x.row, c.tag))
				continue ;
			if (best == NULL || x.tested_rank > best->tested_rank
				|| (x.tested_rank == best->tested_rank
					&& (x.fit == "good") != (best->fit == "good")
					&& x.fit == "good")
				|| (x.tested_rank == best->tested_rank
					&& x.fit == best->fit
					&& x.variant->size_bytes > best->variant->size_bytes)
				|| (x.tested_rank == best->tested_rank && x.fit == best->fit
					&& x.variant->size_bytes == best->variant->size_bytes
					&& x.row.id < best->row.id))
				best = &x;
		}
		if (best != NULL)
		{
			picked.push_back(*best);
			titles.push_back(c.title);
		}
	}
	if (want_json)
	{
		json	arr = json::array();

		for (size_t i = 0; i < picked.size(); ++i)
		{
			json	j = row_json(picked[i].row);

			j["category"] = titles[i];
			j["variant"] = picked[i].variant->quant;
			j["download_bytes"] = picked[i].variant->size_bytes;
			j["expected_fit"] = picked[i].fit;
			j["fit_basis"] = "variant file size vs measured available RAM "
				"(weights only; KV cache not included)";
			arr.push_back(j);
		}
		printf("%s\n", json{{"schema_version", 1}, {"mode",
			"models_recommend"}, {"hardware", {{"ram_total_bytes",
			mem.ok ? json(mem.total_bytes) : json(nullptr)},
			{"ram_available_bytes", mem.ok ? json(mem.available_bytes)
			: json(nullptr)}}}, {"recommendations", arr}}.dump().c_str());
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("Suggested for this computer\n\n");
	printf("  Measured: %s RAM, %s available\n\n", mem.ok
		? membrane_human_size(mem.total_bytes).c_str() : "unknown",
		mem.ok ? membrane_human_size(mem.available_bytes).c_str()
			: "unknown");
	if (picked.empty())
	{
		printf("  Nothing in the catalog is expected to fit the memory "
			"available right now.\n  Free some memory (close other "
			"programs) and try again.\n");
		return (MEMBRANE_EXIT_SUCCESS);
	}
	for (size_t i = 0; i < picked.size(); ++i)
	{
		const rec_t	&x = picked[i];

		printf("  %zu. %s  --  %s\n", i + 1, x.row.display.c_str(),
			titles[i].c_str());
		printf("     Download: %s (%s)   Expected fit: %s%s\n",
			membrane_human_size(x.variant->size_bytes).c_str(),
			x.variant->quant.c_str(), x.fit == "good"
			? membrane_ui_paint(MEMBRANE_UI_OK, "good").c_str()
			: membrane_ui_paint(MEMBRANE_UI_WARN, x.fit).c_str(),
			x.row.installed ? "   (already installed)" : "");
		printf("     Evidence: %s   License: %s\n", tested_label(
				x.row.tested_status).c_str(), x.row.license.c_str());
		printf("     %s\n", x.row.purpose.c_str());
	}
	printf("\nHow to read this: \"Expected fit\" compares the file size with "
		"your measured available RAM (an estimate; the KV cache is not "
		"included). Within a category the suggestion is the validated model "
		"first, then the largest that fits comfortably -- a stated rule, not "
		"a quality ranking.\n");
	printf("\nNext:\n  membrane use NAME             install and select "
		"one\n  membrane plan NAME            see the exact plan first\n");
	return (MEMBRANE_EXIT_SUCCESS);
}

int	membrane_models_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json)
{
	if (args.empty())
		return (cmd_installed(want_json));
	if (args.size() == 1 && (args[0] == "available" || args[0] == "catalog"))
		return (cmd_available(want_json));
	if (args.size() == 1 && args[0] == "recommend")
		return (cmd_recommend(want_json));
	print_err(want_json, "CLI_ERROR", "usage: membrane models "
		"[available | recommend] [--json]");
	return (MEMBRANE_EXIT_CLI_ERROR);
}
