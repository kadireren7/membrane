#include "model_views.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <set>

#include <nlohmann/json.hpp>

#include "model_catalog.h"
#include "registry_core.h"
#include "server_config.h"
#include "status_client.h"

using json = nlohmann::json;

/* See model_views.h's top comment. */

std::string	membrane_human_size(uint64_t bytes)
{
	char	buf[32];

	if (bytes >= 1000ull * 1000 * 1000)
		snprintf(buf, sizeof(buf), "%.1f GB", (double)bytes / 1e9);
	else if (bytes >= 1000ull * 1000)
		snprintf(buf, sizeof(buf), "%.0f MB", (double)bytes / 1e6);
	else
		snprintf(buf, sizeof(buf), "%.0f KB", (double)bytes / 1e3);
	return (buf);
}

std::string	membrane_quant_from_filename(const std::string &filename)
{
	static const char	*known[] = {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
		"Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
		"Q5_K_M", "Q6_K", "Q8_0", "BF16", "F16", "F32", "IQ4_XS", "IQ3_M",
		"IQ2_M"};
	std::string			up;

	for (char c : filename)
		up.push_back((char)std::toupper((unsigned char)c));
	/* longest match first so Q4_K_M is not read as Q4_K */
	std::string	best;

	for (const char *k : known)
	{
		std::string	key = k;
		size_t		pos = up.find(key);

		while (pos != std::string::npos)
		{
			bool	left = pos == 0 || !std::isalnum((unsigned char)up[pos - 1]);
			bool	right = pos + key.size() >= up.size()
				|| !std::isalnum((unsigned char)up[pos + key.size()]);

			if (left && right && key.size() > best.size())
				best = key;
			pos = up.find(key, pos + 1);
		}
	}
	return (best);
}

static std::string	lower(const std::string &s)
{
	std::string	r;

	for (char c : s)
		r.push_back((char)std::tolower((unsigned char)c));
	return (r);
}

static void	fill_from_family(membrane_model_row_t *r,
				const membrane_catalog_family_t &f)
{
	r->display = f.display_name;
	r->params = f.parameter_count;
	r->arch = f.arch;
	r->tags = f.tags;
	r->purpose = f.purpose;
	r->tested_status = f.tested_status;
	r->license = f.license;
	r->in_catalog = true;
}

static const membrane_catalog_family_t	*family_for_entry(
		const membrane_catalog_t &cat, const membrane_registry_entry_t &e,
		std::string *quant)
{
	for (const auto &f : cat.families)
	{
		for (const auto &v : f.variants)
			if (lower(v.filename) == lower(e.basename))
			{
				*quant = v.quant;
				return (&f);
			}
	}
	for (const auto &f : cat.families)
		if (lower(f.name) == lower(e.name))
			return (&f);
	return (NULL);
}

void	membrane_model_rows_installed(std::vector<membrane_model_row_t> *out,
			bool probe_server)
{
	std::string					path = membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	err;
	membrane_catalog_t			cat = membrane_catalog_load();
	membrane_server_config_t	cfg = membrane_server_config_defaults();
	std::string					cpath = membrane_server_config_resolve_path();
	std::set<std::string>		loaded;
	bool						probed = false;

	out->clear();
	if (path.empty() || !membrane_registry_load(path, &reg, &err))
		return ;
	if (!cpath.empty())
	{
		membrane_server_config_error_t	cerr;

		membrane_server_config_load(cpath, &cfg, &cerr);
	}
	if (probe_server)
	{
		json	status;

		if (membrane_fetch_server_status(cfg.listen_address, cfg.port, &status))
		{
			probed = true;
			if (status.contains("resident_models")
				&& status["resident_models"].is_array())
				for (const auto &m : status["resident_models"])
					loaded.insert(m.value("model", std::string()));
		}
	}
	for (const auto &e : reg.entries)
	{
		membrane_model_row_t	r;
		std::string				quant;
		const membrane_catalog_family_t	*f = family_for_entry(cat, e, &quant);

		r.id = e.name;
		r.display = e.name;
		r.installed = true;
		r.path = e.path;
		r.size_bytes = e.file_size_bytes;
		r.arch = e.arch_name;
		r.quant = quant.empty() ? membrane_quant_from_filename(e.basename)
			: quant;
		if (f != NULL)
			fill_from_family(&r, *f);
		r.is_default = !cfg.default_model.empty() && cfg.default_model == e.name;
		r.resident_known = probed;
		r.resident = loaded.count(e.name) > 0;
		out->push_back(r);
	}
	std::sort(out->begin(), out->end(), [](const membrane_model_row_t &a,
			const membrane_model_row_t &b)
	{
		if (lower(a.display) != lower(b.display))
			return (lower(a.display) < lower(b.display));
		return (a.id < b.id);
	});
}

static double	params_value(const std::string &p)
{
	double	v = atof(p.c_str());

	if (!p.empty() && (p.back() == 'M' || p.back() == 'm'))
		v /= 1000.0;
	return (v);
}

void	membrane_model_rows_catalog(std::vector<membrane_model_row_t> *out)
{
	membrane_catalog_t				cat = membrane_catalog_load();
	std::vector<membrane_model_row_t>	inst;

	out->clear();
	membrane_model_rows_installed(&inst, false);
	for (const auto &f : cat.families)
	{
		membrane_model_row_t	r;

		r.id = f.name;
		fill_from_family(&r, f);
		for (const auto &v : f.variants)
			if (r.size_bytes == 0 || v.size_bytes < r.size_bytes)
			{
				r.size_bytes = v.size_bytes;
				r.quant = v.quant;
			}
		for (const auto &i : inst)
		{
			bool	match = i.id == f.name;

			for (const auto &v : f.variants)
				if (lower(v.filename) == lower(std::string(
						i.path.substr(i.path.rfind('/') == std::string::npos
							? 0 : i.path.rfind('/') + 1))))
					match = true;
			if (match)
			{
				r.installed = true;
				r.path = i.path;
				r.is_default = i.is_default;
			}
		}
		out->push_back(r);
	}
	std::sort(out->begin(), out->end(), [](const membrane_model_row_t &a,
			const membrane_model_row_t &b)
	{
		if (params_value(a.params) != params_value(b.params))
			return (params_value(a.params) < params_value(b.params));
		return (a.id < b.id);
	});
}

std::vector<membrane_model_cand_t>	membrane_model_candidates(
			bool include_catalog)
{
	std::vector<membrane_model_cand_t>	v;
	std::vector<membrane_model_row_t>	inst;
	std::set<std::string>				seen;

	membrane_model_rows_installed(&inst, false);
	for (const auto &r : inst)
	{
		membrane_model_cand_t	c;

		c.id = r.id;
		c.display = r.display == r.id ? "" : r.display;
		c.installed = true;
		v.push_back(c);
		seen.insert(r.id);
	}
	if (include_catalog)
	{
		membrane_catalog_t	cat = membrane_catalog_load();

		for (const auto &f : cat.families)
		{
			if (seen.count(f.name))
			{
				for (auto &c : v)
					if (c.id == f.name)
					{
						c.display = f.display_name;
						c.aliases = f.aliases;
						c.catalog = true;
					}
				continue ;
			}
			membrane_model_cand_t	c;

			c.id = f.name;
			c.display = f.display_name;
			c.aliases = f.aliases;
			c.catalog = true;
			v.push_back(c);
		}
	}
	return (v);
}

std::string	membrane_model_detail_line(const membrane_model_row_t &r)
{
	std::string	s;
	auto		add = [&](const std::string &x)
	{
		if (x.empty())
			return ;
		s += (s.empty() ? "" : " . ") + x;
	};

	for (const auto &t : r.tags)
		add(t);
	if (r.size_bytes > 0)
		add(membrane_human_size(r.size_bytes));
	add(r.quant);
	if (r.resident)
		add("loaded");
	if (r.is_default)
		add("default");
	return (s);
}
