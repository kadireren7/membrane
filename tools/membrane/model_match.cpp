#include "model_match.h"

#include <algorithm>
#include <cctype>

/* See model_match.h's top comment. Pure. */

const char	*membrane_match_kind_name(membrane_match_kind_t k)
{
	switch (k)
	{
		case MEMBRANE_MATCH_EXACT: return ("exact");
		case MEMBRANE_MATCH_PREFIX: return ("prefix");
		case MEMBRANE_MATCH_SUBSTRING: return ("substring");
		case MEMBRANE_MATCH_FUZZY: return ("fuzzy");
		default: return ("none");
	}
}

static std::string	lower(const std::string &s)
{
	std::string	r;

	for (char c : s)
		r.push_back((char)std::tolower((unsigned char)c));
	return (r);
}

std::string	membrane_model_normalize(const std::string &name)
{
	std::string	s = lower(name);
	std::string	r;

	if (s.size() > 7 && s.compare(s.size() - 7, 7, ":latest") == 0)
		s.erase(s.size() - 7);
	for (char c : s)
		if (c != '-' && c != '_' && c != ':' && c != '/' && c != ' ')
			r.push_back(c);
	return (r);
}

static std::vector<std::string>	names_of(const membrane_model_cand_t &c)
{
	std::vector<std::string>	v;

	v.push_back(c.id);
	if (!c.display.empty())
		v.push_back(c.display);
	for (const auto &a : c.aliases)
		v.push_back(a);
	return (v);
}

static size_t	edit_distance(const std::string &a, const std::string &b)
{
	std::vector<size_t>	prev(b.size() + 1);
	std::vector<size_t>	cur(b.size() + 1);

	for (size_t j = 0; j <= b.size(); ++j)
		prev[j] = j;
	for (size_t i = 1; i <= a.size(); ++i)
	{
		cur[0] = i;
		for (size_t j = 1; j <= b.size(); ++j)
		{
			size_t	sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);

			cur[j] = std::min(std::min(prev[j] + 1, cur[j - 1] + 1), sub);
		}
		prev = cur;
	}
	return (prev[b.size()]);
}

static std::vector<std::string>	tokens_of(const std::string &q)
{
	std::vector<std::string>	t;
	std::string					cur;

	for (char c : q)
	{
		if (c == ' ' || c == '\t')
		{
			if (!cur.empty())
				t.push_back(membrane_model_normalize(cur));
			cur.clear();
		}
		else
			cur.push_back(c);
	}
	if (!cur.empty())
		t.push_back(membrane_model_normalize(cur));
	return (t);
}

static void	sort_hits(std::vector<size_t> *hits,
				const std::vector<membrane_model_cand_t> &cands)
{
	std::sort(hits->begin(), hits->end(), [&](size_t a, size_t b)
	{
		if (cands[a].installed != cands[b].installed)
			return (cands[a].installed);
		return (cands[a].id < cands[b].id);
	});
	hits->erase(std::unique(hits->begin(), hits->end()), hits->end());
}

membrane_match_t	membrane_model_match(const std::string &query,
						const std::vector<membrane_model_cand_t> &cands)
{
	membrane_match_t			m;
	std::string					q = lower(query);
	std::string					qn = membrane_model_normalize(query);
	std::vector<std::string>	toks = tokens_of(query);

	if (qn.empty())
		return (m);
	/* 1. exact */
	for (size_t i = 0; i < cands.size(); ++i)
		for (const auto &n : names_of(cands[i]))
			if (lower(n) == q || membrane_model_normalize(n) == qn)
			{
				m.hits.push_back(i);
				break ;
			}
	if (!m.hits.empty())
	{
		m.kind = MEMBRANE_MATCH_EXACT;
		sort_hits(&m.hits, cands);
		return (m);
	}
	/* 2. prefix */
	for (size_t i = 0; i < cands.size(); ++i)
		for (const auto &n : names_of(cands[i]))
			if (membrane_model_normalize(n).compare(0, qn.size(), qn) == 0)
			{
				m.hits.push_back(i);
				break ;
			}
	if (!m.hits.empty())
	{
		m.kind = MEMBRANE_MATCH_PREFIX;
		sort_hits(&m.hits, cands);
		return (m);
	}
	/* 3. substring (whole query, or every token) */
	for (size_t i = 0; i < cands.size(); ++i)
		for (const auto &n : names_of(cands[i]))
		{
			std::string	nn = membrane_model_normalize(n);
			bool		all = !toks.empty();

			for (const auto &t : toks)
				if (!t.empty() && nn.find(t) == std::string::npos)
					all = false;
			if (nn.find(qn) != std::string::npos || all)
			{
				m.hits.push_back(i);
				break ;
			}
		}
	if (!m.hits.empty())
	{
		m.kind = MEMBRANE_MATCH_SUBSTRING;
		sort_hits(&m.hits, cands);
		return (m);
	}
	/* 4. fuzzy */
	if (qn.size() < 4)
		return (m);
	size_t	limit = std::max((size_t)1, std::min((size_t)3, qn.size() / 4));
	size_t	best = limit + 1;

	for (size_t i = 0; i < cands.size(); ++i)
		for (const auto &n : names_of(cands[i]))
		{
			size_t	d = edit_distance(qn, membrane_model_normalize(n));

			if (d < best)
				best = d;
		}
	if (best > limit)
		return (m);
	for (size_t i = 0; i < cands.size(); ++i)
		for (const auto &n : names_of(cands[i]))
			if (edit_distance(qn, membrane_model_normalize(n)) == best)
			{
				m.hits.push_back(i);
				break ;
			}
	m.kind = MEMBRANE_MATCH_FUZZY;
	m.distance = (int)best;
	sort_hits(&m.hits, cands);
	return (m);
}
