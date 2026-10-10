#include "model_resolve.h"

#include <iostream>
#include <sstream>

#include "cli_shared.h"
#include "picker.h"
#include "ui.h"

/* See model_resolve.h's top comment. */

membrane_resolve_io_t	membrane_resolve_io_for_terminal(void)
{
	membrane_resolve_io_t	io;

	io.interactive = membrane_cli_is_interactive();
	return (io);
}

static std::string	label_of(const membrane_model_cand_t &c)
{
	std::string	l = c.display.empty() ? c.id : c.display;

	if (!c.display.empty() && c.display != c.id)
		l += " (" + c.id + ")";
	if (c.installed)
		l += " [installed]";
	else if (c.catalog)
		l += " [available to install]";
	return (l);
}

static bool	ask_yes(std::istream &in, std::ostream &out,
				const std::string &prompt)
{
	std::string	line;

	out << prompt << " [Y/n] ";
	out.flush();
	if (!std::getline(in, line))
		return (false);
	size_t	a = line.find_first_not_of(" \t\r\n");

	if (a == std::string::npos)
		return (true);
	char	c = (char)std::tolower((unsigned char)line[a]);

	return (c == 'y');
}

membrane_resolve_result_t	membrane_model_resolve(const std::string &query,
		const std::vector<membrane_model_cand_t> &cands,
		const std::string &hint, const membrane_resolve_io_t &io)
{
	membrane_resolve_result_t	r;
	std::istream				&in = io.in ? *io.in : std::cin;
	std::ostream				&out = io.out ? *io.out : std::cout;
	membrane_match_t			m = membrane_model_match(query, cands);

	r.kind = m.kind;
	if (m.kind == MEMBRANE_MATCH_NONE)
	{
		r.status = MEMBRANE_RESOLVE_NOT_FOUND;
		r.message = "no model matches '" + query + "'";
		return (r);
	}
	if (m.kind == MEMBRANE_MATCH_EXACT && m.hits.size() == 1)
	{
		r.status = MEMBRANE_RESOLVE_EXACT;
		r.index = m.hits[0];
		r.id = cands[r.index].id;
		r.installed = cands[r.index].installed;
		return (r);
	}
	/* a suggestion: never chosen silently */
	std::ostringstream	msg;

	if (m.hits.size() == 1)
		msg << "There is no model named \"" << query << "\". Did you mean:\n";
	else
		msg << "Multiple models match \"" << query << "\":\n";
	for (size_t i = 0; i < m.hits.size() && i < 8; ++i)
		msg << "  - " << label_of(cands[m.hits[i]]) << "\n";
	if (m.hits.size() > 8)
		msg << "  ... and " << (m.hits.size() - 8) << " more\n";
	if (!io.interactive)
	{
		msg << "Not choosing one automatically (no terminal to confirm on). "
			"Re-run with the exact name, e.g. `" << hint << " "
			<< cands[m.hits[0]].id << "`.";
		r.status = MEMBRANE_RESOLVE_NEEDS_CHOICE;
		r.message = msg.str();
		return (r);
	}
	size_t	chosen = 0;

	if (m.hits.size() == 1)
	{
		const membrane_model_cand_t	&c = cands[m.hits[0]];

		if (!ask_yes(in, out, "Did you mean " + membrane_ui_paint(
				MEMBRANE_UI_INFO, "\"" + (c.display.empty() ? c.id : c.display)
				+ "\"") + (c.installed ? " (installed)" : " (not installed)")
				+ "?"))
		{
			r.status = MEMBRANE_RESOLVE_DECLINED;
			r.message = "cancelled";
			return (r);
		}
	}
	else
	{
		std::vector<membrane_picker_item_t>	items;
		membrane_picker_opts_t				po;

		for (size_t h : m.hits)
			items.push_back({label_of(cands[h]), ""});
		po.title = "Models matching \"" + query + "\"";
		membrane_picker_result_t	pr = membrane_picker_run(items, po, in, out);

		if (pr.status != MEMBRANE_PICK_SELECTED)
		{
			r.status = MEMBRANE_RESOLVE_DECLINED;
			r.message = "cancelled";
			return (r);
		}
		chosen = pr.index;
	}
	r.status = MEMBRANE_RESOLVE_CONFIRMED;
	r.index = m.hits[chosen];
	r.id = cands[r.index].id;
	r.installed = cands[r.index].installed;
	return (r);
}

int	membrane_model_pick(const std::string &title,
		const std::vector<membrane_model_row_t> &rows,
		const membrane_resolve_io_t &io)
{
	std::istream						&in = io.in ? *io.in : std::cin;
	std::ostream						&out = io.out ? *io.out : std::cout;
	std::vector<membrane_picker_item_t>	items;
	membrane_picker_opts_t				po;

	for (const auto &r : rows)
		items.push_back({r.display.empty() ? r.id : r.display,
			membrane_model_detail_line(r)});
	po.title = title;
	membrane_picker_result_t	pr = membrane_picker_run(items, po, in, out);

	return (pr.status == MEMBRANE_PICK_SELECTED ? (int)pr.index : -1);
}
