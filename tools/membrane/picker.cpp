#include "picker.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>

#include "ui.h"

/* See picker.h's top comment. */

static std::string	lower(const std::string &s)
{
	std::string	r;

	for (char c : s)
		r.push_back((char)std::tolower((unsigned char)c));
	return (r);
}

static std::string	trim(const std::string &s)
{
	size_t	a = s.find_first_not_of(" \t\r\n");

	if (a == std::string::npos)
		return ("");
	return (s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1));
}

static void	render(const std::vector<membrane_picker_item_t> &items,
				const std::vector<size_t> &view,
				const membrane_picker_opts_t &o, size_t page,
				const std::string &filter, std::ostream &out)
{
	size_t	pages = std::max((size_t)1, (view.size() + o.page_size - 1)
			/ o.page_size);
	size_t	begin = page * o.page_size;
	size_t	end = std::min(view.size(), begin + o.page_size);

	out << "\n" << membrane_ui_paint(MEMBRANE_UI_BOLD, o.title) << " -- Page "
		<< (page + 1) << "/" << pages;
	if (!filter.empty())
		out << "  (filter: \"" << filter << "\")";
	out << "\n\n";
	if (view.empty())
		out << "  (nothing matches)\n";
	for (size_t i = begin; i < end; ++i)
	{
		const membrane_picker_item_t	&it = items[view[i]];

		out << "  " << membrane_ui_paint(MEMBRANE_UI_INFO,
			std::to_string(i + 1) + ".") << " " << it.label << "\n";
		if (!it.detail.empty())
			out << "     " << membrane_ui_paint(MEMBRANE_UI_DIM, it.detail)
				<< "\n";
	}
	out << "\n  [" << (view.empty() ? std::string("-")
		: "1-" + std::to_string(view.size())) << "] select";
	if (page + 1 < pages)
		out << "   n next";
	if (page > 0)
		out << "   p prev";
	if (o.allow_search)
		out << "   / search";
	out << "   q quit\n> ";
	out.flush();
}

membrane_picker_result_t	membrane_picker_run(
								const std::vector<membrane_picker_item_t> &items,
								const membrane_picker_opts_t &opts,
								std::istream &in, std::ostream &out)
{
	membrane_picker_result_t	res;
	membrane_picker_opts_t		o = opts;
	std::vector<size_t>			view;
	std::string					filter;
	size_t						page = 0;
	size_t						bad = 0;
	std::string					line;

	if (o.page_size == 0)
		o.page_size = 8;
	for (size_t i = 0; i < items.size(); ++i)
		view.push_back(i);
	while (true)
	{
		size_t	pages = std::max((size_t)1, (view.size() + o.page_size - 1)
				/ o.page_size);

		if (page >= pages)
			page = pages - 1;
		render(items, view, o, page, filter, out);
		if (!std::getline(in, line))
		{
			out << "\n";
			return (res);				/* EOF => quit */
		}
		std::string	cmd = trim(line);
		std::string	lc = lower(cmd);

		if (lc == "q" || lc == "quit" || lc == "exit")
			return (res);
		if (lc == "n" || lc == "next")
		{
			if (page + 1 < pages)
				page++;
			bad = 0;
			continue ;
		}
		if (lc == "p" || lc == "prev" || lc == "previous")
		{
			if (page > 0)
				page--;
			bad = 0;
			continue ;
		}
		if (o.allow_search && !cmd.empty() && cmd[0] == '/')
		{
			filter = trim(cmd.substr(1));
			view.clear();
			for (size_t i = 0; i < items.size(); ++i)
				if (filter.empty() || lower(items[i].label + " "
						+ items[i].detail).find(lower(filter))
					!= std::string::npos)
					view.push_back(i);
			page = 0;
			bad = 0;
			continue ;
		}
		if (!cmd.empty() && std::all_of(cmd.begin(), cmd.end(),
				[](char c) { return (std::isdigit((unsigned char)c) != 0); })
			&& cmd.size() < 9)
		{
			size_t	n = (size_t)std::stoul(cmd);

			if (n >= 1 && n <= view.size())
			{
				res.status = MEMBRANE_PICK_SELECTED;
				res.index = view[n - 1];
				return (res);
			}
		}
		out << "  " << membrane_ui_paint(MEMBRANE_UI_WARN, "Please enter a "
			"number from the list, n, p, /search or q.") << "\n";
		if (++bad >= o.max_invalid)
			return (res);
	}
}
