#include "stop_sequence.h"

bool	membrane_stop_find_match(const std::string &acc,
			const std::vector<std::string> &stops, size_t *out_pos)
{
	bool	found = false;
	size_t	best = std::string::npos;

	for (const auto &s : stops)
	{
		if (s.empty())
			continue ;
		size_t	pos = acc.find(s);

		if (pos != std::string::npos && (!found || pos < best))
		{
			found = true;
			best = pos;
		}
	}
	if (found)
		*out_pos = best;
	return (found);
}

size_t	membrane_stop_pending_suffix_len(const std::string &acc,
			const std::vector<std::string> &stops)
{
	size_t	longest = 0;

	for (const auto &s : stops)
	{
		if (s.empty())
			continue ;
		/* Only a PROPER prefix can still be completed by a later token;
		 * s.size() itself would already be a full match, which
		 * membrane_stop_find_match() reports instead. Longest overlap
		 * first, so the first hit for this stop string is its best. */
		size_t	max_k = s.size() - 1;

		if (max_k > acc.size())
			max_k = acc.size();
		for (size_t k = max_k; k > longest; --k)
		{
			if (acc.compare(acc.size() - k, k, s, 0, k) == 0)
			{
				longest = k;
				break ;
			}
		}
	}
	return (longest);
}
