#include <algorithm>
#include <cstdio>
#include <map>

#include "attn_workload.h"

namespace wssim
{

static uint32_t	xorshift32(uint32_t *state)
{
	uint32_t	x;

	x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return (x);
}

bool	load_attn_trace(const std::string &path, attn_trace_t *out)
{
	FILE						*f;
	membrane_attntrace_header_t	h;
	size_t						n;

	f = fopen(path.c_str(), "rb");
	if (f == NULL)
		return (false);
	if (membrane_attntrace_read_header(f, &h) != MEMBRANE_OK)
	{
		fclose(f);
		return (false);
	}
	n = membrane_attntrace_entry_count(&h);
	out->entries.resize(n);
	if (membrane_attntrace_read_entries(f, &h, out->entries.data())
			!= MEMBRANE_OK)
	{
		fclose(f);
		return (false);
	}
	fclose(f);
	out->model = h.model;
	out->is_real_capture = (h.source == MEMBRANE_ATTNTRACE_SOURCE_REAL_CAPTURE);
	out->n_layer = h.n_layer;
	out->n_head = h.n_head;
	out->block_size_tokens = h.block_size_tokens;
	out->prompt_len = h.prompt_len;
	out->step_count = h.step_count;
	out->top_k = h.top_k;
	return (true);
}

static void	top_k_from_map(const std::map<uint32_t, double> &acc,
				uint32_t top_k, std::vector<membrane_attntrace_entry_t> &dst,
				size_t base)
{
	std::vector<std::pair<uint32_t, double>>	v(acc.begin(), acc.end());

	std::sort(v.begin(), v.end(),
		[](const std::pair<uint32_t, double> &a,
			const std::pair<uint32_t, double> &b)
		{ return (a.second > b.second); });
	for (uint32_t k = 0; k < top_k; k++)
	{
		if (k < v.size())
		{
			dst[base + k].block_id = v[k].first;
			dst[base + k].score = (float)v[k].second;
		}
		else
		{
			dst[base + k].block_id = UINT32_MAX;
			dst[base + k].score = 0.0f;
		}
	}
}

attn_trace_t	regroup_to_block_size(const attn_trace_t &src,
						uint32_t target_block_size_tokens)
{
	attn_trace_t	out;
	uint32_t		b0;
	uint32_t		coarsen_ratio;
	uint32_t		split_ratio;

	out = src;
	out.block_size_tokens = target_block_size_tokens;
	if (target_block_size_tokens == src.block_size_tokens)
		return (out);
	b0 = src.block_size_tokens;
	out.entries.assign(src.entries.size(),
		membrane_attntrace_entry_t{UINT32_MAX, 0.0f});
	coarsen_ratio = 0;
	split_ratio = 0;
	if (target_block_size_tokens > b0
			&& target_block_size_tokens % b0 == 0)
		coarsen_ratio = target_block_size_tokens / b0;
	else if (b0 > target_block_size_tokens
			&& b0 % target_block_size_tokens == 0)
		split_ratio = b0 / target_block_size_tokens;
	for (uint32_t step = 0; step < src.step_count; step++)
	{
		for (uint32_t layer = 0; layer < src.n_layer; layer++)
		{
			for (uint32_t head = 0; head < src.n_head; head++)
			{
				const membrane_attntrace_entry_t	*native
					= src.at(step, layer, head);
				std::map<uint32_t, double>			acc;

				for (uint32_t k = 0; k < src.top_k; k++)
				{
					if (native[k].block_id == UINT32_MAX)
						continue ;
					if (coarsen_ratio > 0)
						acc[native[k].block_id / coarsen_ratio]
							+= native[k].score;
					else if (split_ratio > 0)
					{
						double	share = (double)native[k].score
							/ split_ratio;
						for (uint32_t j = 0; j < split_ratio; j++)
							acc[native[k].block_id * split_ratio + j]
								+= share;
					}
				}
				size_t	base = (((size_t)step * src.n_layer + layer)
						* src.n_head + head) * src.top_k;
				top_k_from_map(acc, src.top_k, out.entries, base);
			}
		}
	}
	return (out);
}

attn_trace_t	extend_synthetic(const attn_trace_t &src,
						uint32_t target_step_count, uint32_t seed)
{
	attn_trace_t	out;
	uint32_t		blocks_per_cycle;
	uint32_t		state;

	out = src;
	out.is_real_capture = false;
	out.step_count = target_step_count;
	blocks_per_cycle = (src.prompt_len + src.step_count
			+ src.block_size_tokens - 1) / src.block_size_tokens;
	out.entries.assign((size_t)target_step_count * src.n_layer
			* src.n_head * src.top_k,
		membrane_attntrace_entry_t{UINT32_MAX, 0.0f});
	state = seed | 1u;
	for (uint32_t step = 0; step < target_step_count; step++)
	{
		uint32_t	cycle = step / src.step_count;
		uint32_t	src_step = step % src.step_count;
		uint32_t	shift = cycle * blocks_per_cycle;

		for (uint32_t layer = 0; layer < src.n_layer; layer++)
		{
			for (uint32_t head = 0; head < src.n_head; head++)
			{
				const membrane_attntrace_entry_t	*native
					= src.at(src_step, layer, head);
				size_t	base = (((size_t)step * src.n_layer + layer)
						* src.n_head + head) * src.top_k;

				for (uint32_t k = 0; k < src.top_k; k++)
				{
					if (native[k].block_id == UINT32_MAX)
					{
						out.entries[base + k].block_id = UINT32_MAX;
						out.entries[base + k].score = 0.0f;
						continue ;
					}
					/* block 0 is the real, repeatedly-observed
					 * attention-sink block (see docs) -- kept fixed
					 * across cycles rather than shifted forward,
					 * matching the real captured behavior instead of
					 * diluting it away at longer synthetic contexts. */
					uint32_t	bid = native[k].block_id;
					if (bid != 0)
						bid += shift;
					double	jitter = 1.0 + (((double)(xorshift32(&state)
							% 2001) - 1000.0) / 1000.0) * 0.03;
					out.entries[base + k].block_id = bid;
					out.entries[base + k].score
						= (float)((double)native[k].score * jitter);
				}
			}
		}
	}
	return (out);
}

}	/* namespace wssim */
