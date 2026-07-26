#include <algorithm>
#include <vector>

#include "hotcache.h"

namespace wssim
{

const char	*eviction_policy_name(eviction_policy_t p)
{
	switch (p)
	{
	case eviction_policy_t::LRU: return ("lru");
	case eviction_policy_t::LFU: return ("lfu");
	case eviction_policy_t::ATTENTION_SCORE_AWARE: return ("attention-score-aware");
	case eviction_policy_t::SEGMENTED_LRU: return ("segmented-lru");
	default: return ("unknown");
	}
}

hot_cache_t::hot_cache_t(uint64_t capacity_bytes, eviction_policy_t policy)
	: m_capacity(capacity_bytes), m_policy(policy), m_bytes_used(0),
	  m_seq(0), m_protected_bytes_used(0)
{
}

bool	hot_cache_t::contains(const cache_key_t &k) const
{
	return (m_entries.find(k) != m_entries.end());
}

/* Ascending: lower value is evicted first. LRU/segmented use a
 * monotonically increasing sequence number stashed in `score` via
 * touch_hit/insert's caller-visible bytes-of-recency trick below --
 * kept simple by storing recency directly as the metric for those two
 * policies (see the m_seq usage in touch_hit/insert). */
double	hot_cache_t::metric_for(const cache_key_t &k, const entry_t &e) const
{
	(void)k;
	if (m_policy == eviction_policy_t::LFU)
		return ((double)e.freq);
	if (m_policy == eviction_policy_t::ATTENTION_SCORE_AWARE)
		return (e.score);
	/* LRU and SEGMENTED_LRU both rank by recency; segmented additionally
	 * always evicts probationary entries before any protected one
	 * (encoded as a large additive offset for protected entries). */
	double	recency = e.score;	/* recency sequence stored in `score` */
	if (m_policy == eviction_policy_t::SEGMENTED_LRU && e.protected_segment)
		return (recency + 1e18);
	return (recency);
}

void	hot_cache_t::evict_until_fits(uint64_t incoming_bytes)
{
	if (m_bytes_used + incoming_bytes <= m_capacity || m_entries.empty())
		return ;
	/* Sorted once per call, then walked in ascending-priority order --
	 * evicting a victim never changes any OTHER surviving entry's
	 * metric, so a single sort suffices instead of the O(evictions *
	 * n log n) blowup a re-sort-per-victim loop would cause (a real
	 * performance bug caught during this phase's own development: an
	 * earlier version re-ranked the whole map on every single
	 * eviction, which made FULL-policy/small-hot-cache scenarios
	 * effectively hang). */
	std::vector<std::pair<double, cache_key_t>>	ranked;

	ranked.reserve(m_entries.size());
	for (const auto &kv : m_entries)
		ranked.emplace_back(metric_for(kv.first, kv.second), kv.first);
	std::sort(ranked.begin(), ranked.end(),
		[](const auto &a, const auto &b) { return (a.first < b.first); });
	size_t	i = 0;
	while (m_bytes_used + incoming_bytes > m_capacity && i < ranked.size())
	{
		const cache_key_t	&victim = ranked[i].second;
		auto	it = m_entries.find(victim);
		if (it != m_entries.end())
		{
			m_bytes_used -= it->second.bytes;
			if (it->second.protected_segment)
				m_protected_bytes_used -= it->second.bytes;
			m_entries.erase(it);
		}
		i++;
	}
}

void	hot_cache_t::touch_hit(const cache_key_t &k, double score)
{
	auto	it = m_entries.find(k);
	if (it == m_entries.end())
		return ;
	it->second.freq++;
	m_seq++;
	if (m_policy == eviction_policy_t::ATTENTION_SCORE_AWARE)
		it->second.score = score;
	else
		it->second.score = (double)m_seq;
	if (m_policy == eviction_policy_t::SEGMENTED_LRU
			&& !it->second.protected_segment)
	{
		uint64_t	protected_cap = (m_capacity * 4) / 5;	/* 80/20 split */
		if (m_protected_bytes_used + it->second.bytes <= protected_cap)
		{
			it->second.protected_segment = true;
			m_protected_bytes_used += it->second.bytes;
		}
	}
}

uint64_t	hot_cache_t::insert(const cache_key_t &k, uint64_t bytes,
					double score)
{
	if (m_entries.find(k) != m_entries.end())
	{
		touch_hit(k, score);
		return (0);
	}
	uint64_t	before = m_bytes_used;
	evict_until_fits(bytes);
	uint64_t	evicted = 0;
	if (m_bytes_used < before)
		evicted = before - m_bytes_used;
	m_seq++;
	entry_t	e;
	e.bytes = bytes;
	e.freq = 1;
	e.score = (m_policy == eviction_policy_t::ATTENTION_SCORE_AWARE)
		? score : (double)m_seq;
	e.protected_segment = false;
	if (bytes <= m_capacity)
	{
		m_entries[k] = e;
		m_bytes_used += bytes;
	}
	return (evicted);
}

}	/* namespace wssim */
