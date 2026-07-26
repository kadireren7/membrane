#ifndef MEMBRANE_WSSIM_CONFIG_H
#define MEMBRANE_WSSIM_CONFIG_H

#include "sim_config.h"

/*
 * Phase 6.2 calibration constants, layered on top of Phase 6.1's
 * sim_config.h (reused unmodified: CXL link latency/bandwidth, the
 * near-memory quant/dequant pipeline rate, Q8/Q4 compression ratios).
 * Everything new here is for the working-set/hot-cache/prefetch
 * machinery this phase adds -- labeled ASSUMED like Phase 6.1's own
 * device-local numbers, since no real CXL/near-memory hardware exists
 * to measure block-metadata-SRAM or decompressed-hot-cache access
 * time on.
 */
namespace wssim
{

/* ---- ASSUMED: on-device block metadata SRAM lookup (component 3 of
 * the near-memory pipeline, docs/phase6-cxl-near-memory.md section
 * 10) -- small associative lookup, SRAM-class access time. ---- */
constexpr double METADATA_LOOKUP_NS_PER_BLOCK = 5.0;

/* ---- ASSUMED: host/GPU-resident decompressed hot-cache lookup
 * (component 6) -- on-chip cache/SRAM-class access, faster than the
 * device-local metadata SRAM above only because it never crosses the
 * CXL link at all. ---- */
constexpr double HOTCACHE_LOOKUP_NS_PER_BLOCK = 2.0;

/* REAL, reused from sim_config.h's Phase 6.1 pipeline-count
 * sensitivity study default (docs/phase6-cxl-near-memory.md section
 * 7's "8 (default)" row). */
constexpr int	DEFAULT_QUANT_PIPELINES = 8;

}	/* namespace wssim */

#endif
