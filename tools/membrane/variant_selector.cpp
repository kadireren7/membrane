#include "variant_selector.h"
#include "host_memory_guard.h"

const membrane_catalog_variant_t	*membrane_select_variant(
			const membrane_catalog_family_t &family,
			const membrane_variant_selector_input_t &hw,
			std::vector<membrane_variant_fit_t> *out_all_considered)
{
	if (out_all_considered != NULL)
		out_all_considered->clear();
	const membrane_catalog_variant_t	*best = NULL;

	for (const auto &v : family.variants)
	{
		membrane_host_guard_request_t	req;

		req.host_total_bytes = hw.host_total_bytes;
		req.host_available_bytes = hw.host_available_bytes;
		req.host_available_known = hw.host_available_known ? 1 : 0;
		req.host_weight_bytes = v.size_bytes;
		req.host_kv_bytes = 0;	/* no context chosen yet -- see this
								 * module's own header comment */

		membrane_host_guard_result_t	res;
		bool		fits = membrane_host_memory_guard_resolve(&req, &res) != 0;

		if (out_all_considered != NULL)
		{
			membrane_variant_fit_t	fit;

			fit.quant = v.quant;
			fit.fits = fits;
			fit.reason_code = res.reason_code;
			fit.reason = res.reason;
			out_all_considered->push_back(fit);
		}
		if (fits && (best == NULL || v.size_bytes > best->size_bytes))
			best = &v;
	}
	return (best);
}
