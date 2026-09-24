#include "plan_cmd.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <nlohmann/json.hpp>

#include <ggml.h>

#include "registry_core.h"
#include "model_catalog.h"
#include "variant_selector.h"
#include "gpu_device.h"
#include "runtime_session.h"
#include "kv_store_telemetry.h"
#include "membrane_plan.h"
#include "plan_v2_resolver.h"
#include "product_cli.h"
#include "runtime_adapter.h"
#include "runtime_plan_cmd.h"

using json = nlohmann::json;

/* See plan_cmd.h's own top comment for the full read-only contract.
 *
 * Milestone G2: this file now orchestrates the joint VARIANT/QUANT
 * dimension too (plan_v2_resolver.h) -- see docs/planner-v2-joint-
 * variant.md for the full architecture. The key change from G1: a
 * variant is no longer chosen before the context/GPU/KV joint pipeline
 * runs (variant_selector.h's pre-G2 role); every feasible candidate
 * variant is now run through the SAME real joint pipeline G1 already
 * built (or, when no real hparams exist for it, an explicitly disclosed
 * coarse fallback -- see s_variant_spec's own comment below), and the
 * SELECTED variant emerges from that, via plan_v2_resolver.h's own
 * small, fixed, documented policy. */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", code}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane plan: %s\n", message.c_str());
}

struct s_plan_opts
{
	std::string	name;
	bool		want_ctx = false;
	bool		ctx_auto = true;
	uint64_t	ctx_value = 0;
	bool		want_kv = false;
	int			precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	bool		want_gpu_layers = false;
	int32_t		gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	bool		want_quant = false;
	std::string	quant;
	std::string	runtime;	/* Milestone H3: --runtime ID ("" = none) */
};

static bool	parse_plan_opts(const std::vector<std::string> &args,
				s_plan_opts *o, std::string *err)
{
	for (size_t i = 0; i < args.size(); ++i)
	{
		const std::string	&a = args[i];

		if (a == "--ctx" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_ctx = true;
			if (v == "auto")
				o->ctx_auto = true;
			else
			{
				char	*end = NULL;
				long long	n = strtoll(v.c_str(), &end, 10);

				if (end == NULL || *end != '\0' || n <= 0)
				{
					*err = "--ctx must be 'auto' or a positive integer";
					return (false);
				}
				o->ctx_auto = false;
				o->ctx_value = (uint64_t)n;
			}
			continue ;
		}
		if (a == "--kv" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_kv = true;
			if (v == "native")
				o->precision_request = MEMBRANE_JOINT_KV_NATIVE;
			else if (v == "q8")
				o->precision_request = MEMBRANE_JOINT_KV_Q8;
			else if (v == "q5")
				o->precision_request = MEMBRANE_JOINT_KV_Q5;
			else if (v == "adaptive")
				o->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
			else
			{
				*err = "--kv must be one of native|q8|q5|adaptive";
				return (false);
			}
			continue ;
		}
		if (a == "--gpu-layers" && i + 1 < args.size())
		{
			std::string	v = args[++i];

			o->want_gpu_layers = true;
			if (v == "all")
				o->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL;
			else if (v == "auto")
				o->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
			else
			{
				char	*end = NULL;
				long long	n = strtoll(v.c_str(), &end, 10);

				if (end == NULL || *end != '\0' || n < 0)
				{
					*err = "--gpu-layers must be all|auto|N (N >= 0)";
					return (false);
				}
				o->gpu_layers_request = (int32_t)n;
			}
			continue ;
		}
		if ((a == "--quant" || a == "--variant") && i + 1 < args.size())
		{
			o->want_quant = true;
			o->quant = args[++i];
			continue ;
		}
		if (a == "--runtime" && i + 1 < args.size())
		{
			o->runtime = args[++i];
			continue ;
		}
		if (o->name.empty() && a.size() > 0 && a[0] != '-')
		{
			o->name = a;
			continue ;
		}
		*err = "unknown option '" + a + "'";
		return (false);
	}
	if (o->name.empty())
	{
		*err = "usage: membrane plan MODEL [--ctx auto|N] [--kv "
			"native|q8|q5|adaptive] [--gpu-layers all|auto|N] "
			"[--quant QUANT] [--runtime RUNTIME_ID]";
		return (false);
	}
	return (true);
}

/* Same real ggml_row_size()-based per-mode KV byte formula every other
 * real caller in this project uses (context_recommender_dryrun.cpp's
 * own real_kv_bytes(), main.cpp's native_kv_bytes()/q8_kv_bytes()/
 * q5_kv_bytes()) -- never a second, independently-derived formula.
 * Takes plain hparams rather than membrane_gpu_model_estimate_t so it
 * works identically for a real installed model AND a scaled-estimate
 * sibling variant (Milestone G2: see s_variant_hparams below). */
static uint64_t	real_kv_bytes(int32_t n_head, int32_t n_embd,
					int32_t n_head_kv, int32_t n_layer, uint64_t ctx,
					enum ggml_type type)
{
	int64_t	n_embd_gqa = (n_head > 0)
			? (int64_t)(n_embd / n_head) * n_head_kv : 0;
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)n_layer;
	b.kv_size = ctx;
	b.bytes_per_token_k = ggml_row_size(type, n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(type, n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static int	find_gpu_device_index(const membrane_gpu_device_info_t *devices,
				size_t n_devices)
{
	for (size_t i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
			return ((int)i);
	return (-1);
}

/* Best-effort, exact-match-only quant identification for an already-
 * installed model -- the registry (registry_core.h) does NOT store
 * which catalog quant a file is (Section 13 of registry_core.h's own
 * top comment: only a cheap size/mtime identity signature), so this
 * never guesses from file size alone. Returns the matched catalog
 * variant itself (Milestone G2: its real size_bytes anchors sibling
 * scaling, see scale_hparams() below), not just the quant string --
 * G1's own best_effort_installed_variant() only needed the string. */
static const membrane_catalog_variant_t	*find_installed_catalog_variant(
				const membrane_catalog_family_t *fam,
				const std::string &basename)
{
	if (fam == NULL)
		return (NULL);
	for (const auto &v : fam->variants)
		if (v.filename == basename)
			return (&v);
	return (NULL);
}

static bool	ieq(const std::string &a, const std::string &b)
{
	if (a.size() != b.size())
		return (false);
	for (size_t i = 0; i < a.size(); ++i)
		if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i]))
			return (false);
	return (true);
}

/* Plain per-variant model facts, independent of whether they came from
 * a real GGUF read or a scaled estimate (Milestone G2, Part 5 of the
 * task). n_layer_all/arch_name/n_embd/n_head/n_head_kv/model_max_
 * context are a real, physical property of the whole model FAMILY, not
 * of one quant -- quantization changes per-tensor byte width, never
 * tensor shapes/counts or the training context-length ceiling -- so
 * these fields are IDENTICAL across every variant of one real installed
 * sibling's family (see scale_hparams()'s own comment). Only the byte-
 * COUNT fields differ per variant. */
struct s_variant_hparams
{
	int32_t		n_layer_all = 0;
	uint64_t	bytes_per_layer = 0;
	uint64_t	output_role_bytes = 0;
	uint64_t	total_weight_bytes = 0;
	std::string	arch_name;
	int32_t		n_embd = 0;
	int32_t		n_head = 0;
	int32_t		n_head_kv = 0;
	uint64_t	model_max_context = 0;
	int			model_max_context_known = 0;
};

static s_variant_hparams	hparams_from_estimate(
				const membrane_gpu_model_estimate_t &m)
{
	s_variant_hparams	h;

	h.n_layer_all = m.n_layer;
	h.bytes_per_layer = m.bytes_per_layer;
	h.output_role_bytes = m.output_role_bytes;
	h.total_weight_bytes = m.total_bytes;
	h.arch_name = m.hparams_available ? m.arch_name : "";
	h.n_embd = m.hparams_available ? m.n_embd : 0;
	h.n_head = m.hparams_available ? m.n_head : 0;
	h.n_head_kv = m.hparams_available ? m.n_head_kv : 0;
	h.model_max_context = m.model_max_context;
	h.model_max_context_known = m.model_max_context_available;
	return (h);
}

/* Rescales a real installed sibling's byte-count fields to another
 * variant of the SAME family, by the real ratio of two catalog-
 * recorded, verified download sizes -- the only real, evidence-grounded
 * basis available without the target file itself (Part 5 of the G2
 * task: "do not invent tensor-level exactness without the model
 * file"). Every non-byte-count field (n_layer/arch/n_embd/n_head/
 * n_head_kv/model_max_context) is copied verbatim -- see s_variant_
 * hparams's own top comment for why that is real, not estimated.
 * Disclosed as an ESTIMATE by the caller (variant.estimate_only=1),
 * never claimed exact -- different quant methods do not necessarily
 * scale every tensor uniformly (e.g. K-quants may keep some tensors at
 * higher precision than others), so this is a real, useful, but
 * DISCLOSED approximation, not a measurement. */
static s_variant_hparams	scale_hparams(const s_variant_hparams &base,
				uint64_t base_size_bytes, uint64_t target_size_bytes)
{
	s_variant_hparams	h = base;

	if (base_size_bytes == 0)
		return (h);
	double	ratio = (double)target_size_bytes / (double)base_size_bytes;

	h.bytes_per_layer = (uint64_t)((double)base.bytes_per_layer * ratio);
	h.output_role_bytes = (uint64_t)((double)base.output_role_bytes * ratio);
	h.total_weight_bytes = target_size_bytes;
	return (h);
}

/* One candidate variant's full identity + either real/scaled hparams
 * (hparams_known) or a coarse catalog-size-only fit (!hparams_known) --
 * the C++-side staging area plan_v2_resolver.h's own membrane_plan_v2_
 * variant_t is built from. Kept as owned std::strings (not just
 * pointers) so a std::vector<s_variant_spec> can be freely sorted/
 * filtered before its string storage is handed to the pure C resolver
 * as const char* -- see build_v2_request()'s own comment for the
 * lifetime contract this relies on. */
struct s_variant_spec
{
	std::string				model_name;
	std::string				display_name;
	std::string				parameter_count;
	std::string				model_path;
	std::string				arch_name;
	std::string				quant;
	bool					has_variant = false;
	bool					installed = false;
	bool					arch_known = false;
	membrane_plan_source_t	source = MEMBRANE_PLAN_SOURCE_UNKNOWN;
	bool					estimate_only = false;
	bool					hparams_known = false;
	s_variant_hparams		h;				/* iff hparams_known */
	bool					coarse_fits = false;	/* iff !hparams_known */
	std::string				coarse_reason_code;
	std::string				coarse_reason;
	uint64_t				quality_bytes = 0;	/* Part 8: caller-supplied
									 * quality-order key -- see
									 * plan_v2_resolver.h's own top
									 * comment on variants[] order */
};

static void	fill_ctxrec_for_spec(const s_variant_spec &spec,
				const s_plan_opts &opts, bool have_gpu_device,
				membrane_ctxrec_request_t *req,
				membrane_plan_request_meta_t *meta)
{
	memset(req, 0, sizeof(*req));
	req->n_layer_all = spec.h.n_layer_all;
	req->bytes_per_layer = spec.h.bytes_per_layer;
	req->output_role_bytes = spec.h.output_role_bytes;
	req->arch_name = spec.h.arch_name.c_str();
	req->n_embd = spec.h.n_embd;
	req->n_head = spec.h.n_head;
	req->n_head_kv = spec.h.n_head_kv;
	req->total_weight_bytes = spec.h.total_weight_bytes;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;

	memset(meta, 0, sizeof(*meta));
	meta->kv_placement_source = MEMBRANE_PLAN_SOURCE_FALLBACK_DEFAULT;

	if (opts.want_kv)
	{
		req->precision_request = opts.precision_request;
		meta->precision_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else if (!have_gpu_device)
	{
		/* No real GPU device -- matches context_recommender_dryrun.cpp's
		 * own documented default (adaptive has no working CPU-only
		 * fallback in the existing, unchanged joint planner today). */
		req->precision_request = MEMBRANE_JOINT_KV_NATIVE;
		meta->precision_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}
	else
	{
		req->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
		meta->precision_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	if (opts.want_gpu_layers)
	{
		req->gpu_layers_request = opts.gpu_layers_request;
		meta->gpu_layers_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else
	{
		req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
		meta->gpu_layers_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}

	if (opts.want_ctx && !opts.ctx_auto)
	{
		/* Explicit single-candidate path (Part 4/6 of the G2 task):
		 * exactly ONE candidate per variant, still evaluated through
		 * the real, unchanged ctxrec pipeline -- never a second,
		 * looser feasibility check, and never silently reduced. */
		req->model_max_context
			= (spec.h.model_max_context_known
				&& opts.ctx_value <= spec.h.model_max_context)
				? spec.h.model_max_context : opts.ctx_value;
		req->model_max_context_known = 1;
		req->minimum_required_context = opts.ctx_value;
		req->candidates[0].ctx = opts.ctx_value;
		req->candidates[0].kv_bytes_native = real_kv_bytes(spec.h.n_head,
				spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
				opts.ctx_value, GGML_TYPE_F16);
		req->candidates[0].kv_bytes_q8 = real_kv_bytes(spec.h.n_head,
				spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
				opts.ctx_value, GGML_TYPE_Q8_0);
		req->candidates[0].kv_bytes_q5 = real_kv_bytes(spec.h.n_head,
				spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
				opts.ctx_value, GGML_TYPE_Q5_1);
		req->candidate_count = 1;
		meta->requested_context = opts.ctx_value;
		meta->context_source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
	}
	else
	{
		req->model_max_context = spec.h.model_max_context;
		req->model_max_context_known = spec.h.model_max_context_known;
		uint64_t	ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
		size_t		n_ctxs = membrane_ctxrec_generate_candidates(
				req->model_max_context, 0, ctxs,
				MEMBRANE_CTXREC_MAX_CANDIDATES);

		for (size_t i = 0; i < n_ctxs; ++i)
		{
			req->candidates[i].ctx = ctxs[i];
			req->candidates[i].kv_bytes_native = real_kv_bytes(spec.h.n_head,
					spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
					ctxs[i], GGML_TYPE_F16);
			req->candidates[i].kv_bytes_q8 = real_kv_bytes(spec.h.n_head,
					spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
					ctxs[i], GGML_TYPE_Q8_0);
			req->candidates[i].kv_bytes_q5 = real_kv_bytes(spec.h.n_head,
					spec.h.n_embd, spec.h.n_head_kv, spec.h.n_layer_all,
					ctxs[i], GGML_TYPE_Q5_1);
		}
		req->candidate_count = n_ctxs;
		meta->requested_context = 0;
		meta->context_source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
	}
}

/* Builds the pure resolver's request from specs -- specs must not be
 * mutated or resized after this call while out is in use: every
 * membrane_plan_v2_variant_t::identity/variant const char* field points
 * directly at a spec's own std::string storage (never copied), valid
 * for exactly as long as `specs` itself is (membrane_plan_v2_resolve()
 * copies every string out via membrane_plan_assemble()'s own copy_str()
 * before returning -- see membrane_plan.c). Part 13: clamps to
 * MEMBRANE_PLAN_V2_MAX_VARIANTS, same defensive bound the pure resolver
 * itself also enforces. */
static void	build_v2_request(const std::vector<s_variant_spec> &specs,
				const s_plan_opts &opts, bool have_gpu_device,
				const membrane_host_meminfo_t &meminfo,
				uint64_t device_free_bytes, uint64_t device_total_bytes,
				membrane_plan_v2_request_t *out)
{
	memset(out, 0, sizeof(*out));
	out->host_total_bytes = meminfo.total_bytes;
	out->host_available_bytes = meminfo.available_bytes;
	out->host_available_known = meminfo.ok;
	out->device_free_bytes = device_free_bytes;
	out->device_total_bytes = device_total_bytes;

	size_t	n = specs.size();

	if (n > MEMBRANE_PLAN_V2_MAX_VARIANTS)
		n = MEMBRANE_PLAN_V2_MAX_VARIANTS;
	out->variant_count = n;
	for (size_t i = 0; i < n; ++i)
	{
		const s_variant_spec			&spec = specs[i];
		membrane_plan_v2_variant_t		*v = &out->variants[i];

		memset(v, 0, sizeof(*v));
		v->identity.model_name = spec.model_name.c_str();
		v->identity.model_path = spec.installed ? spec.model_path.c_str()
				: "";
		v->identity.installed = spec.installed ? 1 : 0;
		v->identity.arch_known = spec.arch_known ? 1 : 0;
		v->identity.arch_name = spec.arch_name.c_str();
		v->identity.display_name = spec.display_name.c_str();
		v->identity.parameter_count = spec.parameter_count.c_str();
		if (spec.has_variant)
		{
			v->variant.has_variant = 1;
			v->variant.quant = spec.quant.c_str();
			v->variant.estimate_only = spec.estimate_only ? 1 : 0;
			v->variant.source = spec.source;
		}
		v->hparams_known = spec.hparams_known ? 1 : 0;
		if (spec.hparams_known)
			fill_ctxrec_for_spec(spec, opts, have_gpu_device, &v->ctxrec_req,
				&v->ctxrec_meta);
		else
		{
			v->coarse_fits = spec.coarse_fits ? 1 : 0;
			snprintf(v->coarse_reason_code, sizeof(v->coarse_reason_code),
				"%s", spec.coarse_reason_code.c_str());
			snprintf(v->coarse_reason, sizeof(v->coarse_reason), "%s",
				spec.coarse_reason.c_str());
		}
	}
}

static void	sort_by_quality(std::vector<s_variant_spec> *specs)
{
	std::sort(specs->begin(), specs->end(),
		[](const s_variant_spec &a, const s_variant_spec &b)
		{
			return (a.quality_bytes > b.quality_bytes);
		});
}

static bool	filter_explicit_quant(std::vector<s_variant_spec> *specs,
				const std::string &quant)
{
	std::vector<s_variant_spec>	kept;

	for (auto &s : *specs)
		if (s.has_variant && ieq(s.quant, quant))
		{
			s.source = MEMBRANE_PLAN_SOURCE_EXPLICIT_USER;
			kept.push_back(s);
		}
	*specs = kept;
	return (!specs->empty());
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

static void	print_json_reasons(json *j, const membrane_plan_t &plan)
{
	json	arr = json::array();

	for (size_t i = 0; i < plan.reason_count; ++i)
		arr.push_back({{"code", plan.reasons[i].code},
			{"detail", plan.reasons[i].detail}});
	(*j)["reasons"] = arr;
}

static const char	*kv_precision_label(int p)
{
	if (p == MEMBRANE_JOINT_KV_NATIVE)
		return ("native");
	if (p == MEMBRANE_JOINT_KV_Q8)
		return ("q8");
	if (p == MEMBRANE_JOINT_KV_Q5)
		return ("q5");
	return ("unknown");
}

static const char	*kv_placement_label(int m)
{
	return (membrane_kv_placement_mode_name(m));
}

static const char	*precision_request_label(int p)
{
	if (p == MEMBRANE_JOINT_PRECISION_REQUEST_AUTO)
		return ("adaptive");
	return (kv_precision_label(p));
}

static const char	*gpu_layers_request_label(int32_t v)
{
	if (v == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_ALL)
		return ("all");
	if (v == MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO)
		return ("auto");
	static char	buf[32];

	snprintf(buf, sizeof(buf), "%d", v);
	return (buf);
}

static std::string	fmt_mib(uint64_t bytes)
{
	char	buf[64];

	snprintf(buf, sizeof(buf), "%.1f MiB", (double)bytes / (1024.0 * 1024.0));
	return (std::string(buf));
}

static const char	*variant_label(const membrane_plan_t &p)
{
	return (p.identity.variant_known ? p.identity.variant : "unknown");
}

static json	plan_json_object(const membrane_plan_t &plan)
{
	json	j;

	j["identity"] = {
		{"model_name", plan.identity.model_name},
		{"installed", plan.identity.installed != 0},
		{"model_path", plan.identity.installed ? plan.identity.model_path
			: ""},
		{"display_name", plan.identity.display_name},
		{"parameter_count", plan.identity.parameter_count},
		{"arch_known", plan.identity.arch_known != 0},
		{"arch_name", plan.identity.arch_known ? plan.identity.arch_name
			: ""},
		{"variant_known", plan.identity.variant_known != 0},
		{"variant", plan.identity.variant_known ? plan.identity.variant : ""},
		{"variant_source",
			membrane_plan_source_name(plan.identity.variant_source)},
		{"variant_estimate_only", plan.identity.variant_estimate_only != 0},
	};

	j["workload"] = {
		{"requested_context", plan.workload.requested_context},
		{"context_source",
			membrane_plan_source_name(plan.workload.context_source)},
		{"precision_request",
			precision_request_label(plan.workload.precision_request)},
		{"precision_source",
			membrane_plan_source_name(plan.workload.precision_source)},
		{"gpu_layers_request",
			gpu_layers_request_label(plan.workload.gpu_layers_request)},
		{"gpu_layers_source",
			membrane_plan_source_name(plan.workload.gpu_layers_source)},
		{"kv_placement_request",
			kv_placement_label(plan.workload.kv_placement_mode)},
		{"kv_placement_source",
			membrane_plan_source_name(plan.workload.kv_placement_source)},
	};

	j["hardware"] = {
		{"host_total_bytes", plan.hardware.host_total_bytes},
		{"host_available_bytes", plan.hardware.host_available_bytes},
		{"host_available_known", plan.hardware.host_available_known != 0},
		{"host_reserve_bytes", plan.hardware.host_reserve_bytes},
		{"device_known", plan.hardware.device_known != 0},
		{"backend", plan.hardware.device_known ? plan.hardware.backend : ""},
		{"device_name",
			plan.hardware.device_known ? plan.hardware.device_name : ""},
		{"device_total_bytes", plan.hardware.device_total_bytes},
		{"device_free_bytes", plan.hardware.device_free_bytes},
	};

	if (plan.decisions.has_decisions)
		j["decisions"] = {
			{"context", plan.decisions.context},
			{"context_source",
				membrane_plan_source_name(plan.decisions.context_source)},
			{"gpu_layers", plan.decisions.gpu_layers},
			{"gpu_layers_source",
				membrane_plan_source_name(plan.decisions.gpu_layers_source)},
			{"kv_precision", kv_precision_label(plan.decisions.kv_precision)},
			{"kv_precision_source",
				membrane_plan_source_name(plan.decisions.kv_precision_source)},
			{"kv_placement",
				kv_placement_label(plan.decisions.kv_placement)},
			{"kv_placement_source",
				membrane_plan_source_name(plan.decisions.kv_placement_source)},
		};
	else
		j["decisions"] = nullptr;

	j["feasibility"] = {
		{"feasible", plan.feasibility.feasible != 0},
		{"limiting_resource", plan.feasibility.limiting_resource},
		{"max_feasible_context_known",
			plan.feasibility.max_feasible_context_known != 0},
		{"max_feasible_context",
			plan.feasibility.max_feasible_context_known
				? plan.feasibility.max_feasible_context : 0},
		{"host_required_bytes", plan.feasibility.host_required_bytes},
		{"host_headroom_known", plan.feasibility.host_headroom_known != 0},
		{"host_headroom_bytes", plan.feasibility.host_headroom_known
			? plan.feasibility.host_headroom_bytes : 0},
	};

	print_json_reasons(&j, plan);
	j["explanation"] = plan.explanation;
	return (j);
}

static json	plan_v2_json(const membrane_plan_v2_result_t &res)
{
	json	j;

	j["schema_version"] = MEMBRANE_PLAN_V2_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "plan";
	j["ok"] = true;
	j["policy_version"] = res.policy_version;

	if (res.candidate_count > 0)
	{
		const membrane_plan_t	&primary = res.has_selected
				? res.candidates[res.selected_index] : res.candidates[0];
		json	primary_obj = plan_json_object(primary);

		for (auto &el : primary_obj.items())
			j[el.key()] = el.value();
	}

	json	variants = json::array();

	for (size_t i = 0; i < res.candidate_count; ++i)
	{
		const membrane_plan_t	&c = res.candidates[i];
		json	v;

		v["variant"] = variant_label(c);
		v["source"] = membrane_plan_source_name(c.identity.variant_source);
		v["estimate_only"] = c.identity.variant_estimate_only != 0;
		v["evaluated"] = c.decisions.has_decisions != 0;
		v["feasible"] = c.feasibility.feasible != 0;
		v["limiting_resource"] = c.feasibility.limiting_resource;
		v["selected"] = ((int)i == res.selected_index);
		v["plan"] = plan_json_object(c);
		variants.push_back(v);
	}
	j["variants_evaluated"] = variants;

	if (res.has_selected)
		j["selected_variant"]
			= std::string(variant_label(res.candidates[res.selected_index]));
	else
		j["selected_variant"] = nullptr;

	json	alts = json::array();

	for (size_t i = 0; i < res.alternative_count; ++i)
		alts.push_back(variant_label(res.candidates[res.alternative_indices[i]]));
	j["alternatives"] = alts;
	j["feasible"] = res.feasible != 0;
	j["explanation"] = res.explanation;
	return (j);
}

static void	render_json_v2(const membrane_plan_v2_result_t &res)
{
	printf("%s\n", plan_v2_json(res).dump().c_str());
}

static void	render_human_v2(const membrane_plan_v2_result_t &res);

/* Milestone H3: without --runtime, exactly the pre-H3 output. With
 * --runtime membrane-native, the SAME Planner v2 result is classified
 * against the native runtime's capabilities (runtime_plan_cmd.h) and
 * embedded verbatim as "planner_plan" -- nothing is re-planned. */
static int	render_plan_result(const membrane_plan_v2_result_t &res,
				const s_plan_opts &opts, bool want_json)
{
	if (opts.runtime.empty())
	{
		if (want_json)
			render_json_v2(res);
		else
			render_human_v2(res);
		return (MEMBRANE_EXIT_SUCCESS);
	}
	if (res.candidate_count == 0)
	{
		print_err(want_json, "NO_PLAN", "Planner v2 evaluated no candidate "
			"variant, so there is nothing to assess");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	return (membrane_runtime_plan_native(res.has_selected
			? res.candidates[res.selected_index] : res.candidates[0],
		plan_v2_json(res), want_json));
}

static void	render_human_v2(const membrane_plan_v2_result_t &res)
{
	if (res.candidate_count == 0)
	{
		printf("No candidate variants were evaluated.\n");
		return ;
	}

	const membrane_plan_t	&primary = res.has_selected
			? res.candidates[res.selected_index] : res.candidates[0];

	printf("Model\n");
	printf("  %s\n", primary.identity.display_name);
	if (!primary.identity.installed)
		printf("  (not installed -- catalog metadata only)\n");
	if (primary.identity.arch_known)
		printf("  Architecture: %s\n", primary.identity.arch_name);

	printf("\nCandidate variants evaluated\n");
	for (size_t i = 0; i < res.candidate_count; ++i)
	{
		const membrane_plan_t	&c = res.candidates[i];

		printf("  %s%s%s\n", variant_label(c),
			c.identity.variant_estimate_only ? " (estimate only)" : "",
			(int)i == res.selected_index ? " [selected]" : "");
	}

	printf("\nHardware\n");
	if (primary.hardware.device_known)
		printf("  Backend: %s\n  Device: %s\n  VRAM total: %s\n"
			"  VRAM free: %s\n", primary.hardware.backend,
			primary.hardware.device_name,
			fmt_mib(primary.hardware.device_total_bytes).c_str(),
			fmt_mib(primary.hardware.device_free_bytes).c_str());
	else
		printf("  Backend: CPU-only (no GPU device detected)\n");
	if (primary.hardware.host_available_known)
		printf("  Host RAM total: %s\n  Host RAM available: %s\n",
			fmt_mib(primary.hardware.host_total_bytes).c_str(),
			fmt_mib(primary.hardware.host_available_bytes).c_str());
	else
		printf("  Host RAM: unavailable (could not read /proc/meminfo)\n");

	if (res.has_selected)
	{
		const membrane_plan_t	&sel = res.candidates[res.selected_index];

		printf("\nSelected\n");
		printf("  Variant: %s%s\n", variant_label(sel),
			sel.identity.variant_estimate_only ? " (estimate only)" : "");
		if (sel.decisions.has_decisions)
		{
			printf("  Context: %llu (%s)\n",
				(unsigned long long)sel.decisions.context,
				membrane_plan_source_name(sel.decisions.context_source));
			printf("  GPU layers: %d (%s)\n", sel.decisions.gpu_layers,
				membrane_plan_source_name(sel.decisions.gpu_layers_source));
			printf("  KV precision: %s (%s)\n",
				kv_precision_label(sel.decisions.kv_precision),
				membrane_plan_source_name(sel.decisions.kv_precision_source));
			printf("  KV placement: %s (%s)\n",
				kv_placement_label(sel.decisions.kv_placement),
				membrane_plan_source_name(sel.decisions.kv_placement_source));
		}
		else
			printf("  No context/GPU/KV decision -- estimate only (real "
				"GGUF metadata is unavailable for this variant).\n");
	}
	printf("\nFeasible: %s\n", res.feasible ? "yes" : "no");

	if (res.alternative_count > 0)
	{
		printf("\nAlternatives\n");
		for (size_t i = 0; i < res.alternative_count; ++i)
		{
			const membrane_plan_t	&c
					= res.candidates[res.alternative_indices[i]];

			printf("  %s%s\n", variant_label(c),
				c.identity.variant_estimate_only ? " (estimate only)" : "");
		}
	}

	printf("\nWhy\n");
	for (size_t i = 0; i < res.candidate_count; ++i)
	{
		const membrane_plan_t	&c = res.candidates[i];

		printf("  - %s: %s", variant_label(c),
			c.feasibility.feasible ? "feasible" : "infeasible");
		if (!c.feasibility.feasible && c.feasibility.limiting_resource[0]
			!= '\0')
			printf(" (%s)", c.feasibility.limiting_resource);
		else if (c.feasibility.feasible && !c.decisions.has_decisions)
			printf(" (estimate only, no context/GPU/KV decision)");
		printf("\n");
	}
	if (res.has_selected && res.candidates[res.selected_index].reason_count
		> 0)
	{
		const membrane_plan_t	&sel = res.candidates[res.selected_index];

		printf("\nSelected variant's own reasons\n");
		for (size_t i = 0; i < sel.reason_count; ++i)
			printf("  - [%s] %s\n", sel.reasons[i].code,
				sel.reasons[i].detail);
	}
	if (res.explanation[0] != '\0')
		printf("\n%s\n", res.explanation);
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                             */
/* ------------------------------------------------------------------ */

/* Milestone I1: the installed-model Planner v2 resolution, split from its
 * rendering (plan_installed_model_v2() below) so membrane_plan_resolve_
 * installed_v2() can hand the SAME result to `membrane observe` in-
 * process. Pure code motion: identical inputs, identical resolver call;
 * on failure rc, err_code and err_message carry exactly what used to be
 * printed/returned inline. */
static bool	resolve_installed_model_v2(const std::string &name,
				const membrane_registry_entry_t &entry,
				const s_plan_opts &opts, membrane_plan_v2_result_t *res,
				int *rc, std::string *err_code, std::string *err_message)
{
	membrane_gpu_model_estimate_t	m;

	if (!membrane_gpu_estimate_model(entry.path.c_str(), &m))
	{
		*err_code = "MODEL_FILE_UNREADABLE";
		*err_message = "'" + entry.path
			+ "' (registered as '" + name + "') could not be read as a "
			"GGUF file -- it may have moved or been corrupted since it "
			"was registered";
		*rc = MEMBRANE_EXIT_MODEL_ERROR;
		return (false);
	}

	membrane_host_meminfo_t		meminfo;

	membrane_read_host_meminfo(&meminfo);

	membrane_gpu_device_info_t		devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t							n_devices
			= membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	int								gpu_index
			= find_gpu_device_index(devices, n_devices);
	uint64_t						device_free_bytes = 0;
	uint64_t						device_total_bytes = 0;

	if (gpu_index >= 0)
	{
		device_free_bytes = devices[gpu_index].memory_free;
		device_total_bytes = devices[gpu_index].memory_total;
	}

	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*fam
			= membrane_catalog_resolve(cat, name);
	const membrane_catalog_variant_t	*matched
			= find_installed_catalog_variant(fam, entry.basename);

	s_variant_hparams	installed_h = hparams_from_estimate(m);
	std::vector<s_variant_spec>	specs;
	s_variant_spec		installed_spec;

	installed_spec.model_name = name;
	installed_spec.installed = true;
	installed_spec.model_path = entry.path;
	installed_spec.arch_known = m.hparams_available != 0;
	installed_spec.arch_name = installed_h.arch_name;
	installed_spec.display_name = fam != NULL ? fam->display_name : name;
	installed_spec.parameter_count = fam != NULL ? fam->parameter_count : "";
	installed_spec.hparams_known = true;
	installed_spec.h = installed_h;
	installed_spec.quality_bytes = m.total_bytes;
	if (matched != NULL)
	{
		installed_spec.has_variant = true;
		installed_spec.quant = matched->quant;
		installed_spec.source = MEMBRANE_PLAN_SOURCE_MODEL_METADATA;
		installed_spec.estimate_only = false;
	}
	specs.push_back(installed_spec);

	/* Milestone G2's core architectural change (Part 2 of the task):
	 * every OTHER real catalog variant of the SAME family the installed
	 * file anchors to is now evaluated through the exact same joint
	 * pipeline too (via scale_hparams()'s own disclosed estimate),
	 * instead of only ever showing the one installed variant. Only
	 * possible when a real anchor (matched) exists -- otherwise this is
	 * exactly G1's own disclosed limitation (unknown installed variant,
	 * no siblings possible), unchanged. */
	if (fam != NULL && matched != NULL)
		for (const auto &v : fam->variants)
		{
			if (v.filename == matched->filename)
				continue ;
			s_variant_spec	sib;

			sib.model_name = name;
			sib.installed = false;
			sib.arch_known = installed_spec.arch_known;
			sib.arch_name = installed_spec.arch_name;
			sib.display_name = installed_spec.display_name;
			sib.parameter_count = installed_spec.parameter_count;
			sib.has_variant = true;
			sib.quant = v.quant;
			sib.source = MEMBRANE_PLAN_SOURCE_CATALOG_METADATA;
			sib.estimate_only = true;
			sib.hparams_known = true;
			sib.h = scale_hparams(installed_h, matched->size_bytes,
					v.size_bytes);
			sib.quality_bytes = v.size_bytes;
			specs.push_back(sib);
		}

	if (opts.want_quant)
	{
		if (!filter_explicit_quant(&specs, opts.quant))
		{
			*err_code = "NOT_FOUND";
			*err_message = "'" + opts.quant + "' is not "
				"a known variant of the installed model '" + name + "' (or "
				"its catalog family) -- see `membrane model info " + name
				+ "`";
			*rc = MEMBRANE_EXIT_CLI_ERROR;
			return (false);
		}
	}
	sort_by_quality(&specs);

	membrane_plan_v2_request_t	req;

	build_v2_request(specs, opts, gpu_index >= 0, meminfo, device_free_bytes,
		device_total_bytes, &req);
	membrane_plan_v2_resolve(&req, res);
	*rc = MEMBRANE_EXIT_SUCCESS;
	return (true);
}

static int	plan_installed_model_v2(const std::string &name,
				const membrane_registry_entry_t &entry,
				const s_plan_opts &opts, bool want_json)
{
	membrane_plan_v2_result_t	res;
	int							rc;
	std::string					err_code;
	std::string					err_message;

	if (!resolve_installed_model_v2(name, entry, opts, &res, &rc, &err_code,
			&err_message))
	{
		print_err(want_json, err_code, err_message);
		return (rc);
	}
	return (render_plan_result(res, opts, want_json));
}

bool	membrane_plan_resolve_installed_v2(const std::string &name,
			membrane_plan_v2_result_t *out, std::string *err_code,
			std::string *err_message)
{
	std::string					registry_path = membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;
	s_plan_opts					opts;
	int							rc;

	if (registry_path.empty())
	{
		*err_code = "IO_ERROR";
		*err_message = "neither XDG_DATA_HOME nor HOME is set";
		return (false);
	}
	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		*err_code = reg_err.code;
		*err_message = reg_err.message;
		return (false);
	}

	const membrane_registry_entry_t	*entry
			= membrane_registry_find(reg, name);

	if (entry == NULL)
	{
		*err_code = "NOT_FOUND";
		*err_message = "'" + name + "' is not a registered model";
		return (false);
	}
	opts.name = name;
	return (resolve_installed_model_v2(name, *entry, opts, out, &rc, err_code,
			err_message));
}

static int	plan_catalog_only_model_v2(const membrane_catalog_family_t &fam,
				const s_plan_opts &opts, bool want_json)
{
	membrane_host_meminfo_t				meminfo;
	membrane_variant_selector_input_t	hw;

	membrane_read_host_meminfo(&meminfo);
	hw.host_total_bytes = meminfo.total_bytes;
	hw.host_available_bytes = meminfo.available_bytes;
	hw.host_available_known = meminfo.ok;

	std::vector<membrane_variant_fit_t>	considered;

	/* Reuses variant_selector.h's own real host-memory-fit check for
	 * every family variant -- never re-derived here (Part 5/12 of the
	 * G2 task). The RETURNED "best" pointer is deliberately ignored:
	 * Milestone G2's own point is that variant selection no longer
	 * happens independently before the joint reasoning below -- here
	 * there IS no real joint reasoning possible (no installed sibling
	 * anchors any hparams for this family at all), so every variant is
	 * surfaced side by side instead of one being silently pre-picked. */
	membrane_select_variant(fam, hw, &considered);

	std::vector<s_variant_spec>	specs;

	for (size_t i = 0; i < fam.variants.size() && i < considered.size(); ++i)
	{
		const membrane_catalog_variant_t	&v = fam.variants[i];
		s_variant_spec						spec;

		spec.model_name = fam.name;
		spec.installed = false;
		spec.arch_known = !fam.arch.empty();
		spec.arch_name = fam.arch;
		spec.display_name = fam.display_name;
		spec.parameter_count = fam.parameter_count;
		spec.has_variant = true;
		spec.quant = v.quant;
		spec.source = MEMBRANE_PLAN_SOURCE_HARDWARE_AUTO;
		spec.estimate_only = true;
		spec.hparams_known = false;
		spec.coarse_fits = considered[i].fits;
		spec.coarse_reason_code = considered[i].reason_code;
		spec.coarse_reason = considered[i].reason;
		spec.quality_bytes = v.size_bytes;
		specs.push_back(spec);
	}

	if (opts.want_quant)
	{
		if (membrane_catalog_find_variant(fam, opts.quant) == NULL)
		{
			print_err(want_json, "NOT_FOUND", "'" + opts.quant + "' is not "
				"an available variant of '" + fam.name + "' -- see "
				"`membrane model info " + fam.name + "`");
			return (MEMBRANE_EXIT_CLI_ERROR);
		}
		filter_explicit_quant(&specs, opts.quant);
	}
	sort_by_quality(&specs);

	membrane_plan_v2_request_t	req;
	membrane_plan_v2_result_t	res;

	build_v2_request(specs, opts, false, meminfo, 0, 0, &req);
	membrane_plan_v2_resolve(&req, &res);
	return (render_plan_result(res, opts, want_json));
}

/* Milestone H3: an external runtime's model is identified ONLY by the
 * runtime's own id -- no registry/catalog lookup happens (their
 * identities are never merged), and no Planner v2 plan is built (its
 * memory math needs real GGUF hparams the runtime API does not expose),
 * so this is always a capability-only assessment. */
static int	plan_external_runtime(const s_plan_opts &opts, bool want_json)
{
	const membrane_runtime_adapter_t	*a
			= membrane_runtime_registry_find(opts.runtime);
	membrane_runtime_plan_request_t	req;

	if (a == NULL)
	{
		print_err(want_json, "CLI_ERROR", opts.runtime == MEMBRANE_RUNTIME_ID_VLLM
			? "runtime '" + opts.runtime + "' is a reserved identifier for a "
				"future external-runtime adapter; no adapter is implemented "
				"in this build"
			: "unknown runtime '" + opts.runtime + "' -- see `membrane "
				"runtime list`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	req.context_known = opts.want_ctx && !opts.ctx_auto;
	req.context = opts.ctx_value;
	req.gpu_layers_known = opts.want_gpu_layers
		&& opts.gpu_layers_request != MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req.gpu_layers = opts.gpu_layers_request;
	req.kv_precision_known = opts.want_kv
		&& opts.precision_request != MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	req.kv_precision = opts.precision_request;
	req.quant = opts.want_quant ? opts.quant : std::string();
	return (membrane_runtime_plan_external(*a, opts.name, req, want_json));
}

int	membrane_plan_cmd_dispatch(const std::vector<std::string> &args,
				bool want_json)
{
	s_plan_opts	opts;
	std::string	err;

	if (!parse_plan_opts(args, &opts, &err))
	{
		print_err(want_json, "CLI_ERROR", err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!opts.runtime.empty() && opts.runtime != MEMBRANE_RUNTIME_ID_NATIVE)
		return (plan_external_runtime(opts, want_json));

	std::string					registry_path = membrane_registry_resolve_path();
	membrane_registry_t			reg;
	membrane_registry_error_t	reg_err;

	if (registry_path.empty())
	{
		print_err(want_json, "IO_ERROR", "neither XDG_DATA_HOME nor HOME "
			"is set");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}
	if (!membrane_registry_load(registry_path, &reg, &reg_err))
	{
		print_err(want_json, reg_err.code, reg_err.message);
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	/* Same resolution precedence as `membrane use` (Section 23 there):
	 * an already-registered name always wins outright; otherwise an
	 * exact catalog id/alias. Never mutates either (Part 4: `membrane
	 * plan` is READ-ONLY -- see plan_cmd.h's own top comment). */
	const membrane_registry_entry_t	*entry
			= membrane_registry_find(reg, opts.name);

	if (entry != NULL)
		return (plan_installed_model_v2(opts.name, *entry, opts, want_json));

	membrane_catalog_t	cat = membrane_catalog_load();
	const membrane_catalog_family_t	*fam
			= membrane_catalog_resolve(cat, opts.name);

	if (fam == NULL)
	{
		print_err(want_json, "NOT_FOUND", "'" + opts.name + "' is not an "
			"installed model or a known catalog model -- try `membrane "
			"model search " + opts.name + "`");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	return (plan_catalog_only_model_v2(*fam, opts, want_json));
}
