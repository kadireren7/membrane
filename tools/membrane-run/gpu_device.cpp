#include "gpu_device.h"

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

static std::string	ci_lower(const std::string &s)
{
	std::string	out(s);

	for (char &c : out)
		c = (char)std::tolower((unsigned char)c);
	return (out);
}

static int	membrane_type_from_ggml(enum ggml_backend_dev_type t)
{
	if (t == GGML_BACKEND_DEVICE_TYPE_CPU)
		return (MEMBRANE_DEV_TYPE_CPU);
	if (t == GGML_BACKEND_DEVICE_TYPE_GPU)
		return (MEMBRANE_DEV_TYPE_GPU);
	if (t == GGML_BACKEND_DEVICE_TYPE_IGPU)
		return (MEMBRANE_DEV_TYPE_IGPU);
	return (MEMBRANE_DEV_TYPE_UNKNOWN);
}

static void	copy_bounded(char *dst, size_t dst_size, const char *src)
{
	if (src == NULL)
		src = "";
	snprintf(dst, dst_size, "%s", src);
}

/* Strictly parses the "blk.<N>." prefix llama.cpp's own tensor-naming
 * convention uses (LLM_TENSOR_NAMES in llama-arch.cpp), e.g.
 * "blk.0.attn_q.weight". atoi() alone would silently map a malformed
 * name like "blk.foo.bias" to layer 0, corrupting the per-layer byte
 * estimate before any model load -- reject anything that isn't
 * exactly one or more decimal digits followed by '.'. */
static bool	parse_blk_layer_index(const char *name, int *out_idx)
{
	const char	*p = name + 4;
	const char	*digits_start = p;
	long		val;

	while (*p >= '0' && *p <= '9')
		++p;
	if (p == digits_start || *p != '.')
		return (false);
	errno = 0;
	val = strtol(digits_start, NULL, 10);
	if (errno == ERANGE)
		return (false);
	if (val < 0 || val > INT32_MAX)
		return (false);
	*out_idx = (int)val;
	return (true);
}

int	membrane_gpu_backend_available(void)
{
	return (llama_supports_gpu_offload() ? 1 : 0);
}

size_t	membrane_gpu_list_devices(membrane_gpu_device_info_t *out,
			size_t max_out)
{
	size_t	n;
	size_t	i;

	n = ggml_backend_dev_count();
	if (n > max_out)
		n = max_out;
	for (i = 0; i < n; ++i)
	{
		ggml_backend_dev_t			dev = ggml_backend_dev_get(i);
		struct ggml_backend_dev_props	props;

		ggml_backend_dev_get_props(dev, &props);
		copy_bounded(out[i].name, sizeof(out[i].name), props.name);
		copy_bounded(out[i].description, sizeof(out[i].description),
			props.description);
		copy_bounded(out[i].backend, sizeof(out[i].backend),
			ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
		out[i].memory_free = props.memory_free;
		out[i].memory_total = props.memory_total;
		out[i].type = membrane_type_from_ggml(props.type);
		out[i].native_handle = dev;
	}
	return (n);
}

size_t	membrane_gpu_match_device(const membrane_gpu_device_info_t *devices,
			size_t n_devices, const char *query, size_t *out_index)
{
	std::string	needle;
	size_t		matches;
	size_t		i;

	if (query == NULL || devices == NULL)
		return (0);
	needle = ci_lower(query);
	if (needle.find_first_not_of(" \t") == std::string::npos)
		return (0);
	matches = 0;
	for (i = 0; i < n_devices; ++i)
	{
		if (devices[i].type != MEMBRANE_DEV_TYPE_GPU
			&& devices[i].type != MEMBRANE_DEV_TYPE_IGPU)
			continue;
		std::string	hay = ci_lower(devices[i].name) + " "
			+ ci_lower(devices[i].description);
		if (hay.find(needle) != std::string::npos)
		{
			if (matches == 0 && out_index != NULL)
				*out_index = i;
			matches++;
		}
	}
	return (matches);
}

static int	read_u32_key(struct gguf_context *ctx, const std::string &key,
				int32_t *out)
{
	int64_t	id = gguf_find_key(ctx, key.c_str());

	if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_UINT32)
		return (0);
	*out = (int32_t)gguf_get_val_u32(ctx, id);
	return (1);
}

/* Phase 33: LLM_KV_CONTEXT_LENGTH's own real GGUF key shape
 * ("%s.context_length", llama-arch.cpp), read the same UINT32 type
 * llama-model.cpp's own `ml.get_key(LLM_KV_CONTEXT_LENGTH,
 * hparams.n_ctx_train)` reads into its uint32_t n_ctx_train field --
 * source-verified, not guessed. A present-but-zero value is left
 * unavailable (0/0 is never a real model's context ceiling, and
 * context_recommender.h's INVALID_MODEL_MAX_CONTEXT status exists
 * specifically to distinguish "present but nonsensical" from "genuinely
 * missing" one layer up -- this function itself only distinguishes
 * "read a usable positive value" from "did not", exactly like every
 * other key read in this file). */
static void	read_model_max_context(struct gguf_context *ctx,
				const std::string &arch, membrane_gpu_model_estimate_t *out)
{
	int32_t	v = 0;

	out->model_max_context_available = 0;
	out->model_max_context = 0;
	if (!read_u32_key(ctx, arch + ".context_length", &v) || v <= 0)
		return;
	out->model_max_context = (uint64_t)v;
	out->model_max_context_available = 1;
}

static void	read_hparams(struct gguf_context *ctx,
				membrane_gpu_model_estimate_t *out)
{
	int64_t	arch_id = gguf_find_key(ctx, "general.architecture");
	std::string	arch;

	out->hparams_available = 0;
	if (arch_id < 0 || gguf_get_kv_type(ctx, arch_id) != GGUF_TYPE_STRING)
		return;
	arch = gguf_get_val_str(ctx, arch_id);
	if (arch.empty())
		return;
	/* Independent of hparams_available below (Section 26: additive,
	 * never gated behind an unrelated field) -- a model could in
	 * principle expose context_length without embedding_length/
	 * head_count/head_count_kv naming this file recognizes, or vice
	 * versa; each is its own honest "did I actually read this" signal. */
	read_model_max_context(ctx, arch, out);
	/* Same three fields native_kv_bytes()/q8_kv_bytes() need
	 * (llama_model_n_embd/n_head/n_head_kv on an already-loaded
	 * model) -- read here from the arch-prefixed GGUF keys llama.cpp
	 * itself writes, so the KV estimate is available before model
	 * load. Any missing/wrong-typed key leaves hparams_available at 0
	 * -- an unrecognized architecture's differently-named keys are a
	 * real, expected case, not a bug to work around by guessing. */
	if (!read_u32_key(ctx, arch + ".embedding_length", &out->n_embd))
		return;
	if (!read_u32_key(ctx, arch + ".attention.head_count", &out->n_head))
		return;
	if (!read_u32_key(ctx, arch + ".attention.head_count_kv",
			&out->n_head_kv))
		return;
	copy_bounded(out->arch_name, sizeof(out->arch_name), arch.c_str());
	out->hparams_available = 1;
}

int	membrane_gpu_estimate_model(const char *model_path,
			membrane_gpu_model_estimate_t *out)
{
	struct gguf_init_params	params;
	struct gguf_context			*ctx;
	int64_t						n_tensors;
	int64_t						i;
	uint64_t						total_bytes;
	uint64_t						layer_bytes;
	std::map<int, bool>			layer_indices_seen;

	if (out == NULL)
		return (0);
	memset(out, 0, sizeof(*out));
	if (model_path == NULL)
		return (0);
	params.no_alloc = true;
	params.ctx = NULL;
	ctx = gguf_init_from_file(model_path, params);
	if (ctx == NULL)
		return (0);
	n_tensors = gguf_get_n_tensors(ctx);
	total_bytes = 0;
	layer_bytes = 0;

	uint64_t	token_embd_bytes = 0;
	uint64_t	output_bytes = 0;
	uint64_t	output_norm_bytes = 0;
	bool		have_output = false;

	for (i = 0; i < n_tensors; ++i)
	{
		const char	*name = gguf_get_tensor_name(ctx, i);
		size_t		sz = gguf_get_tensor_size(ctx, i);

		total_bytes += sz;
		if (name != NULL && strncmp(name, "blk.", 4) == 0)
		{
			int	layer_idx;

			if (parse_blk_layer_index(name, &layer_idx))
			{
				layer_bytes += sz;
				layer_indices_seen[layer_idx] = true;
			}
		}
		/* Phase 20: exact-name match against llama.cpp's own canonical
		 * tensor names (llama-arch.cpp's TENSOR_NAMES: "token_embd",
		 * "output", "output_norm", each with a ".weight" suffix here --
		 * this project's other tensors, like KV cache tensors, are
		 * never present in a GGUF file at all, only created at context
		 * construction, so there is no ambiguity). See
		 * output_role_bytes' doc comment in gpu_device.h for why only
		 * these three names matter. */
		else if (name != NULL && strcmp(name, "token_embd.weight") == 0)
			token_embd_bytes = sz;
		else if (name != NULL && strcmp(name, "output.weight") == 0)
		{
			output_bytes = sz;
			have_output = true;
		}
		else if (name != NULL && strcmp(name, "output_norm.weight") == 0)
			output_norm_bytes = sz;
	}
	if (layer_indices_seen.empty())
		return (gguf_free(ctx), 0);
	out->total_bytes = total_bytes;
	out->n_layer = (int32_t)layer_indices_seen.size();
	out->bytes_per_layer = layer_bytes / layer_indices_seen.size();
	/* Untied: real "output.weight" is what's GPU-eligible. Tied (no
	 * separate output.weight in this GGUF): llama.cpp duplicates
	 * token_embd.weight's bytes into a separately-allocated,
	 * GPU-eligible "output" tensor instead (see this field's doc
	 * comment in gpu_device.h) -- token_embd_bytes is the right
	 * stand-in for that duplicate's size, NOT a claim that the real
	 * (always-CPU) input tensor itself becomes GPU-resident. */
	out->output_role_bytes = (have_output ? output_bytes : token_embd_bytes)
		+ output_norm_bytes;
	read_hparams(ctx, out);
	gguf_free(ctx);
	return (1);
}
