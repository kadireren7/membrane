#include "gpu_device.h"

#include <cctype>
#include <cstdio>
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
	for (i = 0; i < n_tensors; ++i)
	{
		const char	*name = gguf_get_tensor_name(ctx, i);
		size_t		sz = gguf_get_tensor_size(ctx, i);

		total_bytes += sz;
		if (name != NULL && strncmp(name, "blk.", 4) == 0)
		{
			int	layer_idx = atoi(name + 4);

			layer_bytes += sz;
			layer_indices_seen[layer_idx] = true;
		}
	}
	if (layer_indices_seen.empty())
		return (gguf_free(ctx), 0);
	out->total_bytes = total_bytes;
	out->n_layer = (int32_t)layer_indices_seen.size();
	out->bytes_per_layer = layer_bytes / layer_indices_seen.size();
	read_hparams(ctx, out);
	gguf_free(ctx);
	return (1);
}
