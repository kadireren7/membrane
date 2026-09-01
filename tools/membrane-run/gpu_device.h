#ifndef MEMBRANE_RUN_GPU_DEVICE_H
# define MEMBRANE_RUN_GPU_DEVICE_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 9B.1: the ONE place membrane-run touches ggml_backend_dev_*()/
 * gguf_*() directly for GPU device enumeration and model-size
 * estimation. Everything else (main.cpp, gpu_policy.c) works with
 * this module's plain-data output, never ggml_backend_dev_t itself --
 * a future CUDA/ROCm backend needs no change here beyond a new CMake
 * build flag, since ggml_backend_dev_*() already abstracts vendor
 * differences uniformly.
 */

enum e_membrane_gpu_device_type
{
	MEMBRANE_DEV_TYPE_CPU = 0,
	MEMBRANE_DEV_TYPE_GPU,
	MEMBRANE_DEV_TYPE_IGPU,
	MEMBRANE_DEV_TYPE_UNKNOWN
};

# define MEMBRANE_GPU_DEVICE_NAME_MAX	64
# define MEMBRANE_GPU_DEVICE_DESC_MAX	256
# define MEMBRANE_GPU_MAX_DEVICES		16

typedef struct s_membrane_gpu_device_info
{
	char		name[MEMBRANE_GPU_DEVICE_NAME_MAX];
	char		description[MEMBRANE_GPU_DEVICE_DESC_MAX];
	char		backend[MEMBRANE_GPU_DEVICE_NAME_MAX];	/* ggml backend
								 * registry name, e.g. "Vulkan" */
	uint64_t	memory_free;
	uint64_t	memory_total;
	int			type;			/* enum e_membrane_gpu_device_type */
	void		*native_handle;	/* opaque ggml_backend_dev_t -- pass
								 * back unmodified into
								 * llama_model_params.devices; never
								 * dereferenced outside this module */
}	membrane_gpu_device_info_t;

/* llama_supports_gpu_offload(): whether ANY GPU backend is compiled
 * into this build, without enumerating devices at all. */
int		membrane_gpu_backend_available(void);

/* Enumerates every registered ggml backend device, CPU included.
 * Returns the number written into out (capped at max_out, which
 * should be >= MEMBRANE_GPU_MAX_DEVICES). */
size_t	membrane_gpu_list_devices(membrane_gpu_device_info_t *out,
			size_t max_out);

/* Case-insensitive substring match of `query` against each GPU/IGPU
 * device's name+description (never matches CPU-type entries).
 * Returns the number of matches; when that is exactly 1, *out_index
 * is set to the matching device's index into `devices`. */
size_t	membrane_gpu_match_device(const membrane_gpu_device_info_t *devices,
			size_t n_devices, const char *query, size_t *out_index);

# define MEMBRANE_GPU_ARCH_NAME_MAX	64

typedef struct s_membrane_gpu_model_estimate
{
	uint64_t	bytes_per_layer;	/* real average, from GGUF tensor
									 * sizes -- not a hardcoded number */
	uint64_t	total_bytes;
	/* Phase 20: real bytes of whichever tensor(s) llama.cpp assigns the
	 * LLM_TENSOR_LAYER_OUTPUT role to (see llama-model-loader.cpp's
	 * create_tensor()) -- source-verified against llama.cpp: the
	 * output-role tensor(s) are GPU-eligible as soon as n_gpu_layers >=
	 * 1, counted as ONE extra "layer" beyond the model's n_layer_all
	 * blk.N. layers (llama_model::n_gpu_layers()'s own comment: "plus 1
	 * for the 'output' layer"), while the input token-embedding tensor
	 * is ALWAYS CPU-resident regardless of n_gpu_layers
	 * (llama-model.cpp's dev_input assignment: "there is very little
	 * benefit to offloading the input layer, so always keep it on the
	 * CPU"). Concretely: real "output.weight" + "output_norm.weight"
	 * bytes if a separate output.weight tensor exists; on a TIED-
	 * embedding model (no separate output.weight in the GGUF -- e.g.
	 * SmolLM2), llama.cpp creates a DUPLICATE, separately-allocated
	 * copy of token_embd.weight specifically for the output role
	 * (TENSOR_DUPLICATED, see llama-model-loader.cpp:1086 routing it to
	 * the output buft_list, not the input one) -- so this field is
	 * token_embd.weight's own bytes + output_norm.weight's bytes in
	 * that case. 0 if neither tensor was found (unrecognized/nonstandard
	 * architecture tensor naming) -- callers must not assume every
	 * architecture's output-role footprint is captured here. */
	uint64_t	output_role_bytes;
	int32_t		n_layer;			/* real count of distinct "blk.N."
									 * groups seen */
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	int			hparams_available;	/* 1 if n_embd/n_head/n_head_kv
									 * were read from arch-prefixed
									 * GGUF keys (e.g. an unrecognized
									 * architecture leaves this 0) --
									 * callers must not estimate a KV
									 * budget when this is 0 */
	/* Phase 20: "general.architecture"'s raw string value, always set
	 * whenever hparams_available is 1 (both are read from the same GGUF
	 * key lookup) -- lets a caller run membrane_check_kv_compat()
	 * PRE-LOAD, before this field existed only main.cpp's post-load
	 * read_model_shape() could supply an architecture string. Empty
	 * string if hparams_available is 0. */
	char		arch_name[MEMBRANE_GPU_ARCH_NAME_MAX];

	/* Phase 33: the model's own training/metadata context-length
	 * ceiling -- source-verified against the pinned llama.cpp
	 * (llama-arch.cpp's LLM_KV_CONTEXT_LENGTH == "%s.context_length",
	 * read the same way llama-model.cpp itself reads hparams.n_ctx_train:
	 * `ml.get_key(LLM_KV_CONTEXT_LENGTH, hparams.n_ctx_train)`), read
	 * from the SAME already-open gguf_context as every other hparam
	 * above -- never a second gguf_init_from_file() open (see
	 * membrane_gpu_estimate_model()'s own doc comment). Additive: every
	 * existing field/caller above is unaffected by this field's
	 * presence. model_max_context_available is 0 whenever the key is
	 * missing, wrong-typed, or reads back as 0 -- callers (e.g.
	 * context_recommender.h) must fail closed (MODEL_MAX_CONTEXT_UNKNOWN
	 * or INVALID_MODEL_MAX_CONTEXT) rather than guess a ceiling, exactly
	 * like hparams_available's own existing contract above. */
	uint64_t	model_max_context;
	int			model_max_context_available;
}	membrane_gpu_model_estimate_t;

/* Reads real per-tensor byte sizes AND hparams from GGUF metadata
 * alone (gguf_init_from_file with no_alloc=true) -- no full model
 * load, no tensor data ever read into memory, so this is cheap and
 * safe to call before deciding how many layers to offload. Weight
 * bytes: sums every tensor whose name starts with the real llama.cpp
 * per-layer prefix "blk.N." into an average bytes-per-layer figure
 * (over however many distinct "blk.N." groups are present), every
 * tensor's bytes (layer and non-layer) into a total, and separately
 * tracks the output-role tensor(s)' real bytes (see output_role_bytes
 * above). Hparams: reads "general.architecture" then the arch-prefixed
 * "{arch}.attention.head_count" / "{arch}.attention.head_count_kv" /
 * "{arch}.embedding_length" keys llama.cpp itself writes -- exactly
 * the fields membrane-run's own native_kv_bytes()/q8_kv_bytes()
 * formula needs, without loading the model to call
 * llama_model_n_embd() etc.
 *
 * Returns 1 if at least the tensor-byte estimate succeeded (out->
 * bytes_per_layer/n_layer are then usable); out->hparams_available
 * must be checked separately before trusting n_embd/n_head/n_head_kv/
 * arch_name -- an unrecognized architecture, or missing keys, leaves
 * it 0 rather than a guessed value. Returns 0 (all outputs left at
 * 0/unset) only if the file can't be read as GGUF or has no
 * recognizable "blk." tensors at all -- never a fabricated fallback. */
int		membrane_gpu_estimate_model(const char *model_path,
			membrane_gpu_model_estimate_t *out);

# ifdef __cplusplus
}
# endif

#endif
