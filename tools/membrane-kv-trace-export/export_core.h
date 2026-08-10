#ifndef MEMBRANE_EXPORT_CORE_H
# define MEMBRANE_EXPORT_CORE_H

# include <stddef.h>
# include <stdint.h>
# include <stdio.h>

# include "membrane/kvdump.h"
# include "membrane/codec.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Shared, testable core for membrane-kv-trace-export: record filtering,
 * deterministic batch filenames, and the batch (.kvdump -> many .memkv)
 * driver. main.c is CLI glue only. Single-record --output semantics
 * (find_record/build_label in main.c) are untouched by this file.
 *
 * ggml's GGML_TYPE_F16 enumerator value, per the pinned llama.cpp
 * commit's ggml/include/ggml.h (GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1).
 * Duplicated from main.c's own comment: membrane_kv_header_t.dtype is
 * documented (kvdump.h) as an opaque "producer's element type id", and
 * membrane-kv-capture's producer is always ggml.
 */
# define MEMBRANE_EXPORT_DTYPE_F16_GGML		1u

/* -1 = no filter (both K and V); else a membrane_kv_tensor_t value. */
# define MEMBRANE_EXPORT_TENSOR_BOTH		(-1)

/* Upper bound on records a single batch export will process -- a
 * defensive cap against a hostile/corrupt .kvdump claiming an
 * implausible number of tiny records, not a claim about what a real
 * capture looks like (this project's real captures have dozens). */
# define MEMBRANE_EXPORT_BATCH_MAX_RECORDS	4096u

typedef struct s_membrane_export_range
{
	long	layer_start;	/* -1 = unbounded (treated as 0) */
	long	layer_end;		/* -1 = unbounded */
	int		tensor_filter;	/* MEMBRANE_EXPORT_TENSOR_BOTH or a
							 * membrane_kv_tensor_t value */
}	membrane_export_range_t;

/* True if `h` is a batch-export candidate: F16 dtype, layer within
 * [range->layer_start, range->layer_end] (either bound may be -1 for
 * unbounded), and tensor_type matching range->tensor_filter. */
int	membrane_export_record_selected(const membrane_kv_header_t *h,
		const membrane_export_range_t *range);

/* Writes the deterministic, portable batch filename for (layer, tensor)
 * into out (out_cap bytes): "layer-NNN-k.memkv" / "layer-NNN-v.memkv",
 * zero-padded to at least 3 digits. Carries no model name, prompt, or
 * any other unsafe/arbitrary string. */
void	membrane_export_batch_filename(char *out, size_t out_cap,
			uint32_t layer, uint32_t tensor_type);

typedef struct s_membrane_export_batch_opts
{
	const char					*input_path;
	const char					*output_dir;	/* must already exist */
	membrane_export_range_t	range;
	uint32_t					elements_per_block;
}	membrane_export_batch_opts_t;

typedef struct s_membrane_export_batch_result
{
	uint32_t	exported_count;
	uint32_t	skipped_too_small_count;	/* fewer than one block */
}	membrane_export_batch_result_t;

/*
 * Reads every record of o->input_path, exports every record selected by
 * membrane_export_record_selected() into o->output_dir using the
 * deterministic filename above, and fills *out.
 *
 * Order is deterministic (the .kvdump's own record order, which
 * membrane-kv-capture itself writes in a fixed layer-ascending,
 * K-then-V-per-layer sequence). A record whose payload has fewer than
 * one o->elements_per_block-sized block is skipped (counted in
 * out->skipped_too_small_count), matching single-record --output
 * behavior, which fails outright for the same condition -- batch mode
 * skips instead so one short record cannot abort exporting the rest.
 *
 * Fails clearly (MEMBRANE_ERR_INVALID_ARG, message in err_buf) rather
 * than silently overwriting a file if two selected records would
 * produce the same output filename (i.e. the .kvdump itself contains a
 * duplicate (layer, tensor) F16 record) -- this is a corrupt/unexpected
 * capture, not a normal condition to paper over.
 *
 * Returns MEMBRANE_ERR_IO if input_path cannot be opened, output_dir
 * does not exist/is not a directory, or a destination file cannot be
 * created; MEMBRANE_ERR_CORRUPT_DATA if the .kvdump is truncated/
 * corrupt; MEMBRANE_ERR_INVALID_ARG for a bad range/elements_per_block,
 * a duplicate destination filename (see above), or more than
 * MEMBRANE_EXPORT_BATCH_MAX_RECORDS candidate records.
 */
membrane_status_t	membrane_export_batch_run(
						const membrane_export_batch_opts_t *o,
						membrane_export_batch_result_t *out,
						char *err_buf, size_t err_cap);

# ifdef __cplusplus
}
# endif

#endif
