#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "membrane/block.h"
#include "membrane/policy.h"
#include "test_helpers.h"

# define N_LAYERS	4

static char	g_path[] = "/tmp/membrane-policy-XXXXXX";

static void	fill_build(membrane_policy_build_t *b,
				const membrane_precision_t *k, const membrane_precision_t *v)
{
	static uint8_t	hash[MEMBRANE_SHA256_DIGEST_BYTES];
	static int		hash_init = 0;

	if (!hash_init)
	{
		membrane_sha256((const uint8_t *)"smollm2-135m-f16", 16, hash);
		hash_init = 1;
	}
	memset(b, 0, sizeof(*b));
	memcpy(b->model_sha256, hash, MEMBRANE_SHA256_DIGEST_BYTES);
	b->llama_cpp_commit = "c0bc8591e8815c63cb01dd3f051a8b0df02501c9";
	b->layer_count = N_LAYERS;
	b->k_prec = k;
	b->v_prec = v;
	b->model_name = "SmolLM2-135M";
	b->tier_name = "balanced";
	b->cosine_min = 0.995;
	b->top1_min = 98.0;
	b->top5_min = 99.0;
	b->cosine_margin = 0.001;
	b->top1_margin = 0.5;
	b->top5_margin = 0.1;
	b->search_budget = 60;
	b->evals_used = 64;
	b->created_unix_time = 1784900000ULL;
}

static void	corrupt_byte(const char *path, long offset, uint8_t xor_with)
{
	FILE	*f;
	int		c;

	f = fopen(path, "r+b");
	TEST_ASSERT(f != NULL, "reopen policy file for corruption");
	TEST_ASSERT(fseek(f, offset, SEEK_SET) == 0, "seek to corrupt offset");
	c = fgetc(f);
	TEST_ASSERT(c != EOF, "byte exists at corrupt offset");
	TEST_ASSERT(fseek(f, offset, SEEK_SET) == 0, "seek back");
	TEST_ASSERT(fputc(c ^ xor_with, f) != EOF, "write corrupted byte");
	fclose(f);
}

static long	file_len(const char *path)
{
	FILE	*f;
	long	n;

	f = fopen(path, "rb");
	TEST_ASSERT(f != NULL, "open for length check");
	fseek(f, 0, SEEK_END);
	n = ftell(f);
	fclose(f);
	return (n);
}

static void	test_round_trip(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_FP16, MEMBRANE_PRECISION_Q8
	};
	membrane_precision_t	v[N_LAYERS] = {
		MEMBRANE_PRECISION_Q4, MEMBRANE_PRECISION_Q4,
		MEMBRANE_PRECISION_Q4, MEMBRANE_PRECISION_FP16
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;
	membrane_precision_t	got;
	uint32_t				i;

	fill_build(&b, k, v);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK,
		"save succeeds");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_OK,
		"load succeeds");
	TEST_ASSERT(membrane_policy_layer_count(p) == N_LAYERS,
		"layer count round-trips");
	i = 0;
	while (i < N_LAYERS)
	{
		TEST_ASSERT(membrane_policy_query(p, i, 0, &got) == MEMBRANE_OK
			&& got == k[i], "K precision round-trips per layer");
		TEST_ASSERT(membrane_policy_query(p, i, 1, &got) == MEMBRANE_OK
			&& got == v[i], "V precision round-trips per layer");
		i++;
	}
	membrane_policy_destroy(p);
}

static void	test_deterministic_loading(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q4,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q4
	};
	membrane_precision_t	v[N_LAYERS] = {
		MEMBRANE_PRECISION_Q4, MEMBRANE_PRECISION_Q4,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p1;
	membrane_policy_t		*p2;
	membrane_precision_t	g1;
	membrane_precision_t	g2;
	uint32_t				i;

	fill_build(&b, k, v);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK,
		"save for determinism check");
	TEST_ASSERT(membrane_policy_load(g_path, &p1) == MEMBRANE_OK,
		"first load");
	TEST_ASSERT(membrane_policy_load(g_path, &p2) == MEMBRANE_OK,
		"second load of the same file");
	i = 0;
	while (i < N_LAYERS)
	{
		membrane_policy_query(p1, i, 0, &g1);
		membrane_policy_query(p2, i, 0, &g2);
		TEST_ASSERT(g1 == g2, "two independent loads agree on K precision");
		membrane_policy_query(p1, i, 1, &g1);
		membrane_policy_query(p2, i, 1, &g2);
		TEST_ASSERT(g1 == g2, "two independent loads agree on V precision");
		i++;
	}
	membrane_policy_destroy(p1);
	membrane_policy_destroy(p2);
}

static void	test_precision_query_out_of_range(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;
	membrane_precision_t	out;

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK,
		"save for range check");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_OK, "load");
	TEST_ASSERT(membrane_policy_query(p, N_LAYERS, 0, &out)
		== MEMBRANE_ERR_INVALID_ARG, "query at layer_count is out of range");
	TEST_ASSERT(membrane_policy_query(p, 999, 1, &out)
		== MEMBRANE_ERR_INVALID_ARG, "query far out of range");
	TEST_ASSERT(membrane_policy_query(NULL, 0, 0, &out)
		== MEMBRANE_ERR_INVALID_ARG, "query on a NULL policy");
	TEST_ASSERT(membrane_policy_query(p, 0, 0, NULL)
		== MEMBRANE_ERR_INVALID_ARG, "query with a NULL out-param");
	membrane_policy_destroy(p);
}

static void	test_unsupported_precision_rejected(void)
{
	membrane_precision_t	bad[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, (membrane_precision_t)32,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_precision_t	ok[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;

	fill_build(&b, bad, ok);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_ERR_INVALID_ARG,
		"save rejects an out-of-range K precision value");
	fill_build(&b, ok, bad);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_ERR_INVALID_ARG,
		"save rejects an out-of-range V precision value");
}

/*
 * Corrupts one precision byte AND recomputes the checksum over the
 * modified body, so the checksum still passes and the failure can only
 * come from the precision-range check itself -- otherwise this would
 * just be re-testing checksum corruption (a flipped precision byte also
 * breaks the checksum, so without repairing it the two failure modes
 * would be indistinguishable).
 */
static void	test_unsupported_precision_on_disk_rejected(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;
	FILE					*f;
	uint8_t					body[244 + N_LAYERS * 2];
	uint32_t				checksum;
	uint8_t					checksum_le[4];

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK,
		"save a valid policy to hand-corrupt afterward");
	f = fopen(g_path, "r+b");
	TEST_ASSERT(f != NULL, "reopen to inject a bad precision byte");
	TEST_ASSERT(fread(body, 1, sizeof(body), f) == sizeof(body),
		"read the body back for hand-editing");
	body[244] = 200;	/* layer 0's K precision: out of {16, 8, 4} */
	checksum = membrane_block_checksum(body, sizeof(body));
	checksum_le[0] = (uint8_t)checksum;
	checksum_le[1] = (uint8_t)(checksum >> 8);
	checksum_le[2] = (uint8_t)(checksum >> 16);
	checksum_le[3] = (uint8_t)(checksum >> 24);
	TEST_ASSERT(fseek(f, 0, SEEK_SET) == 0, "seek to start");
	TEST_ASSERT(fwrite(body, 1, sizeof(body), f) == sizeof(body),
		"write the edited body back");
	TEST_ASSERT(fwrite(checksum_le, 1, 4, f) == 4,
		"write the recomputed checksum so only the precision check can fail");
	fclose(f);
	TEST_ASSERT(membrane_policy_load(g_path, &p)
		== MEMBRANE_ERR_CORRUPT_DATA,
		"a policy with an invalid on-disk precision byte is rejected "
		"even when its checksum is internally consistent");
	TEST_ASSERT(p == NULL, "rejected load leaves *out NULL");
}

static void	test_model_hash_mismatch(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t		b;
	membrane_policy_t			*p;
	membrane_policy_context_t	ctx;
	char						reason[256];

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_OK, "load");
	memset(&ctx, 0, sizeof(ctx));
	membrane_sha256((const uint8_t *)"a-different-model", 18,
		ctx.model_sha256);
	ctx.llama_cpp_commit = b.llama_cpp_commit;
	ctx.layer_count = N_LAYERS;
	TEST_ASSERT(membrane_policy_validate(p, &ctx, reason, sizeof(reason))
		== MEMBRANE_ERR_MISMATCH, "wrong model hash is rejected");
	TEST_ASSERT(strstr(reason, "model hash") != NULL,
		"mismatch reason names the model hash");
	memcpy(ctx.model_sha256, b.model_sha256, MEMBRANE_SHA256_DIGEST_BYTES);
	TEST_ASSERT(membrane_policy_validate(p, &ctx, reason, sizeof(reason))
		== MEMBRANE_OK, "matching context validates cleanly");
	membrane_policy_destroy(p);
}

static void	test_llama_commit_mismatch(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t		b;
	membrane_policy_t			*p;
	membrane_policy_context_t	ctx;
	char						reason[256];

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_OK, "load");
	memset(&ctx, 0, sizeof(ctx));
	memcpy(ctx.model_sha256, b.model_sha256, MEMBRANE_SHA256_DIGEST_BYTES);
	ctx.llama_cpp_commit = "0000000000000000000000000000000000dead";
	ctx.layer_count = N_LAYERS;
	TEST_ASSERT(membrane_policy_validate(p, &ctx, reason, sizeof(reason))
		== MEMBRANE_ERR_MISMATCH, "wrong llama.cpp commit is rejected");
	TEST_ASSERT(strstr(reason, "commit") != NULL,
		"mismatch reason names the commit");
	membrane_policy_destroy(p);
}

static void	test_layer_count_mismatch(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t		b;
	membrane_policy_t			*p;
	membrane_policy_context_t	ctx;
	char						reason[256];

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_OK, "load");
	memset(&ctx, 0, sizeof(ctx));
	memcpy(ctx.model_sha256, b.model_sha256, MEMBRANE_SHA256_DIGEST_BYTES);
	ctx.llama_cpp_commit = b.llama_cpp_commit;
	ctx.layer_count = N_LAYERS + 1;
	TEST_ASSERT(membrane_policy_validate(p, &ctx, reason, sizeof(reason))
		== MEMBRANE_ERR_MISMATCH, "wrong layer count is rejected");
	TEST_ASSERT(strstr(reason, "layer count") != NULL,
		"mismatch reason names the layer count");
	membrane_policy_destroy(p);
}

static void	test_checksum_corruption(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	/* flip a byte inside model_name (offset 90), far from any field
	 * whose own range check would catch it -- only the checksum can. */
	corrupt_byte(g_path, 90, 0xFF);
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"a single flipped byte is caught by the checksum");
}

static void	test_truncated_policy(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;
	long					full_len;

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	full_len = file_len(g_path);
	TEST_ASSERT(truncate(g_path, full_len - 1) == 0,
		"truncate the checksum itself");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"truncated checksum is rejected");
	TEST_ASSERT(truncate(g_path, 100) == 0, "truncate mid-header");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"truncated header is rejected");
	TEST_ASSERT(truncate(g_path, 0) == 0, "truncate to empty");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"empty file is rejected");
}

static void	test_missing_layer_records(void)
{
	membrane_precision_t	k[N_LAYERS] = {
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8,
		MEMBRANE_PRECISION_Q8, MEMBRANE_PRECISION_Q8
	};
	membrane_policy_build_t	b;
	membrane_policy_t		*p;

	fill_build(&b, k, k);
	TEST_ASSERT(membrane_policy_save(g_path, &b) == MEMBRANE_OK, "save");
	/* full file is header (244) + layer_count*2 (8) + checksum (4) = 256;
	 * keep only 2 of the 4 layers' records (244 + 2*2 = 248) with no
	 * checksum at all -- the header still claims layer_count=4, so the
	 * exact-size check must reject this as too short, even though the
	 * missing bytes are mid-array, not at EOF. */
	TEST_ASSERT(truncate(g_path, 244 + 2 * 2) == 0,
		"truncate to 2 of 4 layers' records, dropping the rest and the "
		"checksum");
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"missing layer records (short file for the claimed layer_count) "
		"are rejected");
}

static void	test_wrong_format(void)
{
	FILE				*f;
	membrane_policy_t	*p;

	f = fopen(g_path, "wb");
	TEST_ASSERT(f != NULL, "open for garbage write");
	TEST_ASSERT(fwrite("not a membrane policy file at all, just text\n",
			1, 46, f) == 46, "write non-policy garbage");
	fclose(f);
	TEST_ASSERT(membrane_policy_load(g_path, &p) == MEMBRANE_ERR_CORRUPT_DATA,
		"a file that isn't the policy format at all is rejected");
}

static void	test_null_and_missing_path_safety(void)
{
	membrane_policy_t	*p;

	TEST_ASSERT(membrane_policy_load(NULL, &p) == MEMBRANE_ERR_INVALID_ARG,
		"NULL path is rejected, not dereferenced");
	TEST_ASSERT(membrane_policy_load("/nonexistent/path/policy.mpol", &p)
		== MEMBRANE_ERR_IO, "a missing file is reported as MEMBRANE_ERR_IO");
	TEST_ASSERT(p == NULL, "failed load leaves *out NULL");
	TEST_ASSERT(membrane_policy_save(NULL, NULL) == MEMBRANE_ERR_INVALID_ARG,
		"NULL build args are rejected, not dereferenced");
	membrane_policy_destroy(NULL);
}

int	main(void)
{
	int	fd;

	fd = mkstemp(g_path);
	TEST_ASSERT(fd >= 0, "temp file created");
	close(fd);
	test_round_trip();
	test_deterministic_loading();
	test_precision_query_out_of_range();
	test_unsupported_precision_rejected();
	test_unsupported_precision_on_disk_rejected();
	test_model_hash_mismatch();
	test_llama_commit_mismatch();
	test_layer_count_mismatch();
	test_checksum_corruption();
	test_truncated_policy();
	test_missing_layer_records();
	test_wrong_format();
	test_null_and_missing_path_safety();
	unlink(g_path);
	return (0);
}
