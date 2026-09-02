#include <cstdio>
#include <cstring>

#include "runtime_session.h"
#include "compat_check.h"
#include "test_helpers.h"

/*
 * Mega Phase A, PR A1: ctest-registered unit tests for the reusable
 * runtime-session core (runtime_session.h). CI-safe only -- no real GGUF
 * model file is required or assumed present (models/ is gitignored, dev-
 * local only; "CI tests should not download models" per this project's own
 * convention). Every test below either needs no model at all (pure
 * arithmetic, synthetic-input planner branches that never touch a real
 * device/model) or deliberately exercises a REAL, deterministic failure
 * path (a nonexistent model path) that needs no valid fixture to succeed.
 *
 * The REAL small-model smoke ("3 sequential requests from one loaded
 * model", Section 7 of the Mega Phase A task) is a separate, NOT
 * ctest-registered, dev-local harness -- runtime-session-smoke.cpp, same
 * precedent as context_recommender_dryrun.cpp -- since it genuinely needs
 * a real local GGUF fixture CI does not have.
 */

static model_shape_t	fake_shape(int32_t n_layer, int32_t n_embd,
					int32_t n_head, int32_t n_head_kv)
{
	model_shape_t	s;

	s.arch_name = "llama";
	s.n_layer = n_layer;
	s.n_embd = n_embd;
	s.n_head = n_head;
	s.n_head_kv = n_head_kv;
	s.n_embd_gqa = (n_head > 0) ? (int64_t)(n_embd / n_head) * n_head_kv : 0;
	return (s);
}

/* 1. Runtime init/shutdown -- trivial, idempotent-safe, no leaked state. */
static void	test_runtime_init_shutdown(void)
{
	membrane_runtime_t	rt = {};

	TEST_ASSERT(!rt.initialized, "runtime starts uninitialized");
	membrane_runtime_init(&rt);
	TEST_ASSERT(rt.initialized, "runtime_init sets initialized");
	membrane_runtime_shutdown(&rt);
	TEST_ASSERT(!rt.initialized, "runtime_shutdown clears initialized");
	/* Shutdown on an already-shutdown runtime must not double-free the
	 * llama backend (real crash risk if it did). */
	membrane_runtime_shutdown(&rt);
	TEST_ASSERT(!rt.initialized, "double shutdown is safe, stays cleared");
}

/* 2. Deterministic pure KV-byte arithmetic -- no model, no device. */
static void	test_kv_bytes_deterministic(void)
{
	model_shape_t	s = fake_shape(4, 256, 8, 8);
	uint64_t		native1 = membrane_native_kv_bytes(s, 512);
	uint64_t		native2 = membrane_native_kv_bytes(s, 512);
	uint64_t		q8 = membrane_q8_kv_bytes(s, 512);
	uint64_t		q5 = membrane_q5_kv_bytes(s, 512);

	TEST_ASSERT(native1 == native2, "native KV bytes deterministic");
	TEST_ASSERT(native1 > 0, "native KV bytes nonzero for a real shape");
	TEST_ASSERT(q8 < native1, "q8 KV bytes smaller than native (compressed)");
	TEST_ASSERT(q5 < q8, "q5 KV bytes smaller than q8 (more compressed)");
	TEST_ASSERT(membrane_kv_bytes_for_mode(s, 512, MEMBRANE_KV_STORE_NATIVE)
		== native1, "kv_bytes_for_mode dispatches NATIVE correctly");
	TEST_ASSERT(membrane_kv_bytes_for_mode(s, 512, MEMBRANE_KV_STORE_Q8)
		== q8, "kv_bytes_for_mode dispatches Q8 correctly");
	TEST_ASSERT(membrane_kv_bytes_for_mode(s, 512, MEMBRANE_KV_STORE_Q5)
		== q5, "kv_bytes_for_mode dispatches Q5 correctly");
}

static void	test_kv_mode_name_and_label(void)
{
	TEST_ASSERT(strcmp(membrane_kv_mode_name(MEMBRANE_KV_STORE_NATIVE),
		"native") == 0, "native mode name");
	TEST_ASSERT(strcmp(membrane_kv_mode_name(MEMBRANE_KV_STORE_Q8), "q8")
		== 0, "q8 mode name");
	TEST_ASSERT(strcmp(membrane_kv_mode_name(MEMBRANE_KV_STORE_Q5), "q5")
		== 0, "q5 mode name");
	TEST_ASSERT(strcmp(membrane_kv_type_label(MEMBRANE_KV_STORE_NATIVE),
		"F16") == 0, "native type label");
	TEST_ASSERT(strcmp(membrane_kv_type_label(MEMBRANE_KV_STORE_Q8), "Q8_0")
		== 0, "q8 type label");
	TEST_ASSERT(strcmp(membrane_kv_type_label(MEMBRANE_KV_STORE_Q5), "Q5_1")
		== 0, "q5 type label");
}

static void	test_gpu_layers_label(void)
{
	TEST_ASSERT(membrane_gpu_layers_label(MEMBRANE_GPU_LAYERS_ALL) == "all",
		"ALL sentinel labeled \"all\"");
	TEST_ASSERT(membrane_gpu_layers_label(MEMBRANE_GPU_LAYERS_AUTO) == "auto",
		"AUTO sentinel labeled \"auto\"");
	TEST_ASSERT(membrane_gpu_layers_label(16) == "16",
		"a concrete count is its own decimal string");
	TEST_ASSERT(membrane_gpu_layers_label(0) == "0",
		"zero is its own decimal string, not a sentinel");
}

/* 3. resolve_gpu_config's CPU-only-requested branch: deterministic in
 * every build configuration (never touches a real device query). */
static void	test_resolve_gpu_config_cpu_only(void)
{
	membrane_run_opts_t				o = {};
	llama_model_params					mp;
	std::vector<ggml_backend_dev_t>	device_storage;
	membrane_gpu_state_t				gs;
	membrane_run_error_t				err;

	o.gpu_layers = 0;
	mp = llama_model_default_params();
	TEST_ASSERT(membrane_resolve_gpu_config(o, 512, &device_storage, &mp,
		&gs, &err) == true, "explicit CPU-only (gpu_layers=0) always ok");
	TEST_ASSERT(!err.set, "no error populated on the CPU-only success path");
	TEST_ASSERT(!gs.requested, "gs.requested is false for CPU-only");
	TEST_ASSERT(gs.gpu_layers_selected == 0,
		"gpu_layers_selected is 0 for CPU-only");
	TEST_ASSERT(mp.n_gpu_layers == 0, "llama_model_params.n_gpu_layers is 0");
	TEST_ASSERT(mp.main_gpu == -1,
		"llama_model_params.main_gpu is -1 (no implicit GPU pick)");
}

/* 4. resolve_kv_placement's DEFAULT no-op branch: deterministic, no
 * device/model needed. */
static void	test_resolve_kv_placement_default_noop(void)
{
	membrane_run_opts_t	o = {};
	model_shape_t			s = fake_shape(4, 256, 8, 8);
	membrane_gpu_state_t	gs = {};
	membrane_run_error_t	err;

	o.kv_placement = MEMBRANE_KV_PLACEMENT_DEFAULT;
	TEST_ASSERT(membrane_resolve_kv_placement(o, s, 512,
		MEMBRANE_KV_STORE_NATIVE, &gs, &err) == true,
		"DEFAULT placement is always a no-op success");
	TEST_ASSERT(!err.set, "no error populated for the DEFAULT no-op");
	TEST_ASSERT(!gs.kv_placement_resolved,
		"kv_placement_resolved stays false for DEFAULT (Section 4/19)");
}

/* GPU/AUTO placement without a verified GPU weight estimate must fail
 * closed with a structured error, not print anything itself. */
static void	test_resolve_kv_placement_gpu_without_estimate_fails_closed(void)
{
	membrane_run_opts_t	o = {};
	model_shape_t			s = fake_shape(4, 256, 8, 8);
	membrane_gpu_state_t	gs = {};
	membrane_run_error_t	err;

	o.kv_placement = MEMBRANE_KV_PLACEMENT_GPU;
	gs.policy_used = false;
	gs.auto_cpu_fallback = false;
	TEST_ASSERT(membrane_resolve_kv_placement(o, s, 512,
		MEMBRANE_KV_STORE_NATIVE, &gs, &err) == false,
		"--kv-placement gpu without a verified estimate fails closed");
	TEST_ASSERT(err.set, "error is populated, not silently discarded");
	TEST_ASSERT(err.exit_code == MEMBRANE_EXIT_UNSUPPORTED_KV,
		"exit code is UNSUPPORTED_KV");
	TEST_ASSERT(!err.human.empty(), "human message is non-empty");
	TEST_ASSERT(!err.message.empty(), "JSON message is non-empty");
	TEST_ASSERT(strlen(err.reason_code) > 0, "reason_code is non-empty");
}

/* 5. resolve_cpu_adaptive_kv: deterministic, synthetic budget inputs. */
static void	test_resolve_cpu_adaptive_kv_picks_within_budget(void)
{
	membrane_run_opts_t	o = {};
	model_shape_t			s = fake_shape(4, 256, 8, 8);
	membrane_gpu_state_t	gs = {};
	membrane_run_error_t	err;
	uint64_t				q8_bytes = membrane_q8_kv_bytes(s, 512);

	o.want_kv_budget = true;
	o.kv_budget_bytes = q8_bytes;	/* exactly fits q8 */
	TEST_ASSERT(membrane_resolve_cpu_adaptive_kv(o, s, 512, &gs, &err)
		== true, "a budget that fits q8 resolves ok");
	TEST_ASSERT(gs.adaptive_used, "adaptive_used set true");
	TEST_ASSERT(gs.adaptive_selected_mode == MEMBRANE_KV_STORE_Q8
		|| gs.adaptive_selected_mode == MEMBRANE_KV_STORE_Q5,
		"a concrete precision was selected, never left as ADAPTIVE");
}

static void	test_resolve_cpu_adaptive_kv_impossible_budget_fails_closed(void)
{
	membrane_run_opts_t	o = {};
	model_shape_t			s = fake_shape(4, 256, 8, 8);
	membrane_gpu_state_t	gs = {};
	membrane_run_error_t	err;

	o.want_kv_budget = true;
	o.kv_budget_bytes = 1;	/* impossibly small -- neither q8 nor q5 fit */
	TEST_ASSERT(membrane_resolve_cpu_adaptive_kv(o, s, 512, &gs, &err)
		== false, "an impossible budget fails closed, never picks anyway");
	TEST_ASSERT(err.set, "error is populated on failure");
	TEST_ASSERT(err.exit_code == MEMBRANE_EXIT_UNSUPPORTED_KV,
		"exit code is UNSUPPORTED_KV");
}

/* Compatibility failure (Section 7's own required category): an
 * unrecognized architecture must reject q8/q5/adaptive KV cleanly, no
 * model or device needed -- membrane_check_kv_compat() is the same pure
 * gate membrane_session_generate() itself calls before every generate. */
static void	test_compat_check_unsupported_architecture_fails_closed(void)
{
	membrane_compat_result_t	compat;

	TEST_ASSERT(membrane_check_kv_compat("totally-unrecognized-arch", 256,
		8, 8, 512, MEMBRANE_KV_STORE_Q8, &compat) == false,
		"an unrecognized architecture rejects q8 KV compression");
	TEST_ASSERT(strlen(compat.reason) > 0,
		"a specific reason is given, not a bare failure");
	TEST_ASSERT(membrane_check_kv_compat("totally-unrecognized-arch", 256,
		8, 8, 512, MEMBRANE_KV_STORE_NATIVE, &compat) == true,
		"native KV has no architecture restriction (Section 27 of the "
		"Mega Phase A task's own error-mapping convention)");
}

/* 6. membrane_model_open() against a nonexistent model path -- a REAL,
 * deterministic failure needing no valid fixture. Confirms: no exit()/
 * abort(), no crash, a clean structured error, no leaked llama_model. */
static void	test_model_open_nonexistent_path_fails_cleanly(void)
{
	membrane_runtime_t			rt = {};
	membrane_run_opts_t			o = {};
	membrane_model_session_t	session;
	membrane_run_error_t		err;

	membrane_runtime_init(&rt);
	o.gpu_layers = 0;
	TEST_ASSERT(membrane_model_open(&rt, "/nonexistent/definitely-not-a-"
			"real-model.gguf", o, 512, &session, &err) == false,
		"opening a nonexistent model path fails");
	TEST_ASSERT(session.model == NULL,
		"session.model stays NULL on a failed open -- nothing to free");
	TEST_ASSERT(err.set, "error is populated");
	TEST_ASSERT(err.exit_code == MEMBRANE_EXIT_MODEL_ERROR,
		"exit code is MODEL_ERROR");
	TEST_ASSERT(strcmp(err.reason_code, MEMBRANE_REASON_MODEL_LOAD_FAILED)
		== 0, "reason_code is MODEL_LOAD_FAILED");
	TEST_ASSERT(!err.human.empty(), "human message is non-empty");
	/* membrane_model_close() on a never-opened session must be a safe
	 * no-op, never a double-free/crash. */
	membrane_model_close(&session);
	TEST_ASSERT(session.model == NULL, "close on a NULL model is a no-op");
	membrane_runtime_shutdown(&rt);
}

/* Two independent failed opens in the SAME process must not leak or
 * corrupt state between them (Section 7: "no leaked global state"). */
static void	test_model_open_repeated_failure_no_leaked_state(void)
{
	membrane_runtime_t	rt = {};
	membrane_run_opts_t	o = {};
	int					i;

	membrane_runtime_init(&rt);
	o.gpu_layers = 0;
	i = 0;
	while (i < 3)
	{
		membrane_model_session_t	session;
		membrane_run_error_t		err;

		TEST_ASSERT(membrane_model_open(&rt, "/nonexistent/again.gguf", o,
				512, &session, &err) == false,
			"each repeated failed open fails independently");
		TEST_ASSERT(session.model == NULL,
			"each failed open leaves session.model NULL");
		membrane_model_close(&session);
		i++;
	}
	membrane_runtime_shutdown(&rt);
}

int	main(void)
{
	test_runtime_init_shutdown();
	test_kv_bytes_deterministic();
	test_kv_mode_name_and_label();
	test_gpu_layers_label();
	test_resolve_gpu_config_cpu_only();
	test_resolve_kv_placement_default_noop();
	test_resolve_kv_placement_gpu_without_estimate_fails_closed();
	test_resolve_cpu_adaptive_kv_picks_within_budget();
	test_resolve_cpu_adaptive_kv_impossible_budget_fails_closed();
	test_compat_check_unsupported_architecture_fails_closed();
	test_model_open_nonexistent_path_fails_cleanly();
	test_model_open_repeated_failure_no_leaked_state();
	printf("test_runtime_session: all tests passed\n");
	return (0);
}
