#ifndef MEMBRANE_ADAPTIVE_CONTROLLER_H
# define MEMBRANE_ADAPTIVE_CONTROLLER_H

# include <atomic>
# include <condition_variable>
# include <cstdint>
# include <functional>
# include <memory>
# include <mutex>
# include <string>
# include <thread>

# include <nlohmann/json.hpp>

# include "adaptive.h"
# include "apply_cmd.h"
# include "server_config.h"

/*
 * Milestone K: the bounded adaptive controller.
 *
 * ARCHITECTURE DECISION (docs/adaptive-runtime.md section 8): the
 * controller is ONE thread inside the existing `membrane serve` process,
 * not a second daemon. That process already owns the runtime state
 * (resident models, pins, in-flight generations) and the lifecycle
 * (signals, graceful shutdown), so there is exactly one owner, no
 * cross-process race on server.json or on a model slot, no extra unit to
 * package and nothing that can outlive the service. The thread sleeps on a
 * condition variable for the configured interval (never a busy loop),
 * does nothing at all while the policy is off, and joins on shutdown.
 *
 * Every tick is: observe (membrane_advise_collect -- the I1/I3 providers)
 * -> sample -> pure decision (adaptive.h) -> and only in apply mode, the J
 * pipeline (membrane_apply_execute: live re-read, Planner v2 validation,
 * stale-plan guards, rollback, audit). It has no mutation path of its own.
 *
 * Everything time- or machine-dependent is injected so the tick is tested
 * with a fake clock, fake observations and a fake executor, no sleeps.
 */

typedef struct s_membrane_adaptive_deps
{
	std::function<int64_t(void)>	now_ms;
	/* observation + I3 intelligence for a runtime id */
	std::function<bool(const std::string &runtime,
		membrane_observation_snapshot_t *snap,
		membrane_memory_intelligence_t *intel, std::string *err)>	observe;
	std::function<bool(const std::string &runtime,
		membrane_apply_live_t *live, std::string *err)>				read_live;
	std::function<bool(const std::string &runtime,
		membrane_apply_surface_t *surface)>							surface;
	/* the policy currently in force (re-read every tick) */
	std::function<bool(membrane_adaptive_policy_t *policy,
		std::string *config_error)>									load_policy;
	/* the J pipeline */
	std::function<bool(const membrane_apply_request_t &,
		membrane_apply_outcome_t *, std::string *err)>				execute;
}	membrane_adaptive_deps_t;

/* Real dependencies (system clock, advise collection, J apply, the
 * server.json policy). */
membrane_adaptive_deps_t	membrane_adaptive_default_deps(void);

/* server.json `adaptive` block -> validated pure policy. "" on success. */
std::string	membrane_adaptive_policy_from_config(
				const membrane_server_adaptive_t &cfg,
				membrane_adaptive_policy_t *out);
/* The inverse, for `membrane adaptive enable`. */
membrane_server_adaptive_t	membrane_adaptive_config_from_policy(
				const membrane_adaptive_policy_t &p);

class membrane_adaptive_controller_t
{
	public:
		explicit membrane_adaptive_controller_t(
			const membrane_adaptive_deps_t &deps);
		~membrane_adaptive_controller_t(void);

		/* One observation + decision (+ action in apply mode). Public so
		 * tests drive it directly with a fake clock. */
		void			tick(void);

		void			start(void);
		void			stop(void);			/* idempotent; joins */

		/* Thread-safe snapshot for `membrane adaptive status`. */
		nlohmann::json	status_json(void);

	private:
		void			run(void);
		void			publish(const membrane_adaptive_decision_t &d,
							const std::string &runtime,
							const nlohmann::json &last_action);

		membrane_adaptive_deps_t		deps_;
		std::mutex						mtx_;			/* guards state_/status_ */
		membrane_adaptive_policy_t		policy_;
		membrane_adaptive_state_t		*state_;		/* heap: large */
		bool							state_valid_;
		nlohmann::json					status_;
		nlohmann::json					last_action_;
		std::string						config_error_;
		int								interval_s_;

		std::thread						thread_;
		std::mutex						wake_mtx_;
		std::condition_variable			wake_cv_;
		std::atomic<bool>				stop_;
		std::atomic<bool>				running_;
};

#endif
