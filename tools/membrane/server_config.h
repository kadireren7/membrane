#ifndef MEMBRANE_SERVER_CONFIG_H
# define MEMBRANE_SERVER_CONFIG_H

# include <cstdint>
# include <string>
# include <vector>

/*
 * Mega Phase B, PR B1: a small, persistent server config -- Section 8 of
 * the task ("Keep minimal. Do not turn this into a giant config
 * system."). `membrane serve` (no --port/--bind flags) reads this so a
 * `membrane service install`-generated systemd unit's own ExecStart can
 * stay simply "<exec_path> serve" -- changing the listen address/port/
 * default model later never requires regenerating the unit file.
 * Explicit --port/--bind CLI flags always override the config (same
 * "explicit beats implicit" convention as every other flag in this
 * project).
 *
 * schema_version (Section 44): if a future phase needs an incompatible
 * config shape, it bumps this field and adds explicit migration/rejection
 * logic -- an old config is never silently reinterpreted under a new
 * schema.
 */

/*
 * Milestone J (active control): the explicit, user-requested LOAD/REQUEST
 * overrides `membrane apply` can persist, and the only way a native
 * service's automatic context/GPU-layer/KV choices are ever changed.
 * Every field is optional (has_* == false = "automatic", the pre-J
 * behavior, byte-for-byte). Consumed by server.cpp's fill_auto_opts()
 * and request paths; never written by anything but `membrane apply`
 * (or the user editing server.json). Values are validated on load --
 * an invalid overrides block fails the whole config load (fail closed)
 * instead of being partially honored.
 */
typedef struct s_membrane_server_overrides
{
	bool		has_context = false;
	uint64_t	context = 0;		/* fixed context tokens per request */
	bool		has_kv_precision = false;
	std::string	kv_precision;		/* native|q8|q5 */
	bool		has_kv_placement = false;
	std::string	kv_placement;		/* default|cpu|gpu|auto */
	bool		has_gpu_layers = false;
	int32_t		gpu_layers = 0;			/* -1 = all, -2 = auto, N >= 0 */
}	membrane_server_overrides_t;

bool	membrane_server_overrides_empty(const membrane_server_overrides_t &o);
bool	membrane_server_overrides_equal(const membrane_server_overrides_t &a,
			const membrane_server_overrides_t &b);
/* "" when valid, else a short reason. */
std::string	membrane_server_overrides_validate(
			const membrane_server_overrides_t &o);

/*
 * Milestone K: the persisted adaptive-runtime policy (docs/adaptive-
 * runtime.md). OPT-IN: a missing block, or enabled=false, means adaptive
 * control does nothing. Written by `membrane adaptive enable|disable`,
 * never required to be hand-edited. Value semantics (mode names, class
 * names, ranges) are validated by adaptive.h when the controller builds
 * its policy; here only the JSON types are checked, and a bad type fails
 * the whole load (fail closed).
 */
typedef struct s_membrane_server_adaptive
{
	bool		present = false;
	bool		enabled = false;
	std::string	mode = "recommend";			/* observe|recommend|apply */
	std::string	runtime = "membrane-native";
	int32_t		interval_s = 30;
	int32_t		cooldown_s = 600;
	std::vector<std::string>	allow = {"context", "kv_precision"};
	std::string	min_pressure = "warning";
	int32_t		max_actions_per_window = 2;
	bool		auto_restore = false;
}	membrane_server_adaptive_t;

typedef struct s_membrane_server_config
{
	int32_t		schema_version;		/* == 1 today */
	std::string	listen_address;		/* default "127.0.0.1" */
	int32_t		port;				/* default 8642 */
	std::string	default_model;		/* "" = none configured (Section 9:
									 * "do not force one" -- the server
									 * starts healthy with no model
									 * loaded either way) */
	std::string	log_level;			/* "info" default -- currently
									 * informational only; no logger
									 * exists yet to consult it */
	membrane_server_overrides_t	overrides;	/* Milestone J; absent = all
									 * automatic */
	membrane_server_adaptive_t	adaptive;	/* Milestone K; absent = off */
}	membrane_server_config_t;

typedef struct s_membrane_server_config_error
{
	bool		set;
	std::string	code;
	std::string	message;
}	membrane_server_config_error_t;

# define MEMBRANE_SERVER_CONFIG_SCHEMA_VERSION	1

/* Section 8: $XDG_CONFIG_HOME/membrane/server.json, falling back to
 * $HOME/.config/membrane/server.json -- the standard XDG fallback rule,
 * same pattern as registry_core.h's own path resolution.
 * MEMBRANE_SERVER_CONFIG_PATH overrides it (tests/CI never touch a real
 * user config). */
std::string	membrane_server_config_resolve_path(void);

/* A default-valued config -- used both as the fallback when no config
 * file exists yet and as the seed `membrane service install` writes on
 * first use. */
membrane_server_config_t	membrane_server_config_defaults(void);

/* A nonexistent file is NOT an error -- *out is left at defaults
 * (Section 8: this project never requires a config file to exist before
 * `membrane serve` works). A file that exists but fails to parse as the
 * expected shape, or carries a schema_version this build does not
 * understand, IS an error (fails closed rather than silently guessing).
 */
bool	membrane_server_config_load(const std::string &path,
			membrane_server_config_t *out,
			membrane_server_config_error_t *err);

/* Atomic write (temp file + fsync + rename in the same directory),
 * exactly like registry_core.h's membrane_registry_save() -- a reader
 * (or a crash mid-write) never observes a half-written config. */
bool	membrane_server_config_save(const std::string &path,
			const membrane_server_config_t &cfg,
			membrane_server_config_error_t *err);

#endif
