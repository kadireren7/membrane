#ifndef MEMBRANE_SERVER_CONFIG_H
# define MEMBRANE_SERVER_CONFIG_H

# include <cstdint>
# include <string>

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
