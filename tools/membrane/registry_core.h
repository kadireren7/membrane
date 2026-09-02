#ifndef MEMBRANE_REGISTRY_CORE_H
# define MEMBRANE_REGISTRY_CORE_H

# include <cstdint>
# include <string>
# include <vector>

/*
 * Mega Phase A, PR A2: the local model registry's pure core -- add/remove/
 * list/find, atomic file-backed persistence, and staleness detection. No
 * GGUF validation, no llama/ggml types anywhere in this module (Section 10
 * of the task: "Do not download models automatically"; this module also
 * never LOADS a model, valid or not -- it only remembers a name -> path
 * mapping plus cheap metadata the CALLER already extracted). Deliberately
 * llama-free in spirit (matches every other *_policy.h-style pure module
 * in this project) even though it is built only inside the
 * MEMBRANE_ENABLE_LLAMA block for now, purely because its one real
 * dependency (nlohmann::json, vendored under third_party/llama.cpp/vendor)
 * physically lives inside that submodule -- there is no llama.h/ggml.h
 * include here, and no llama_* symbol is ever referenced.
 *
 * Model IDENTIFICATION (Section 13 of the task): a registry entry's
 * file_size_bytes/file_mtime_ns are a cheap identity signature captured at
 * `add` time -- membrane_registry_check_entry() compares them against the
 * real file's current stat() result to decide whether cached metadata
 * (architecture/model_max_context) can still be trusted, WITHOUT ever
 * hashing a multi-gigabyte model file. A stale/missing/moved file is
 * reported as a distinct status, never silently treated as still valid.
 *
 * ATOMICITY (Section 12): membrane_registry_save() always writes to a
 * temp file in the SAME directory as the real target, then rename()s it
 * into place -- rename(2) on the same filesystem is atomic, so a reader
 * (or a crash) never observes a half-written registry file.
 */

typedef struct s_membrane_registry_entry
{
	std::string	name;				/* the user-chosen key, e.g. "qwen" */
	std::string	path;				/* canonical, absolute (Section 11) */
	std::string	basename;			/* safe basename of path, for display */
	std::string	arch_name;			/* "" if unknown at add time */
	uint64_t	model_max_context;	/* 0 if unknown */
	uint64_t	file_size_bytes;	/* real stat() size at add time */
	int64_t		file_mtime_ns;		/* real stat() mtime at add time,
									 * nanosecond-resolution POSIX time */
	int64_t		added_at_unix;		/* wall-clock time this entry was added,
									 * for `list`/`inspect` display only --
									 * never used for identity/staleness */
}	membrane_registry_entry_t;

typedef struct s_membrane_registry
{
	std::vector<membrane_registry_entry_t>	entries;
}	membrane_registry_t;

typedef struct s_membrane_registry_error
{
	bool		set;
	std::string	code;		/* stable machine-readable code, e.g.
							 * "DUPLICATE_NAME", "NOT_FOUND", "IO_ERROR" */
	std::string	message;	/* human-readable detail */
}	membrane_registry_error_t;

/* Section 10: XDG_DATA_HOME/membrane/models.json, falling back to
 * $HOME/.local/share/membrane/models.json when XDG_DATA_HOME is unset --
 * the exact XDG Base Directory fallback rule. Returns an empty string
 * (caller must treat as a hard error) only if neither XDG_DATA_HOME nor
 * HOME is set -- no invented default. */
std::string	membrane_registry_default_path(void);

/* membrane_registry_default_path(), but first honoring a MEMBRANE_MODELS_
 * PATH environment variable override when set and non-empty. The ONE
 * path-resolution function every real command (`membrane model ...`,
 * `membrane serve`) uses -- never two independently-drifting copies of
 * this same three-way fallback (this override exists specifically so
 * tests/CI never touch a real user registry; see model_cmd.cpp's
 * original standalone version of this function, which this replaces). */
std::string	membrane_registry_resolve_path(void);

/* Loads registry_path into *out. A NONEXISTENT file is NOT an error --
 * *out is left as an empty, valid registry (Section 10: the first `model
 * add` on a fresh machine must not require pre-creating the file/
 * directory). A file that exists but fails to parse as the expected JSON
 * shape IS an error (fails closed rather than silently discarding
 * whatever the user already registered). */
bool	membrane_registry_load(const std::string &registry_path,
			membrane_registry_t *out, membrane_registry_error_t *err);

/* Atomic write (see this header's own top comment): temp file in the same
 * directory as registry_path, fsync'd, then rename()'d into place.
 * Creates registry_path's parent directory (mkdir -p semantics) if it
 * does not exist yet. */
bool	membrane_registry_save(const std::string &registry_path,
			const membrane_registry_t &reg, membrane_registry_error_t *err);

/* Rejects a duplicate NAME (DUPLICATE_NAME, regardless of path -- a name
 * is the registry's own primary key). A duplicate PATH under a genuinely
 * different name is allowed (a harmless alias, Section 12's own "handle,"
 * not necessarily "reject," language) -- never silently merged or
 * renamed. entry.path is expected already canonicalized by the caller
 * (this module does no filesystem access itself, kept pure/testable). */
bool	membrane_registry_add(membrane_registry_t *reg,
			const membrane_registry_entry_t &entry,
			membrane_registry_error_t *err);

/* Returns false (NOT_FOUND) if no entry has this exact name. */
bool	membrane_registry_remove(membrane_registry_t *reg,
			const std::string &name, membrane_registry_error_t *err);

/* Returns NULL if not found -- pointer into reg's own storage, valid
 * until the next add/remove call. */
const membrane_registry_entry_t	*membrane_registry_find(
			const membrane_registry_t &reg, const std::string &name);

# define MEMBRANE_REGISTRY_CHECK_OK			"OK"
# define MEMBRANE_REGISTRY_CHECK_MISSING		"MISSING"		/* file no
															 * longer exists
															 * at path */
# define MEMBRANE_REGISTRY_CHECK_MODIFIED		"MODIFIED"		/* file
															 * exists but
															 * size/mtime
															 * differ from
															 * the recorded
															 * identity */
# define MEMBRANE_REGISTRY_CHECK_UNREADABLE	"UNREADABLE"	/* stat()
															 * itself
															 * failed for a
															 * reason other
															 * than "does
															 * not exist"
															 * (permissions,
															 * a symlink
															 * loop, etc.) */

enum e_membrane_registry_stat_status
{
	MEMBRANE_REGISTRY_STAT_OK = 0,		/* stat() succeeded, fields valid */
	MEMBRANE_REGISTRY_STAT_MISSING,		/* stat() failed with ENOENT */
	MEMBRANE_REGISTRY_STAT_ERROR,		/* stat() failed for any other
										 * reason (EACCES, ELOOP, etc.) */
};

/* Pure comparison -- takes the REAL current stat() outcome as plain
 * parameters (the caller does the actual stat() call and classifies its
 * errno; this module has no filesystem access of its own, kept testable
 * with synthetic inputs). current_size_bytes/current_mtime_ns are only
 * meaningful when stat_status == MEMBRANE_REGISTRY_STAT_OK. Returns one
 * of the MEMBRANE_REGISTRY_CHECK_* codes above. */
const char	*membrane_registry_check_identity(
				const membrane_registry_entry_t &entry,
				e_membrane_registry_stat_status stat_status,
				uint64_t current_size_bytes, int64_t current_mtime_ns);

#endif
