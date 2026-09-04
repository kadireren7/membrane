#ifndef MEMBRANE_FS_UTIL_H
# define MEMBRANE_FS_UTIL_H

# include <string>

/*
 * Mega Phase B, PR B1: the "mkdir -p the parent, write to a temp file in
 * the same directory, fsync, rename into place" pattern -- shared by
 * registry_core.cpp (PR A2), server_config.cpp, and service_cmd.cpp's
 * own unit-file write (both PR B1) -- extracted here once this became a
 * real third independent copy, rather than left duplicated a third time.
 */

typedef struct s_membrane_fs_error
{
	bool		set;
	std::string	code;		/* "IO_ERROR" for every failure this module
							 * can produce */
	std::string	message;
}	membrane_fs_error_t;

/* mkdir -p semantics for every path component up to and including `dir`
 * itself. EEXIST on any component is not an error. */
bool	membrane_mkdir_parents(const std::string &dir,
			membrane_fs_error_t *err);

/* Creates path's parent directory (mkdir -p) if needed, writes `content`
 * to a temp file in the SAME directory, fsyncs it, then rename()s it
 * into place -- a reader (or a crash mid-write) never observes a
 * half-written file. */
bool	membrane_atomic_write_file(const std::string &path,
			const std::string &content, membrane_fs_error_t *err);

#endif
