#ifndef MEMBRANE_DIRENT_COMPAT_H
# define MEMBRANE_DIRENT_COMPAT_H

/*
 * Mega Phase D, PR D5: <dirent.h> does not exist on Windows at all (a
 * real, first-attempt Windows CI compile failure: "Cannot open include
 * file: 'dirent.h'"). This project's own real usage
 * (src/backends/backend_file.c) is narrow: opendir/readdir/closedir
 * and struct dirent's own d_name field, iterating a directory's entries
 * in a plain while-loop -- no rewinddir, no telldir/seekdir, no d_type.
 *
 * Rather than touch backend_file.c's own real, working, tested call
 * sites, this header defines the SAME dirent type/function names,
 * backed by Win32's own native FindFirstFileA/FindNextFileA/FindClose
 * (the real, idiomatic Win32 directory-enumeration API) on Windows.
 * Both real functions in backend_file.c compile completely unchanged,
 * calling what looks like POSIX dirent but resolves to real Win32
 * primitives underneath. On every other platform, this header is a
 * pure passthrough to the real <dirent.h>.
 */

#ifdef _WIN32
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include "membrane/windows_lean.h"

struct dirent
{
	char	d_name[MAX_PATH];
};

typedef struct s_membrane_dir
{
	HANDLE			find_handle;
	WIN32_FIND_DATAA	find_data;
	int				have_pending;
	struct dirent	entry;
}	DIR;

static inline DIR	*opendir(const char *path)
{
	DIR			*d;
	char		pattern[MAX_PATH];

	d = (DIR *)calloc(1, sizeof(*d));
	if (d == NULL)
		return (NULL);
	/* FindFirstFileA needs a wildcard pattern, not a bare directory
	 * path -- "<path>\*" enumerates every entry directly inside it,
	 * matching opendir()+readdir()'s own real semantics. */
	snprintf(pattern, sizeof(pattern), "%s\\*", path);
	d->find_handle = FindFirstFileA(pattern, &d->find_data);
	if (d->find_handle == INVALID_HANDLE_VALUE)
	{
		free(d);
		return (NULL);
	}
	d->have_pending = 1;
	return (d);
}

static inline struct dirent	*readdir(DIR *d)
{
	if (!d->have_pending)
	{
		if (!FindNextFileA(d->find_handle, &d->find_data))
			return (NULL);
	}
	d->have_pending = 0;
	strncpy(d->entry.d_name, d->find_data.cFileName,
		sizeof(d->entry.d_name) - 1);
	d->entry.d_name[sizeof(d->entry.d_name) - 1] = '\0';
	return (&d->entry);
}

static inline int	closedir(DIR *d)
{
	if (d == NULL)
		return (-1);
	if (d->find_handle != INVALID_HANDLE_VALUE)
		FindClose(d->find_handle);
	free(d);
	return (0);
}

#else
# include <dirent.h>
#endif

#endif
