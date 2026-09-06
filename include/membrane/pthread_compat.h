#ifndef MEMBRANE_PTHREAD_COMPAT_H
# define MEMBRANE_PTHREAD_COMPAT_H

/*
 * Mega Phase D, PR D5: pthread.h does not exist on Windows at all (a
 * real, first-attempt Windows CI compile failure: "Cannot open include
 * file: 'pthread.h'"). This project's own real pthread usage
 * (src/store/store.c, src/quant/quant_simd.c) is narrow: one non-
 * recursive mutex + one condition variable (init/lock/unlock/destroy/
 * wait/signal/broadcast, no attributes, no timed waits, no recursive/
 * rwlock features), and a simple "spawn N worker threads, join them
 * all" pattern (pthread_create/pthread_join, no thread attributes, no
 * detached threads, no caller ever reads a thread's own return value --
 * every real call site passes NULL for pthread_join's retval).
 *
 * Rather than touch either file's own real, working, tested call
 * sites, this header defines the SAME pthread type/function names,
 * backed by Win32's own native primitives on Windows -- SRWLOCK +
 * CONDITION_VARIABLE (the documented, idiomatic Win32 replacement for
 * a plain mutex + condvar, available since Windows Vista) and
 * _beginthreadex (the CRT-safe alternative to raw CreateThread,
 * required whenever CRT functions are used from the spawned thread,
 * which both real worker functions here do). Both real callers compile
 * completely unchanged on Windows, calling what looks like pthread but
 * resolves to real Win32 primitives underneath. On every other
 * platform, this header is a pure passthrough to the real <pthread.h>.
 *
 * SRWLOCK's own contract requires SleepConditionVariableSRW() to be
 * called while the lock is held in EXCLUSIVE mode -- guaranteed here
 * by construction: this shim exposes only exclusive-mode locking
 * (AcquireSRWLockExclusive/ReleaseSRWLockExclusive), never the shared-
 * mode API, matching store.c's own real usage (it never takes a
 * shared/read lock anywhere).
 */

#ifdef _WIN32
# include <process.h>
# include <stdlib.h>
# include <windows.h>

typedef SRWLOCK			pthread_mutex_t;
typedef CONDITION_VARIABLE	pthread_cond_t;
typedef HANDLE				pthread_t;

static inline int	pthread_mutex_init(pthread_mutex_t *m, const void *attr)
{
	(void)attr;
	InitializeSRWLock(m);
	return (0);
}

static inline int	pthread_mutex_lock(pthread_mutex_t *m)
{
	AcquireSRWLockExclusive(m);
	return (0);
}

static inline int	pthread_mutex_unlock(pthread_mutex_t *m)
{
	ReleaseSRWLockExclusive(m);
	return (0);
}

/* SRWLOCK needs no explicit destruction (no OS handle/resource to
 * free) -- a real, documented Win32 property, not an omission. */
static inline int	pthread_mutex_destroy(pthread_mutex_t *m)
{
	(void)m;
	return (0);
}

static inline int	pthread_cond_init(pthread_cond_t *c, const void *attr)
{
	(void)attr;
	InitializeConditionVariable(c);
	return (0);
}

static inline int	pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m)
{
	return (SleepConditionVariableSRW(c, m, INFINITE, 0) ? 0 : -1);
}

static inline int	pthread_cond_signal(pthread_cond_t *c)
{
	WakeConditionVariable(c);
	return (0);
}

static inline int	pthread_cond_broadcast(pthread_cond_t *c)
{
	WakeAllConditionVariable(c);
	return (0);
}

/* CONDITION_VARIABLE needs no explicit destruction either, same real
 * Win32 property as SRWLOCK above. */
static inline int	pthread_cond_destroy(pthread_cond_t *c)
{
	(void)c;
	return (0);
}

typedef struct s_membrane_pthread_trampoline_ctx
{
	void	*(*fn)(void *);
	void	*arg;
}	membrane_pthread_trampoline_ctx_t;

static inline unsigned __stdcall	membrane_pthread_trampoline(void *raw)
{
	membrane_pthread_trampoline_ctx_t	*ctx
			= (membrane_pthread_trampoline_ctx_t *)raw;
	void	*(*fn)(void *) = ctx->fn;
	void	*arg = ctx->arg;

	free(ctx);
	fn(arg);
	return (0);
}

static inline int	pthread_create(pthread_t *thread, const void *attr,
					void *(*start_routine)(void *), void *arg)
{
	membrane_pthread_trampoline_ctx_t	*ctx;

	(void)attr;
	ctx = (membrane_pthread_trampoline_ctx_t *)
			malloc(sizeof(*ctx));
	if (ctx == NULL)
		return (-1);
	ctx->fn = start_routine;
	ctx->arg = arg;
	*thread = (HANDLE)_beginthreadex(NULL, 0, membrane_pthread_trampoline,
			ctx, 0, NULL);
	if (*thread == NULL)
	{
		free(ctx);
		return (-1);
	}
	return (0);
}

/* retval is always NULL at every real call site in this codebase (the
 * one caller, quant_simd.c's qpool_run(), reads each worker's own
 * result through shared memory instead) -- return-value propagation is
 * therefore not implemented; retval, if given, is always set to NULL
 * rather than left uninitialized, a disclosed, deliberate limitation
 * rather than a silent gap. */
static inline int	pthread_join(pthread_t thread, void **retval)
{
	if (retval != NULL)
		*retval = NULL;
	WaitForSingleObject(thread, INFINITE);
	CloseHandle(thread);
	return (0);
}

#else
# include <pthread.h>
#endif

#endif
