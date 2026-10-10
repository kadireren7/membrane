#include "apply_cmd.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <sys/stat.h>
#ifndef _WIN32
# include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "action.h"
#include "advise_cmd.h"
#include "cli_shared.h"
#include "fs_util.h"
#include "observation.h"
#include "plan_cmd.h"
#include "product_cli.h"
#include "runtime_capabilities.h"
#include "runtime_ollama.h"
#include "server_config.h"
#include "status_client.h"

using json = nlohmann::json;

/* See apply_cmd.h's top comment and docs/active-control.md. */

/* ------------------------------------------------------------------ */
/* Options                                                              */
/* ------------------------------------------------------------------ */

struct s_apply_opts
{
	std::string	runtime = MEMBRANE_RUNTIME_ID_NATIVE;
	bool		yes = false;
	bool		reload = false;
	std::string	model;
	membrane_action_request_t	req;
	bool		explicit_intent = false;	/* any action-selecting flag */
};

static void	set_str(char *dst, size_t size, const std::string &v)
{
	snprintf(dst, size, "%s", v.c_str());
}

static bool	is_uint(const std::string &s)
{
	if (s.empty() || s.size() > 9)
		return (false);
	for (char c : s)
		if (c < '0' || c > '9')
			return (false);
	return (true);
}

static bool	parse_apply_opts(const std::vector<std::string> &args,
				s_apply_opts *o, std::string *err)
{
	memset(&o->req, 0, sizeof(o->req));
	for (size_t i = 0; i < args.size(); ++i)
	{
		const std::string	&a = args[i];
		bool				has_val = i + 1 < args.size();

		if (a == "--yes" || a == "-y")
			o->yes = true;
		else if (a == "--reload")
		{
			o->reload = true;
			o->explicit_intent = true;
		}
		else if (a == "--runtime" && has_val)
			o->runtime = args[++i];
		else if (a == "--model" && has_val)
			o->model = args[++i];
		else if (a == "--context" && has_val)
		{
			std::string	v = args[++i];

			if (v == "auto")
				v = "automatic";
			else if (!is_uint(v))
			{
				*err = "--context must be a positive integer or 'auto'";
				return (false);
			}
			set_str(o->req.context, sizeof(o->req.context), v);
			o->explicit_intent = true;
		}
		else if (a == "--kv" && has_val)
		{
			std::string	v = args[++i];

			if (v == "auto" || v == "adaptive")
				v = "automatic";
			else if (v != "native" && v != "q8" && v != "q5")
			{
				*err = "--kv must be native|q8|q5|auto";
				return (false);
			}
			set_str(o->req.kv_precision, sizeof(o->req.kv_precision), v);
			o->explicit_intent = true;
		}
		else if (a == "--kv-placement" && has_val)
		{
			std::string	v = args[++i];

			if (v == "auto")
				v = "automatic";
			else if (v != "default" && v != "cpu" && v != "gpu")
			{
				*err = "--kv-placement must be default|cpu|gpu|auto";
				return (false);
			}
			set_str(o->req.kv_placement, sizeof(o->req.kv_placement), v);
			o->explicit_intent = true;
		}
		else if (a == "--gpu-layers" && has_val)
		{
			std::string	v = args[++i];

			if (v == "auto")
				v = "automatic";
			else if (v != "all" && !is_uint(v))
			{
				*err = "--gpu-layers must be all|auto|N";
				return (false);
			}
			set_str(o->req.gpu_layers, sizeof(o->req.gpu_layers), v);
			o->explicit_intent = true;
		}
		else if ((a == "--activate" || a == "--unload" || a == "--pin"
				|| a == "--unpin") && has_val)
		{
			std::string	v = args[++i];
			char		*dst = a == "--activate" ? o->req.activate
				: a == "--unload" ? o->req.unload
				: a == "--pin" ? o->req.pin : o->req.unpin;

			if (v.empty() || v.size() >= MEMBRANE_ACTION_NAME_MAX
				|| v[0] == '-')
			{
				*err = a + " needs a model name";
				return (false);
			}
			set_str(dst, MEMBRANE_ACTION_NAME_MAX, v);
			o->explicit_intent = true;
		}
		else
		{
			*err = "unknown option '" + a + "'";
			return (false);
		}
	}
	if (o->model.size() >= MEMBRANE_ACTION_NAME_MAX)
	{
		*err = "--model name too long";
		return (false);
	}
	set_str(o->req.model, sizeof(o->req.model), o->model);
	o->req.reload = o->reload ? 1 : 0;
	set_str(o->req.runtime_id, sizeof(o->req.runtime_id), o->runtime);
	return (true);
}

/* ------------------------------------------------------------------ */
/* Backends                                                             */
/* ------------------------------------------------------------------ */

class apply_backend
{
	public:
		virtual ~apply_backend(void) {}
		virtual void	fill_surface(membrane_apply_surface_t *s) = 0;
		virtual bool	read_live(membrane_apply_live_t *live,
							std::string *err) = 0;
		virtual bool	execute(const membrane_action_t &a, std::string *after,
							std::string *err) = 0;
		virtual bool	rollback(const membrane_action_t &a,
							std::string *err) = 0;
		/* Planner v2 check of the FINAL overrides; native only. */
		virtual bool	validate(const membrane_apply_live_t &live,
							const s_apply_opts &o,
							membrane_action_validation_t *v)
		{
			(void)live; (void)o;
			v->checked = 0;
			snprintf(v->detail, sizeof(v->detail), "no Planner v2 "
				"validation exists for this runtime");
			return (false);
		}
};

static bool	is_loopback_addr(const std::string &a)
{
	return (a == "127.0.0.1" || a == "::1" || a == "localhost");
}

static void	copy_val(char *dst, size_t size, const std::string &v)
{
	snprintf(dst, size, "%s", v.c_str());
}

/* ---- native ------------------------------------------------------- */

class native_backend : public apply_backend
{
	public:
		std::string						config_path;
		membrane_server_config_t		cfg;
		const membrane_apply_io_t		*io;

		native_backend(const membrane_apply_io_t *io_in) : io(io_in)
		{
			config_path = membrane_server_config_resolve_path();
			cfg = membrane_server_config_defaults();
		}

		void	fill_surface(membrane_apply_surface_t *s)
		{
			memset(s, 0, sizeof(*s));
			for (int i = 0; i < MEMBRANE_ACTION_CODE_COUNT; ++i)
			{
				s->state[i] = MEMBRANE_CAPABILITY_SUPPORTED;
				s->reversible[i] = 1;
			}
			s->requires_reload[MEMBRANE_ACTION_SET_KV_PRECISION] = 1;
			s->requires_reload[MEMBRANE_ACTION_SET_KV_PLACEMENT] = 1;
			s->requires_reload[MEMBRANE_ACTION_SET_GPU_LAYERS] = 1;
		}

		static std::string	ov_ctx(const membrane_server_overrides_t &o)
		{
			return (o.has_context ? std::to_string(o.context) : "automatic");
		}
		static std::string	ov_gpu(const membrane_server_overrides_t &o)
		{
			if (!o.has_gpu_layers)
				return ("automatic");
			return (o.gpu_layers == -1 ? "all" : o.gpu_layers == -2 ? "auto"
				: std::to_string(o.gpu_layers));
		}

		bool	load_cfg(std::string *err)
		{
			membrane_server_config_error_t	cerr;

			if (config_path.empty())
			{
				*err = "no server config path (set HOME or "
					"MEMBRANE_SERVER_CONFIG_PATH)";
				return (false);
			}
			if (!membrane_server_config_load(config_path, &cfg, &cerr))
			{
				*err = "server config unreadable: " + cerr.message;
				return (false);
			}
			return (true);
		}

		bool	read_live(membrane_apply_live_t *live, std::string *err)
		{
			json	status;

			memset(live, 0, sizeof(*live));
			if (!load_cfg(err))
				return (false);
			if (!is_loopback_addr(cfg.listen_address))
			{
				*err = "the configured listen address '" + cfg.listen_address
					+ "' is not loopback; active control is local-only";
				return (false);
			}
			live->ctx_known = live->kv_precision_known
				= live->kv_placement_known = live->gpu_layers_known = 1;
			copy_val(live->ctx, sizeof(live->ctx), ov_ctx(cfg.overrides));
			copy_val(live->kv_precision, sizeof(live->kv_precision),
				cfg.overrides.has_kv_precision ? cfg.overrides.kv_precision
					: "automatic");
			copy_val(live->kv_placement, sizeof(live->kv_placement),
				cfg.overrides.has_kv_placement ? cfg.overrides.kv_placement
					: "automatic");
			copy_val(live->gpu_layers, sizeof(live->gpu_layers),
				ov_gpu(cfg.overrides));
			copy_val(live->default_model, sizeof(live->default_model),
				cfg.default_model);
			if (!membrane_fetch_server_status(cfg.listen_address, cfg.port,
					&status))
				return (true);			/* reachable stays 0 */
			live->runtime_reachable = 1;
			if (status.contains("default_model")
				&& status["default_model"].is_string())
				copy_val(live->default_model, sizeof(live->default_model),
					status["default_model"].get<std::string>());
			if (status.contains("resident_models")
				&& status["resident_models"].is_array())
				for (const auto &m : status["resident_models"])
				{
					if (live->resident_count >= MEMBRANE_ACTION_MAX_RESIDENT
						|| !m.is_object() || !m.contains("model")
						|| !m["model"].is_string())
						continue ;
					membrane_apply_resident_t	*r
						= &live->resident[live->resident_count++];
					bool	pinned = m.value("pinned", false);
					bool	evictable = m.value("evictable", false);

					memset(r, 0, sizeof(*r));
					copy_val(r->name, sizeof(r->name),
						m["model"].get<std::string>());
					r->activity_known = m.contains("pinned")
						&& m.contains("evictable");
					r->pinned = pinned;
					r->generating = !pinned && !evictable;
					r->size_bytes = m.value("estimated_model_bytes",
							(uint64_t)0);
				}
			return (true);
		}

		bool	set_override(const membrane_action_t &a,
					const std::string &value, std::string *err)
		{
			std::string	v = value;
			bool		clear = v == "automatic";

			if (!load_cfg(err))
				return (false);
			membrane_server_overrides_t	&o = cfg.overrides;

			switch (a.code)
			{
				case MEMBRANE_ACTION_SET_CONTEXT:
					o.has_context = !clear;
					o.context = clear ? 0 : strtoull(v.c_str(), NULL, 10);
					break ;
				case MEMBRANE_ACTION_SET_KV_PRECISION:
					o.has_kv_precision = !clear;
					o.kv_precision = clear ? "" : v;
					break ;
				case MEMBRANE_ACTION_SET_KV_PLACEMENT:
					o.has_kv_placement = !clear;
					o.kv_placement = clear ? "" : v;
					break ;
				default:
					o.has_gpu_layers = !clear;
					o.gpu_layers = clear ? 0 : v == "all" ? -1
						: v == "auto" ? -2 : atoi(v.c_str());
					break ;
			}
			membrane_server_config_error_t	cerr;

			if (!membrane_server_config_save(config_path, cfg, &cerr))
			{
				*err = "could not write server config: " + cerr.message;
				return (false);
			}
			return (true);
		}

		bool	execute(const membrane_action_t &a, std::string *after,
					std::string *err)
		{
			membrane_activate_result_t	act;
			membrane_pin_result_t		pin;
			membrane_unload_result_t	unl;
			std::string					m = a.model;

			*after = a.requested;
			switch (a.code)
			{
				case MEMBRANE_ACTION_SET_CONTEXT:
				case MEMBRANE_ACTION_SET_KV_PRECISION:
				case MEMBRANE_ACTION_SET_KV_PLACEMENT:
				case MEMBRANE_ACTION_SET_GPU_LAYERS:
					return (set_override(a, a.requested, err));
				case MEMBRANE_ACTION_UNLOAD_MODEL:
					if (!membrane_unload_model(cfg.listen_address, cfg.port,
							m, &unl))
					{
						*err = "service unreachable";
						return (false);
					}
					if (!unl.ok)
					{
						*err = unl.error_code + ": " + unl.error_message;
						return (false);
					}
					return (true);
				case MEMBRANE_ACTION_PIN_MODEL:
				case MEMBRANE_ACTION_UNPIN_MODEL:
					if (!membrane_set_model_pin(cfg.listen_address, cfg.port,
							m, a.code == MEMBRANE_ACTION_PIN_MODEL, &pin))
					{
						*err = "service unreachable";
						return (false);
					}
					if (!pin.ok)
					{
						*err = pin.error_code + ": " + pin.error_message;
						return (false);
					}
					return (true);
				default:
					break ;
			}
			/* ACTIVATE_MODEL. A reload first unloads the (unpinned,
			 * idle) model so the next load re-reads the applied
			 * overrides; a failed activate is restored by the engine. */
			if (a.restore_on_failure
				&& strcmp(a.requested, "reloaded") == 0)
			{
				if (!membrane_unload_model(cfg.listen_address, cfg.port, m,
						&unl))
				{
					*err = "service unreachable";
					return (false);
				}
				if (!unl.ok && unl.error_code != "MODEL_NOT_RESIDENT")
				{
					*err = unl.error_code + ": " + unl.error_message;
					return (false);
				}
			}
			if (!membrane_activate_model(cfg.listen_address, cfg.port, m, &act))
			{
				*err = "service unreachable";
				return (false);
			}
			if (!act.ok)
			{
				*err = act.error_code + ": " + act.error_message;
				return (false);
			}
			*after = "resident";
			return (true);
		}

		bool	rollback(const membrane_action_t &a, std::string *err)
		{
			membrane_activate_result_t	act;
			membrane_pin_result_t		pin;
			membrane_unload_result_t	unl;

			switch (a.code)
			{
				case MEMBRANE_ACTION_SET_CONTEXT:
				case MEMBRANE_ACTION_SET_KV_PRECISION:
				case MEMBRANE_ACTION_SET_KV_PLACEMENT:
				case MEMBRANE_ACTION_SET_GPU_LAYERS:
					return (set_override(a, a.rollback_value, err));
				case MEMBRANE_ACTION_PIN_MODEL:
				case MEMBRANE_ACTION_UNPIN_MODEL:
					if (!membrane_set_model_pin(cfg.listen_address, cfg.port,
							a.model, a.code != MEMBRANE_ACTION_PIN_MODEL,
							&pin) || !pin.ok)
					{
						*err = "could not restore pin state";
						return (false);
					}
					return (true);
				case MEMBRANE_ACTION_UNLOAD_MODEL:
				case MEMBRANE_ACTION_ACTIVATE_MODEL:
					if (!membrane_activate_model(cfg.listen_address, cfg.port,
							a.model, &act) || !act.ok)
					{
						*err = "could not reload '" + std::string(a.model)
							+ "'" + (act.error_message.empty() ? ""
								: ": " + act.error_message);
						return (false);
					}
					return (true);
				default:
					break ;
			}
			(void)unl;
			*err = "no rollback for this action";
			return (false);
		}

		/* The model Planner v2 checks the final overrides against: the
		 * explicit --model, else the single resident model, else the
		 * configured default. None => validation unavailable. */
		static std::string	pick_model(const membrane_apply_live_t &live,
								const s_apply_opts &o)
		{
			if (!o.model.empty())
				return (o.model);
			if (live.resident_count == 1)
				return (live.resident[0].name);
			return (live.default_model);
		}

		bool	validate(const membrane_apply_live_t &live,
					const s_apply_opts &o, membrane_action_validation_t *v)
		{
			std::string							model = pick_model(live, o);
			membrane_plan_override_request_t	ov;
			membrane_plan_override_check_t		chk;
			auto	pick = [](const char *req, const char *cur)
					-> std::string
			{
				std::string	r = req;

				if (!r.empty())
					return (r == "automatic" ? "" : r);
				std::string	c = cur;

				return (c == "automatic" ? "" : c);
			};
			std::string	ctx = pick(o.req.context, live.ctx);
			std::string	kv = pick(o.req.kv_precision, live.kv_precision);
			std::string	pl = pick(o.req.kv_placement, live.kv_placement);
			std::string	gl = pick(o.req.gpu_layers, live.gpu_layers);

			if (model.empty())
			{
				snprintf(v->detail, sizeof(v->detail), "no model to check "
					"against (none resident, no default); pass --model NAME");
				return (false);
			}
			if (!ctx.empty())
			{
				ov.has_ctx = true;
				ov.ctx = strtoull(ctx.c_str(), NULL, 10);
			}
			if (!kv.empty())
			{
				ov.has_kv = true;
				ov.kv_precision = kv;
			}
			if (!pl.empty())
			{
				ov.has_placement = true;
				ov.placement = pl;
			}
			if (!gl.empty())
			{
				ov.has_gpu = true;
				ov.gpu_layers = gl == "all" ? -1 : gl == "auto" ? -2
					: atoi(gl.c_str());
			}
			bool	ran = io != NULL && io->planner_check
				? io->planner_check(model, ov, &chk)
				: membrane_plan_check_overrides(model, ov, &chk);

			if (!ran)
			{
				snprintf(v->detail, sizeof(v->detail), "%.200s",
					chk.detail.c_str());
				return (false);
			}
			v->checked = chk.checked ? 1 : 0;
			v->feasible = chk.feasible ? 1 : 0;
			snprintf(v->detail, sizeof(v->detail), "%.150s (checked "
				"against '%.40s')", chk.detail.c_str(), model.c_str());
			return (true);
		}
};

/* ---- ollama ------------------------------------------------------- */

class ollama_backend : public apply_backend
{
	public:
		void	fill_surface(membrane_apply_surface_t *s)
		{
			static const struct { membrane_action_code_t c; const char *why; }
				why[] = {
				{MEMBRANE_ACTION_SET_CONTEXT, "num_ctx is a per-request "
					"option; Ollama has no persistent context setting"},
				{MEMBRANE_ACTION_SET_GPU_LAYERS, "num_gpu is a per-request "
					"option; there is no persistent setting"},
				{MEMBRANE_ACTION_SET_KV_PRECISION, "OLLAMA_KV_CACHE_TYPE is "
					"a server start-time environment variable, not an API"},
				{MEMBRANE_ACTION_SET_KV_PLACEMENT, "Ollama exposes no KV "
					"placement control"},
				{MEMBRANE_ACTION_ACTIVATE_MODEL, "loading a model needs a "
					"request to an inference endpoint; MEMBRANE never uses "
					"one to apply a setting"},
				{MEMBRANE_ACTION_PIN_MODEL, "keep_alive is per request; "
					"Ollama has no pin"},
				{MEMBRANE_ACTION_UNPIN_MODEL, "keep_alive is per request; "
					"Ollama has no pin"}};

			memset(s, 0, sizeof(*s));
			for (const auto &w : why)
			{
				s->state[w.c] = MEMBRANE_CAPABILITY_UNSUPPORTED;
				snprintf(s->reason[w.c], sizeof(s->reason[w.c]), "%s", w.why);
			}
			s->state[MEMBRANE_ACTION_UNLOAD_MODEL] = MEMBRANE_CAPABILITY_SUPPORTED;
			/* Reloading would need an inference-endpoint request: an
			 * unload cannot be undone by MEMBRANE. */
			s->reversible[MEMBRANE_ACTION_UNLOAD_MODEL] = 0;
		}

		bool	read_live(membrane_apply_live_t *live, std::string *err)
		{
			std::vector<membrane_ollama_process_model_t>	ps;
			membrane_runtime_error_t						rerr;

			memset(live, 0, sizeof(*live));
			if (!membrane_ollama_list_running(&ps, &rerr))
			{
				if (rerr.code == MEMBRANE_RUNTIME_ERR_UNAVAILABLE)
					return (true);			/* reachable stays 0 */
				*err = rerr.message;
				return (false);
			}
			live->runtime_reachable = 1;
			for (const auto &m : ps)
			{
				if (live->resident_count >= MEMBRANE_ACTION_MAX_RESIDENT)
					break ;
				membrane_apply_resident_t	*r
					= &live->resident[live->resident_count++];

				memset(r, 0, sizeof(*r));
				copy_val(r->name, sizeof(r->name),
					m.model.empty() ? m.name : m.model);
				r->activity_known = 0;	/* /api/ps says nothing about it */
				r->size_bytes = m.size_known ? m.size_bytes : 0;
			}
			return (true);
		}

		bool	execute(const membrane_action_t &a, std::string *after,
					std::string *err)
		{
			membrane_runtime_error_t	rerr;

			*after = "unloaded";
			if (a.code != MEMBRANE_ACTION_UNLOAD_MODEL)
			{
				*err = "unsupported action for Ollama";
				return (false);
			}
			if (!membrane_ollama_unload_model(a.model, &rerr))
			{
				*err = rerr.message;
				return (false);
			}
			/* Verify the effect instead of trusting the 2xx. */
			for (int i = 0; i < 15; ++i)
			{
				std::vector<membrane_ollama_process_model_t>	ps;
				bool	still = false;

				if (membrane_ollama_list_running(&ps, &rerr))
				{
					for (const auto &m : ps)
						if (m.model == a.model || m.name == a.model)
							still = true;
					if (!still)
						return (true);
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
			*after = "still loaded";
			*err = "Ollama accepted the unload request but still lists '"
				+ std::string(a.model) + "' as loaded";
			return (false);
		}

		bool	rollback(const membrane_action_t &, std::string *err)
		{
			*err = "Ollama actions cannot be rolled back by MEMBRANE";
			return (false);
		}
};

/* ------------------------------------------------------------------ */
/* Audit log (J12)                                                      */
/* ------------------------------------------------------------------ */

std::string	membrane_apply_audit_path(void)
{
	const char	*ov = getenv("MEMBRANE_APPLY_AUDIT_PATH");

	if (ov != NULL && ov[0] != '\0')
		return (ov);
	const char	*xdg = getenv("XDG_STATE_HOME");

	if (xdg != NULL && xdg[0] != '\0')
		return (std::string(xdg) + "/membrane/actions.jsonl");
	std::string	home = membrane_resolve_home_dir();

	return (home.empty() ? "" : home + "/.local/state/membrane/actions.jsonl");
}

#define AUDIT_MAX_BYTES	(1024 * 1024)

/* One JSON line per executed apply. Records what changed and the
 * outcome only -- never prompt text, generated content or secrets (none
 * of which this command ever sees). Bounded: past 1 MiB the file is
 * rotated to <path>.1 (one generation kept). Best effort: a failure to
 * log is reported, never fatal to an already-performed change. */
static bool	write_audit(const std::string &path, const json &record)
{
	struct stat	sb;

	if (path.empty())
		return (false);
	size_t		slash = path.rfind('/');
	std::string	dir = slash == std::string::npos ? "." : path.substr(0, slash);
	membrane_fs_error_t	fserr;

	if (!membrane_mkdir_parents(dir, &fserr))
		return (false);
	if (stat(path.c_str(), &sb) == 0 && sb.st_size > AUDIT_MAX_BYTES)
		rename(path.c_str(), (path + ".1").c_str());
	FILE	*f = fopen(path.c_str(), "ab");

	if (f == NULL)
		return (false);
#ifndef _WIN32
	chmod(path.c_str(), 0600);
#endif
	std::string	line = record.dump() + "\n";
	bool		ok = fwrite(line.data(), 1, line.size(), f) == line.size();

	fclose(f);
	return (ok);
}

/* ------------------------------------------------------------------ */
/* Rendering                                                            */
/* ------------------------------------------------------------------ */

static json	nullable(const char *s, bool known = true)
{
	return (known && s[0] != '\0' ? json(s) : json(nullptr));
}

static json	action_json(const membrane_action_t &a)
{
	json	j;

	j["code"] = membrane_action_code_name(a.code);
	j["runtime"] = a.runtime_id;
	j["model"] = nullable(a.model);
	j["requested"] = a.requested;
	j["current"] = nullable(a.current, a.current_known != 0);
	j["origin"] = membrane_action_origin_name(a.origin);
	j["source_recommendation"] = nullable(a.source_recommendation);
	j["reason"] = a.reason;
	j["capability"] = membrane_capability_state_name(a.capability);
	j["executable"] = a.executable != 0;
	j["risk"] = membrane_action_risk_name(a.risk);
	j["reversible"] = a.reversible != 0;
	j["rollback"] = {{"known", a.rollback_known != 0},
		{"value", nullable(a.rollback_value, a.rollback_known != 0)}};
	j["requires_reload"] = a.requires_reload != 0;
	j["mutates_state"] = a.mutates_state != 0;
	j["summary"] = a.summary;
	return (j);
}

static json	unapplied_json(const membrane_action_unapplied_t &u)
{
	return (json{{"source", u.source}, {"reason_code", u.reason_code},
		{"text", u.text}});
}

typedef struct s_apply_report
{
	std::string							runtime;
	bool								dry_run = true;
	bool								attempted = false;
	std::string							confirmation;	/* yes_flag|
									 * interactive|declined|
									 * not_interactive|not_needed */
	std::string							status;
	std::string							detail;
	std::vector<std::string>			notes;
	membrane_action_plan_t				plan;
	membrane_action_result_t			result;
	std::vector<std::string>			after;
	json								recs = json::array();
}	apply_report_t;

static json	report_json(const apply_report_t &r)
{
	json	j;
	json	acts = json::array();
	json	un = json::array();
	json	res_actions = json::array();

	for (size_t i = 0; i < r.plan.action_count; ++i)
		acts.push_back(action_json(r.plan.actions[i]));
	for (size_t i = 0; i < r.plan.unapplied_count; ++i)
		un.push_back(unapplied_json(r.plan.unapplied[i]));
	j["schema_version"] = MEMBRANE_ACTION_SCHEMA_VERSION;
	j["membrane_version"] = MEMBRANE_VERSION;
	j["mode"] = "apply";
	j["runtime"] = r.runtime;
	j["dry_run"] = r.dry_run;
	j["mutates_state"] = r.attempted;
	j["applied"] = r.status == "applied";
	j["status"] = r.status;
	j["detail"] = r.detail;
	j["confirmation"] = r.confirmation;
	j["recommendations"] = r.recs;
	j["actions"] = acts;
	j["unapplied"] = un;
	j["requires_reload"] = r.plan.any_requires_reload != 0;
	j["notes"] = r.notes;
	if (r.attempted)
	{
		for (size_t i = 0; i < r.plan.action_count; ++i)
			res_actions.push_back({{"index", i},
				{"code", membrane_action_code_name(r.plan.actions[i].code)},
				{"outcome", membrane_action_outcome_name(r.result.outcome[i])},
				{"error", r.result.error[i][0] ? json(r.result.error[i])
					: json(nullptr)},
				{"after", i < r.after.size() && !r.after[i].empty()
					? json(r.after[i]) : json(nullptr)}});
		j["result"] = {{"status", membrane_apply_status_name(
				r.result.status)}, {"detail", r.result.detail},
			{"actions", res_actions}};
	}
	else
		j["result"] = nullptr;
	return (j);
}

static void	print_human(const apply_report_t &r)
{
	printf("Apply (%s)\n", r.dry_run ? "dry run -- nothing was changed"
		: "executed");
	printf("  Runtime: %s\n", r.runtime.c_str());
	for (size_t i = 0; i < r.recs.size(); ++i)
		printf("  Recommendation: %s (%s)\n",
			r.recs[i]["code"].get<std::string>().c_str(),
			r.recs[i]["applicability"].get<std::string>().c_str());
	printf("\nProposed actions\n");
	if (r.plan.action_count == 0)
		printf("  (none)\n");
	for (size_t i = 0; i < r.plan.action_count; ++i)
	{
		const membrane_action_t	&a = r.plan.actions[i];

		printf("  %zu. %s\n", i + 1, a.summary);
		printf("      Risk: %s | reversible: %s | origin: %s\n",
			membrane_action_risk_name(a.risk), a.reversible ? "yes" : "no",
			membrane_action_origin_name(a.origin));
		if (a.reason[0])
			printf("      Why: %s\n", a.reason);
		if (r.attempted)
			printf("      Result: %s%s%s\n", membrane_action_outcome_name(
					r.result.outcome[i]), r.result.error[i][0] ? " -- " : "",
				r.result.error[i]);
	}
	if (r.plan.unapplied_count > 0)
		printf("\nNot applied (advisory only)\n");
	for (size_t i = 0; i < r.plan.unapplied_count; ++i)
		printf("  [%s] %s\n", r.plan.unapplied[i].reason_code,
			r.plan.unapplied[i].text);
	for (const auto &n : r.notes)
		printf("\nNote: %s\n", n.c_str());
	printf("\nStatus: %s%s%s\n", r.status.c_str(),
		r.detail.empty() ? "" : " -- ", r.detail.c_str());
	if (r.dry_run && r.plan.action_count > 0)
		printf("Nothing was changed. Re-run with --yes to apply.\n");
}

/* ------------------------------------------------------------------ */
/* Executor glue                                                        */
/* ------------------------------------------------------------------ */

typedef struct s_exec_ctx
{
	apply_backend						*backend;
	const membrane_action_plan_t		*plan;
	std::vector<std::string>			*after;
}	exec_ctx_t;

static int	ex_read(void *c, membrane_apply_live_t *out)
{
	std::string	err;

	return (((exec_ctx_t *)c)->backend->read_live(out, &err) ? 1 : 0);
}

static int	ex_exec(void *c, const membrane_action_t *a, char *err, size_t n)
{
	exec_ctx_t	*x = (exec_ctx_t *)c;
	std::string	after;
	std::string	e;
	size_t		idx = (size_t)(a - x->plan->actions);
	bool		ok = x->backend->execute(*a, &after, &e);

	if (idx < x->after->size())
		(*x->after)[idx] = after;
	if (!ok)
		snprintf(err, n, "%s", e.c_str());
	return (ok ? 1 : 0);
}

static int	ex_rollback(void *c, const membrane_action_t *a, char *err,
				size_t n)
{
	std::string	e;
	bool		ok = ((exec_ctx_t *)c)->backend->rollback(*a, &e);

	if (!ok)
		snprintf(err, n, "%s", e.c_str());
	return (ok ? 1 : 0);
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                             */
/* ------------------------------------------------------------------ */

static void	print_err(bool want_json, const std::string &code,
				const std::string &message)
{
	if (want_json)
	{
		json	j;

		j["ok"] = false;
		j["error"] = {{"code", code}, {"message", message}};
		printf("%s\n", j.dump().c_str());
	}
	else
		fprintf(stderr, "membrane apply: %s\n", message.c_str());
}

static std::string	now_utc(void)
{
	char	buf[32];
	int64_t	ms = (int64_t)std::chrono::duration_cast<
			std::chrono::milliseconds>(std::chrono::system_clock::now()
				.time_since_epoch()).count();

	if (!membrane_obs_format_utc(ms, buf, sizeof(buf)))
		return ("");
	return (buf);
}

static bool	default_intelligence(const std::string &runtime,
				membrane_memory_intelligence_t *intel, std::string *err)
{
	std::unique_ptr<membrane_observation_snapshot_t>	snap(
			new membrane_observation_snapshot_t());
	membrane_runtime_capabilities_t					caps;

	return (membrane_advise_collect(runtime, snap.get(), &caps, intel, err));
}

int	membrane_apply_cmd_dispatch_ex(const std::vector<std::string> &args,
		bool want_json, const membrane_apply_io_t &io)
{
	s_apply_opts	o;
	std::string		err;

	if (!parse_apply_opts(args, &o, &err))
	{
		print_err(want_json, "CLI_ERROR", err + " -- usage: membrane apply "
			"[--runtime ID] [--context N|auto] [--kv native|q8|q5|auto] "
			"[--kv-placement default|cpu|gpu|auto] [--gpu-layers all|auto|N] "
			"[--activate M] [--unload M] [--pin M] [--unpin M] [--reload] "
			"[--model M] [--yes] [--json]");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	std::unique_ptr<apply_backend>	backend;

	if (o.runtime == MEMBRANE_RUNTIME_ID_NATIVE)
		backend.reset(new native_backend(&io));
	else if (o.runtime == MEMBRANE_RUNTIME_ID_OLLAMA)
		backend.reset(new ollama_backend());
	else
	{
		print_err(want_json, "CLI_ERROR", "apply for runtime '" + o.runtime
			+ "' is not implemented -- only membrane-native and ollama "
			"are supported");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}

	std::unique_ptr<apply_report_t>	rp(new apply_report_t());
	apply_report_t					&r = *rp;
	std::unique_ptr<membrane_memory_intelligence_t>	intel(
			new membrane_memory_intelligence_t());
	auto	get_intel = io.intelligence ? io.intelligence
		: std::function<bool(const std::string &,
			membrane_memory_intelligence_t *, std::string *)>(
				default_intelligence);
	membrane_apply_surface_t		surface;
	membrane_apply_live_t			live;

	r.runtime = o.runtime;
	memset(&r.plan, 0, sizeof(r.plan));
	memset(&r.result, 0, sizeof(r.result));
	r.confirmation = "not_needed";
	backend->fill_surface(&surface);
	if (!get_intel(o.runtime, intel.get(), &err))
	{
		print_err(want_json, "CLI_ERROR", err);
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!backend->read_live(&live, &err))
	{
		r.status = "refused";
		r.detail = err;
		r.plan.schema_version = MEMBRANE_ACTION_SCHEMA_VERSION;
		if (want_json)
			printf("%s\n", report_json(r).dump().c_str());
		else
			print_human(r);
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	for (size_t i = 0; i < intel->recommendation_count; ++i)
		r.recs.push_back({{"code", intel->recommendations[i].code},
			{"applicability", membrane_intel_applicability_name(
				intel->recommendations[i].applicability)}});

	membrane_action_validation_t	validation;
	bool	wants_set = o.req.context[0] || o.req.kv_precision[0]
		|| o.req.kv_placement[0] || o.req.gpu_layers[0];

	memset(&validation, 0, sizeof(validation));
	if (!live.runtime_reachable)
	{
		r.plan.schema_version = MEMBRANE_ACTION_SCHEMA_VERSION;
		snprintf(r.plan.runtime_id, sizeof(r.plan.runtime_id), "%s",
			o.runtime.c_str());
		r.status = "refused";
		r.detail = "the runtime is not reachable; nothing can be applied";
	}
	else
	{
		if (wants_set)
			backend->validate(live, o, &validation);
		membrane_action_compile(intel.get(), &surface, &live, &o.req,
			&validation, &r.plan);
		r.status = r.plan.action_count > 0 ? "dry_run"
			: "nothing_to_apply";
	}
	if (r.plan.any_requires_reload)
		r.notes.push_back("some changes take effect only when the model is "
			"next loaded; add --reload to unload and reload it now");

	/* Consent (J2). Dry-run is the default; mutation needs --yes or an
	 * interactive yes. */
	bool	interactive = io.interactive_known ? io.interactive
		: membrane_cli_is_interactive();
	bool	confirmed = false;

	if (r.plan.action_count > 0)
	{
		if (o.yes)
		{
			confirmed = true;
			r.confirmation = "yes_flag";
		}
		else if (interactive && !want_json)
		{
			apply_report_t	preview;

			preview.runtime = r.runtime;
			preview.plan = r.plan;
			preview.recs = r.recs;
			preview.notes = r.notes;
			preview.status = "dry_run";
			print_human(preview);
			confirmed = io.confirm ? io.confirm("Apply these changes?")
				: membrane_cli_ask_yes_no("Apply these changes?", false,
					false);
			r.confirmation = confirmed ? "interactive" : "declined";
			if (!confirmed)
			{
				printf("Declined. Nothing was changed.\n");
				return (MEMBRANE_EXIT_SUCCESS);
			}
		}
		else
			r.confirmation = "not_interactive";
	}
	if (!confirmed)
	{
		bool	fail_closed = o.explicit_intent && r.plan.action_count > 0
			&& !interactive;

		if (fail_closed)
			r.notes.push_back("mutation requested without --yes in a "
				"non-interactive session: refused, nothing was changed");
		if (want_json)
			printf("%s\n", report_json(r).dump().c_str());
		else
			print_human(r);
		if (!live.runtime_reachable)
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		return (fail_closed ? MEMBRANE_EXIT_CLI_ERROR : MEMBRANE_EXIT_SUCCESS);
	}

	/* ---- execution: stale-plan guards, then the transaction engine -- */
	r.dry_run = false;
	r.attempted = true;
	r.after.assign(r.plan.action_count, "");
	{
		std::unique_ptr<membrane_memory_intelligence_t>	fresh(
				new membrane_memory_intelligence_t());
		char	missing[MEMBRANE_INTEL_CODE_MAX] = "";

		if (!get_intel(o.runtime, fresh.get(), &err)
			|| !membrane_action_recommendations_still_hold(&r.plan,
				fresh.get(), missing, sizeof(missing)))
		{
			r.result.status = MEMBRANE_APPLY_STALE_PRECONDITION;
			snprintf(r.result.detail, sizeof(r.result.detail), "the "
				"recommendation '%s' no longer holds; nothing was changed "
				"-- rerun `membrane advise` / `membrane apply`",
				missing[0] ? missing : "?");
		}
		else
		{
			exec_ctx_t					ctx = {backend.get(), &r.plan,
				&r.after};
			membrane_action_executor_t	ex = {&ctx, ex_read, ex_exec,
				ex_rollback};

			membrane_action_run(&r.plan, &ex, &r.result);
		}
	}
	r.status = membrane_apply_status_name(r.result.status);
	r.detail = r.result.detail;

	/* Audit (J12): only when something could have changed. */
	bool	touched = r.result.status != MEMBRANE_APPLY_STALE_PRECONDITION
		&& r.result.status != MEMBRANE_APPLY_REFUSED;
	json	rec;
	std::string	audit_path = io.audit_path.empty()
		? membrane_apply_audit_path() : io.audit_path;

	rec["timestamp"] = now_utc();
	rec["runtime"] = r.runtime;
	rec["status"] = r.status;
	rec["success"] = r.result.status == MEMBRANE_APPLY_APPLIED;
	rec["confirmation"] = r.confirmation;
	rec["detail"] = r.detail;
	rec["actions"] = json::array();
	for (size_t i = 0; i < r.plan.action_count; ++i)
	{
		const membrane_action_t	&a = r.plan.actions[i];
		json					reasons = json::array();

		if (a.source_recommendation[0])
			reasons.push_back(a.source_recommendation);
		rec["actions"].push_back({{"action", membrane_action_code_name(a.code)},
			{"model", nullable(a.model)}, {"before", nullable(a.current,
				a.current_known != 0)}, {"requested", a.requested},
			{"after", i < r.after.size() && !r.after[i].empty()
				? json(r.after[i]) : json(nullptr)},
			{"outcome", membrane_action_outcome_name(r.result.outcome[i])},
			{"error", r.result.error[i][0] ? json(r.result.error[i])
				: json(nullptr)},
			{"reason_codes", reasons}});
	}
	if (touched && !write_audit(audit_path, rec))
		r.notes.push_back("the local audit log could not be written ("
			+ audit_path + ")");
	if (want_json)
		printf("%s\n", report_json(r).dump().c_str());
	else
		print_human(r);
	return (r.result.status == MEMBRANE_APPLY_APPLIED
		? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

int	membrane_apply_cmd_dispatch(const std::vector<std::string> &args,
		bool want_json)
{
	return (membrane_apply_cmd_dispatch_ex(args, want_json,
			membrane_apply_io_t()));
}
