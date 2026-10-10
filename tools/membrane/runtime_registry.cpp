#include "runtime_adapter.h"
#include "runtime_ollama.h"
#include "runtime_vllm.h"

/* See runtime_adapter.h's own top comment for the full contract. */

void	membrane_external_model_init(membrane_external_model_t *m)
{
	*m = membrane_external_model_t();
	m->size_known = false;
	m->size_bytes = 0;
	m->context_length_known = false;
	m->context_length = 0;
	m->remote = false;
	m->provenance = MEMBRANE_CAPABILITY_PROVENANCE_UNKNOWN;
}

void	membrane_external_model_detail_init(
			membrane_external_model_detail_t *d)
{
	*d = membrane_external_model_detail_t();
	membrane_external_model_init(&d->model);
	d->parameter_count_known = false;
	d->parameter_count = 0;
	d->parameters_truncated = false;
	d->has_template = false;
	d->has_system = false;
	d->has_license = false;
}

/* membrane-native is described by H1's own pure, static module --
 * never re-implemented here. */
static void	native_describe(membrane_runtime_descriptor_t *out)
{
	membrane_runtime_describe(MEMBRANE_RUNTIME_ID_NATIVE, out);
}

/* Milestone L: loaded-model lists, per adapter (see runtime_adapter.h). */
static bool	ollama_resident(std::vector<std::string> *out,
				membrane_runtime_error_t *err)
{
	std::vector<membrane_ollama_process_model_t>	ps;

	out->clear();
	if (!membrane_ollama_list_running(&ps, err))
		return (false);
	for (const auto &m : ps)
		out->push_back(m.model.empty() ? m.name : m.model);
	return (true);
}

/* vLLM serves exactly the model(s) it was started with: all are loaded. */
static bool	vllm_resident(std::vector<std::string> *out,
				membrane_runtime_error_t *err)
{
	std::vector<membrane_external_model_t>	models;

	out->clear();
	if (!membrane_vllm_list_models(&models, err))
		return (false);
	for (const auto &m : models)
		out->push_back(m.runtime_model_id);
	return (true);
}

static const membrane_runtime_adapter_t	g_adapters[] = {
	{MEMBRANE_RUNTIME_ID_NATIVE, native_describe, NULL, NULL, NULL},
	{MEMBRANE_RUNTIME_ID_OLLAMA, membrane_ollama_describe,
		membrane_ollama_list_models, membrane_ollama_inspect_model,
		ollama_resident},
	{MEMBRANE_RUNTIME_ID_VLLM, membrane_vllm_describe,
		membrane_vllm_list_models, membrane_vllm_inspect_model,
		vllm_resident},
};

size_t	membrane_runtime_registry_count(void)
{
	return (sizeof(g_adapters) / sizeof(g_adapters[0]));
}

const membrane_runtime_adapter_t	*membrane_runtime_registry_at(size_t i)
{
	if (i >= membrane_runtime_registry_count())
		return (NULL);
	return (&g_adapters[i]);
}

const membrane_runtime_adapter_t	*membrane_runtime_registry_find(
										const std::string &id)
{
	size_t	i;

	i = 0;
	while (i < membrane_runtime_registry_count())
	{
		if (id == g_adapters[i].id)
			return (&g_adapters[i]);
		i++;
	}
	return (NULL);
}
