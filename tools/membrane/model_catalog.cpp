#include "model_catalog.h"

#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

/*
 * Mega Phase D, PR D1. Every entry below was verified individually,
 * this phase, against the real upstream repository:
 *   - repo existence + license tag: `curl -s
 *     https://huggingface.co/api/models/<repo>`
 *   - exact filenames: the same API call's own `siblings[].rfilename`
 *   - real file size: `curl -sIL <resolve-URL>` (a real HEAD request,
 *     following the real redirect Hugging Face's CDN issues -- the
 *     un-followed response's own Content-Length is a small redirect
 *     stub, NOT the real file size; this catalog's own download
 *     manager must follow redirects for the same reason)
 * `recorded_at` is the real date this verification happened. Three
 * families only this PR (Section 23's own broader Mistral/Phi/Gemma
 * expansion is real, separate work -- tracked as a known limitation
 * below, not invented here without the same evidence bar).
 */
static const char *MEMBRANE_CATALOG_JSON = R"JSON(
{
  "schema_version": 1,
  "families": [
    {
      "name": "smollm2-135m-instruct",
      "aliases": [
        "smollm2:135m",
        "smollm2-135m"
      ],
      "display_name": "SmolLM2-135M-Instruct",
      "arch": "llama",
      "parameter_count": "135M",
      "provider": "unsloth",
      "license": "apache-2.0",
      "chat_template_status": "present",
      "compatibility_status": "SUPPORTED",
      "compatibility_evidence": "docs/compatibility.json MC-01..MC-05, MC-08..MC-12",
      "repo_url": "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF",
      "recorded_at": "2026-09-05",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf",
          "size_bytes": 105454144,
          "sha256": "ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747"
        },
        {
          "quant": "Q5_K_M",
          "filename": "SmolLM2-135M-Instruct-Q5_K_M.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q5_K_M.gguf",
          "size_bytes": 112103488,
          "sha256": "4ffe3cb59ea03238cea3f4a4b03fe516bcd827b6bd02c868a64cf31b2f94dab9"
        },
        {
          "quant": "Q8_0",
          "filename": "SmolLM2-135M-Instruct-Q8_0.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
          "size_bytes": 144811072,
          "sha256": "c4a3dd037301b6ecea31d6da37f5cd793ead920dd5ddfe6d589294628d6ce66a"
        },
        {
          "quant": "F16",
          "filename": "SmolLM2-135M-Instruct-F16.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-F16.gguf",
          "size_bytes": 270885952,
          "sha256": "5157ca60744d21631818364854ac8e4452e1b8022d2ab4c8a2f9cda2344afb30"
        }
      ]
    },
    {
      "name": "smollm2-360m-instruct",
      "aliases": [
        "smollm2:360m",
        "smollm2-360m"
      ],
      "display_name": "SmolLM2-360M-Instruct",
      "arch": "llama",
      "parameter_count": "360M",
      "provider": "unsloth",
      "license": "apache-2.0",
      "chat_template_status": "present",
      "compatibility_status": "SUPPORTED",
      "compatibility_evidence": "docs/compatibility.json MC-06, MC-07",
      "repo_url": "https://huggingface.co/unsloth/SmolLM2-360M-Instruct-GGUF",
      "recorded_at": "2026-09-05",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
          "size_bytes": 270590560,
          "sha256": "16c7f1667fea34bacad196a57b548effcb37614db4ab5677a20c8c7b823b9e63"
        },
        {
          "quant": "F16",
          "filename": "SmolLM2-360M-Instruct-F16.gguf",
          "download_url": "https://huggingface.co/unsloth/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-F16.gguf",
          "size_bytes": 725553760,
          "sha256": "a41984d8bebc26ee81daf68d15077ad86aac2894ee5e9d3453b6895060df213e"
        }
      ]
    },
    {
      "name": "qwen2.5-1.5b-instruct",
      "aliases": [
        "qwen2.5:1.5b",
        "qwen2.5-1.5b"
      ],
      "display_name": "Qwen2.5-1.5B-Instruct",
      "arch": "qwen2",
      "parameter_count": "1.5B",
      "provider": "Qwen",
      "license": "apache-2.0",
      "chat_template_status": "present",
      "compatibility_status": "SUPPORTED",
      "compatibility_evidence": "docs/compatibility.json MC-13..MC-19",
      "repo_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
      "recorded_at": "2026-09-05",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
          "size_bytes": 1117320736,
          "sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
        },
        {
          "quant": "Q5_K_M",
          "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q5_k_m.gguf",
          "size_bytes": 1285494304,
          "sha256": "b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c"
        },
        {
          "quant": "Q8_0",
          "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf",
          "size_bytes": 1894532128,
          "sha256": "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8"
        },
        {
          "quant": "FP16",
          "filename": "qwen2.5-1.5b-instruct-fp16.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-fp16.gguf",
          "size_bytes": 3560416288,
          "sha256": "fc89e330deb3fd8fa560f1c0f35a1e2b8da96d59e13445559ed190307a6f5649"
        }
      ]
    }
  ]
}
)JSON";

static std::string	to_lower(const std::string &s)
{
	std::string	out = s;

	std::transform(out.begin(), out.end(), out.begin(),
		[](unsigned char c) { return ((char)std::tolower(c)); });
	return (out);
}

membrane_catalog_t	membrane_catalog_load(void)
{
	json				root = json::parse(MEMBRANE_CATALOG_JSON);
	membrane_catalog_t	cat;

	cat.schema_version = root.value("schema_version", 0);
	for (const auto &jf : root["families"])
	{
		membrane_catalog_family_t	f;

		f.name = jf.value("name", std::string());
		for (const auto &a : jf["aliases"])
			f.aliases.push_back(a.get<std::string>());
		f.display_name = jf.value("display_name", std::string());
		f.arch = jf.value("arch", std::string());
		f.parameter_count = jf.value("parameter_count", std::string());
		f.provider = jf.value("provider", std::string());
		f.repo_url = jf.value("repo_url", std::string());
		f.license = jf.value("license", std::string());
		f.chat_template_status = jf.value("chat_template_status", std::string());
		f.compatibility_status = jf.value("compatibility_status", std::string());
		f.compatibility_evidence = jf.value("compatibility_evidence", std::string());
		f.recorded_at = jf.value("recorded_at", std::string());
		for (const auto &jv : jf["variants"])
		{
			membrane_catalog_variant_t	v;

			v.quant = jv.value("quant", std::string());
			v.filename = jv.value("filename", std::string());
			v.download_url = jv.value("download_url", std::string());
			v.size_bytes = jv.value("size_bytes", (uint64_t)0);
			v.sha256 = jv.value("sha256", std::string());
			f.variants.push_back(v);
		}
		cat.families.push_back(f);
	}
	return (cat);
}

static bool	contains_ci(const std::string &haystack, const std::string &needle)
{
	return (to_lower(haystack).find(to_lower(needle)) != std::string::npos);
}

std::vector<const membrane_catalog_family_t *>	membrane_catalog_search(
			const membrane_catalog_t &cat, const std::string &query)
{
	std::vector<const membrane_catalog_family_t *>	out;

	for (const auto &f : cat.families)
	{
		if (query.empty() || contains_ci(f.name, query)
			|| contains_ci(f.display_name, query) || contains_ci(f.arch, query)
			|| contains_ci(f.provider, query))
		{
			out.push_back(&f);
			continue ;
		}
		for (const auto &a : f.aliases)
		{
			if (contains_ci(a, query))
			{
				out.push_back(&f);
				break ;
			}
		}
	}
	return (out);
}

const membrane_catalog_family_t	*membrane_catalog_resolve(
			const membrane_catalog_t &cat, const std::string &name_or_alias)
{
	std::string	target = to_lower(name_or_alias);

	for (const auto &f : cat.families)
	{
		if (to_lower(f.name) == target)
			return (&f);
		for (const auto &a : f.aliases)
			if (to_lower(a) == target)
				return (&f);
	}
	return (NULL);
}

const membrane_catalog_variant_t	*membrane_catalog_find_variant(
			const membrane_catalog_family_t &family, const std::string &quant)
{
	std::string	target = to_lower(quant);

	for (const auto &v : family.variants)
		if (to_lower(v.quant) == target)
			return (&v);
	return (NULL);
}
