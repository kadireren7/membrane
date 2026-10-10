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
      "tags": [
        "tiny",
        "smoke-test"
      ],
      "purpose": "Tiny smoke-test model: verifies MEMBRANE end to end. Weak at following instructions; it may echo, ramble or hallucinate. That is the model, not the runtime.",
      "tested_status": "validated",
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
      "tags": [
        "tiny",
        "smoke-test"
      ],
      "purpose": "Tiny model for quick checks and demos. Limited quality; struggles outside simple English requests. That is the model, not the runtime.",
      "tested_status": "validated",
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
      "tags": [
        "general",
        "multilingual"
      ],
      "purpose": "General chat model of moderate size. Upstream lists multilingual support. Answer quality is the model's.",
      "tested_status": "validated",
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
    },
    {
      "name": "qwen2.5-0.5b-instruct",
      "aliases": [
        "qwen2.5:0.5b",
        "qwen2.5-0.5b"
      ],
      "display_name": "Qwen2.5-0.5B-Instruct",
      "arch": "qwen2",
      "parameter_count": "0.5B",
      "provider": "Qwen",
      "license": "apache-2.0",
      "tags": [
        "tiny",
        "general",
        "multilingual"
      ],
      "purpose": "Smallest Qwen2.5 chat model: fast and light, modest answer quality. Upstream lists multilingual support.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'qwen2' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
          "size_bytes": 491400032,
          "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
        },
        {
          "quant": "Q5_K_M",
          "filename": "qwen2.5-0.5b-instruct-q5_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q5_k_m.gguf",
          "size_bytes": 522186592,
          "sha256": "041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55"
        },
        {
          "quant": "Q8_0",
          "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf",
          "size_bytes": 675710816,
          "sha256": "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e"
        }
      ]
    },
    {
      "name": "qwen2.5-coder-1.5b-instruct",
      "aliases": [
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder-1.5b"
      ],
      "display_name": "Qwen2.5-Coder-1.5B-Instruct",
      "arch": "qwen2",
      "parameter_count": "1.5B",
      "provider": "Qwen",
      "license": "apache-2.0",
      "tags": [
        "coding"
      ],
      "purpose": "Code-focused instruct model: code generation and explanation. MEMBRANE has not measured code quality.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'qwen2' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
          "size_bytes": 1117320768,
          "sha256": "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046"
        },
        {
          "quant": "Q5_K_M",
          "filename": "qwen2.5-coder-1.5b-instruct-q5_k_m.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q5_k_m.gguf",
          "size_bytes": 1285494336,
          "sha256": "f15d02e8e51a5c6f9448b972819acfa66aee4d8bb7d881a8b8ba3d90da08ef09"
        },
        {
          "quant": "Q8_0",
          "filename": "qwen2.5-coder-1.5b-instruct-q8_0.gguf",
          "download_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q8_0.gguf",
          "size_bytes": 1894532160,
          "sha256": "507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47"
        }
      ]
    },
    {
      "name": "llama-3.2-1b-instruct",
      "aliases": [
        "llama3.2:1b",
        "llama-3.2-1b"
      ],
      "display_name": "Llama-3.2-1B-Instruct",
      "arch": "llama",
      "parameter_count": "1B",
      "provider": "bartowski",
      "license": "llama3.2",
      "tags": [
        "general",
        "multilingual",
        "long-context"
      ],
      "purpose": "Small general chat model. Upstream lists several languages and a 128K-token context; MEMBRANE has not validated long-context behavior. Llama 3.2 Community License applies.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'llama' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
          "size_bytes": 807694464,
          "sha256": "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"
        },
        {
          "quant": "Q5_K_M",
          "filename": "Llama-3.2-1B-Instruct-Q5_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
          "size_bytes": 911503488,
          "sha256": "4f22f95fb1679ef8e2fd6f1659d23f06cd1dbedb79ff291db7413a7ae509e2a9"
        },
        {
          "quant": "Q8_0",
          "filename": "Llama-3.2-1B-Instruct-Q8_0.gguf",
          "download_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf",
          "size_bytes": 1321083008,
          "sha256": "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3"
        }
      ]
    },
    {
      "name": "deepseek-r1-distill-qwen-1.5b",
      "aliases": [
        "deepseek-r1:1.5b",
        "r1-distill-1.5b"
      ],
      "display_name": "DeepSeek-R1-Distill-Qwen-1.5B",
      "arch": "qwen2",
      "parameter_count": "1.5B",
      "provider": "bartowski",
      "license": "mit",
      "tags": [
        "reasoning"
      ],
      "purpose": "Reasoning-style distilled model: writes a thinking trace before its answer. License is MIT per the upstream deepseek-ai model.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'qwen2' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
          "size_bytes": 1117320800,
          "sha256": "1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f"
        },
        {
          "quant": "Q5_K_M",
          "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf",
          "size_bytes": 1285494368,
          "sha256": "4d8f6465e96258cf44303eb02812c391d15f6c77f0d72e47757c98fd14994608"
        },
        {
          "quant": "Q8_0",
          "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
          "download_url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
          "size_bytes": 1894532192,
          "sha256": "166baa90e6a963de31a7713ff034f82317ff1fa41602f7c3ce583d3f6b551829"
        }
      ]
    },
    {
      "name": "gemma-2-2b-it",
      "aliases": [
        "gemma2:2b",
        "gemma-2-2b"
      ],
      "display_name": "Gemma-2-2B-it",
      "arch": "gemma2",
      "parameter_count": "2B",
      "provider": "bartowski",
      "license": "gemma",
      "tags": [
        "general"
      ],
      "purpose": "Google's small general instruct model. Gemma Terms of Use apply. MEMBRANE has not validated the gemma2 architecture.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'gemma2' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "gemma-2-2b-it-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
          "size_bytes": 1708582752,
          "sha256": "e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787"
        },
        {
          "quant": "Q5_K_M",
          "filename": "gemma-2-2b-it-Q5_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf",
          "size_bytes": 1923278688,
          "sha256": "be65d5966f7efd40d9d8c9d5f6667582d861de52df20133991e2e9cb491150da"
        },
        {
          "quant": "Q8_0",
          "filename": "gemma-2-2b-it-Q8_0.gguf",
          "download_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q8_0.gguf",
          "size_bytes": 2784495456,
          "sha256": "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe"
        }
      ]
    },
    {
      "name": "phi-3.5-mini-instruct",
      "aliases": [
        "phi3.5:mini",
        "phi-3.5-mini"
      ],
      "display_name": "Phi-3.5-mini-instruct",
      "arch": "phi3",
      "parameter_count": "3.8B",
      "provider": "bartowski",
      "license": "mit",
      "tags": [
        "general",
        "multilingual",
        "long-context"
      ],
      "purpose": "Larger small-model (3.8B): stronger answers, needs more memory. Upstream lists multilingual use and a 128K-token context; MEMBRANE has not validated either.",
      "tested_status": "metadata-only",
      "chat_template_status": "present",
      "compatibility_status": "NOT_YET_VALIDATED",
      "compatibility_evidence": "architecture 'phi3' - see docs/compatibility.json; this exact model has not been validated by MEMBRANE",
      "repo_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF",
      "recorded_at": "2026-10-11",
      "variants": [
        {
          "quant": "Q4_K_M",
          "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
          "download_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
          "size_bytes": 2393232672,
          "sha256": "e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5"
        },
        {
          "quant": "Q8_0",
          "filename": "Phi-3.5-mini-instruct-Q8_0.gguf",
          "download_url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf",
          "size_bytes": 4061222688,
          "sha256": "76fbf02f6fe92af57dbd818409bc8a0240026f1f3609bb405c3be94c973fb823"
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
		f.purpose = jf.value("purpose", std::string());
		f.tested_status = jf.value("tested_status", std::string("metadata-only"));
		if (jf.contains("tags") && jf["tags"].is_array())
			for (const auto &tg : jf["tags"])
				f.tags.push_back(tg.get<std::string>());
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
