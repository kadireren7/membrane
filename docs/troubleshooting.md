# Troubleshooting

Start with `membrane doctor`. It says what is wrong and what to run next.

| You see | Likely cause | What to do |
|---|---|---|
| `membrane` shows **Server Stopped** | nothing is serving | `membrane serve`, or `membrane service start` if the service is installed |
| `membrane chat`: "server not running" | same | same |
| `doctor`: *you are running X, but `membrane` on your PATH is Y* | two installs (e.g. an old `/usr/bin/membrane` and a fresh build) | `which -a membrane`; run the one you intend, or reinstall |
| `doctor`: *a NEWER build exists in this source tree* | you built new code but are running an older binary | run the build-tree binary directly, or `cmake --install` / reinstall the package |
| `doctor`: *shared library 'libcudart.so.12' is missing* | a CUDA-linked build on a machine without the CUDA runtime | install the CUDA runtime, **or use the Vulkan build** (it needs no CUDA), or a CPU build. This is an environment problem, not a MEMBRANE bug |
| GPU `none (this build is CPU-only)` | the build has no GPU backend | install the Vulkan `.deb`, or build with `-DGGML_VULKAN=ON` |
| GPU backend present, no device found | no working GPU driver | Vulkan: install your Vulkan driver (`vulkaninfo` should list a device); CUDA: NVIDIA driver + CUDA runtime |
| `membrane use qwen` lists candidates and exits | a partial name is never chosen for you without a terminal | re-run with the exact id it printed |
| `membrane apply` exits `2` | you asked for a change without `--yes` and there is no terminal | add `--yes`, or run it on a terminal |
| `apply`: `stale_precondition` | the situation changed between planning and applying | nothing was changed; re-run `advise` / `apply` |
| Answers are poor, repetitive, or in the wrong language | the model, usually a tiny one | [model-quality.md](model-quality.md) |
| `model load failed` / out of memory | the model does not fit what is free | `membrane plan MODEL`, `membrane models recommend`, close other programs |
| `membrane run`: *not used -- ... not reachable* for Ollama/vLLM | that runtime is not running | start it, or ignore it: the other runtimes are still considered |

Still stuck: `membrane doctor --json` and `membrane observe --json` contain
everything needed for a bug report (no prompts or secrets are included).
