#!/usr/bin/env python3
"""Milestone G3, Part 5 -- sibling byte-ratio scaling error.

G2's scale_hparams() (tools/membrane/plan_cmd.cpp) predicts an
uninstalled sibling variant's bytes_per_layer/output_role_bytes/
total_weight_bytes by rescaling a real installed sibling's REAL GGUF-
read values by the ratio of two catalog-recorded, verified download
sizes (target_size_bytes / base_size_bytes):

    predicted_field = real_base_field * (catalog_target_bytes / catalog_base_bytes)
    predicted_total_weight_bytes = catalog_target_bytes   (verbatim, not scaled)

This script measures how far that PREDICTION is from the REAL value,
by reading actual tensor bytes out of two real, locally present GGUF
files of the same family (never downloading extra files itself --
scripts/download is a separate, one-time, disclosed step; see
docs/planner-v2-evidence.md).

The tensor categorization below is an intentional line-for-line port
of membrane_gpu_estimate_model() (tools/membrane-run/gpu_device.cpp):
  - total_bytes: sum of every tensor's byte size.
  - bytes_per_layer: sum of "blk.N.*" tensor bytes / number of distinct
    layer indices seen.
  - output_role_bytes: "output.weight" size if present, else
    "token_embd.weight" size (untied case), plus "output_norm.weight"
    size.

This is a measurement/observation script -- it does not change the
planner or the scaling formula (see Part 5 of the G3 task: "Do not
silently adjust the formula before recording baseline evidence.").

Usage:
  PYTHONPATH=third_party/llama.cpp/gguf-py \
    scripts/planner-v2-evidence/measure_sibling_scaling_error.py [OUTPUT_JSON]
"""
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402

BLK_RE = re.compile(r"^blk\.(\d+)\.")


def read_real_hparams(gguf_path: Path) -> dict:
	reader = gguf.GGUFReader(str(gguf_path))
	total_bytes = 0
	layer_bytes = 0
	layer_indices = set()
	token_embd_bytes = 0
	output_bytes = 0
	output_norm_bytes = 0
	have_output = False
	for t in reader.tensors:
		sz = int(t.n_bytes)
		total_bytes += sz
		m = BLK_RE.match(t.name)
		if m:
			layer_bytes += sz
			layer_indices.add(int(m.group(1)))
		elif t.name == "token_embd.weight":
			token_embd_bytes = sz
		elif t.name == "output.weight":
			output_bytes = sz
			have_output = True
		elif t.name == "output_norm.weight":
			output_norm_bytes = sz
	if not layer_indices:
		raise RuntimeError(f"{gguf_path}: no blk.N.* tensors found")
	n_layer = len(layer_indices)
	bytes_per_layer = layer_bytes // n_layer
	output_role_bytes = (output_bytes if have_output else token_embd_bytes) + output_norm_bytes
	return {
		"path": str(gguf_path),
		"file_size_bytes": gguf_path.stat().st_size,
		"total_weight_bytes": total_bytes,
		"n_layer": n_layer,
		"bytes_per_layer": bytes_per_layer,
		"output_role_bytes": output_role_bytes,
		"have_output_tensor": have_output,
	}


def predict_from_base(base: dict, base_catalog_bytes: int, target_catalog_bytes: int) -> dict:
	"""Exact port of plan_cmd.cpp's scale_hparams()."""
	ratio = target_catalog_bytes / base_catalog_bytes
	return {
		"bytes_per_layer": base["bytes_per_layer"] * ratio,
		"output_role_bytes": base["output_role_bytes"] * ratio,
		"total_weight_bytes": target_catalog_bytes,  # verbatim, per scale_hparams()
	}


def pct_error(predicted: float, actual: float) -> float:
	if actual == 0:
		return float("nan")
	return (predicted - actual) / actual * 100.0


# Real, catalog-recorded (model_catalog.cpp) verified download sizes --
# NOT re-derived here; copied verbatim so this script's ratio math is
# provably the same ratio membrane's own catalog uses.
CATALOG_SIZE_BYTES = {
	"F16": 270885952,
	"Q8_0": 144811072,
	"Q5_K_M": 112103488,
	"Q4_K_M": 105454144,
}

FILES = {
	"F16": REPO_ROOT / "models/SmolLM2-135M-Instruct-f16.gguf",
	"Q8_0": REPO_ROOT / "models/SmolLM2-135M-Instruct-Q8_0.gguf",
	"Q4_K_M": REPO_ROOT / "models/SmolLM2-135M-Instruct-Q4_K_M.gguf",
}


def main():
	out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
		REPO_ROOT / "scratch/sibling_scaling_error.json"
	)
	real = {}
	for quant, path in FILES.items():
		if not path.exists():
			print(f"SKIP {quant}: {path} not present", file=sys.stderr)
			continue
		real[quant] = read_real_hparams(path)

	pairs = []
	base_quant = "F16"  # the only variant the project's real registry has installed
	if base_quant in real:
		base = real[base_quant]
		for target_quant in ("Q8_0", "Q4_K_M"):
			if target_quant not in real:
				continue
			actual = real[target_quant]
			predicted = predict_from_base(
				base, CATALOG_SIZE_BYTES[base_quant], CATALOG_SIZE_BYTES[target_quant]
			)
			pairs.append({
				"base_quant": base_quant,
				"target_quant": target_quant,
				"predicted": predicted,
				"actual": {
					"bytes_per_layer": actual["bytes_per_layer"],
					"output_role_bytes": actual["output_role_bytes"],
					"total_weight_bytes": actual["total_weight_bytes"],
				},
				"error_pct": {
					"bytes_per_layer": pct_error(predicted["bytes_per_layer"], actual["bytes_per_layer"]),
					"output_role_bytes": pct_error(predicted["output_role_bytes"], actual["output_role_bytes"]),
					"total_weight_bytes": pct_error(predicted["total_weight_bytes"], actual["total_weight_bytes"]),
				},
			})

	result = {
		"evidence_level": "REAL",
		"description": "Sibling byte-ratio scaling error, measured against real installed GGUFs (Milestone G3, Part 5).",
		"model_family": "smollm2-135m-instruct",
		"real_measurements": real,
		"scaling_error_pairs": pairs,
	}
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2))
	print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
	main()
