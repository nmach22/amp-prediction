"""Export the selected MIC model artifact as a stable inference bundle."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mic_inference import MIC_INFERENCE_SCHEMA_VERSION

DEFAULT_MODEL_NAME = "mlp_mic_physchem_esm2_context_regularized"
DEFAULT_SOURCE_MODEL = ROOT / "results" / "models" / f"{DEFAULT_MODEL_NAME}_model.joblib"
DEFAULT_METRICS = ROOT / "results" / "tables" / f"{DEFAULT_MODEL_NAME}_metrics.csv"
DEFAULT_OUTPUT = ROOT / "results" / "inference" / "best_mic_model.joblib"

FEATURE_BUILDER_BY_MODEL = {
    "mlp_mic_physchem": "mlp_physchem",
    "mlp_mic_physchem_regularized": "mlp_physchem",
    "mlp_mic_physchem_mild_regularized": "mlp_physchem",
    "mlp_mic_esm2_context_regularized": "mlp_esm2_context",
    "mlp_mic_physchem_esm2_context_regularized": "mlp_physchem_esm2_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained MIC model artifact for inference.",
    )
    parser.add_argument(
        "--source-model",
        default=str(DEFAULT_SOURCE_MODEL),
        help="Training artifact produced by run_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output inference bundle path.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name to store in the inference manifest.",
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS),
        help="Optional metrics CSV to include in the manifest.",
    )
    parser.add_argument(
        "--feature-builder",
        default=None,
        choices=sorted(set(FEATURE_BUILDER_BY_MODEL.values())),
        help="Override the inferred inference feature builder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_model = Path(args.source_model)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics)

    if not source_model.exists():
        raise FileNotFoundError(f"Source model artifact does not exist: {source_model}")

    artifact = joblib.load(source_model)
    model_name = artifact.get("model_name", args.model_name)
    feature_builder = args.feature_builder or FEATURE_BUILDER_BY_MODEL.get(model_name)
    if feature_builder is None:
        raise ValueError(
            f"No default feature builder is known for model {model_name!r}. "
            "Pass --feature-builder explicitly."
        )

    feature_columns = artifact.get("feature_columns")
    if not feature_columns:
        raise ValueError("Source model artifact does not include feature_columns.")

    manifest = build_manifest(
        artifact=artifact,
        model_name=model_name,
        source_model=source_model,
        metrics_path=metrics_path,
        feature_builder=feature_builder,
    )
    bundle = {
        "schema_version": MIC_INFERENCE_SCHEMA_VERSION,
        "model_name": model_name,
        "model": artifact["model"],
        "feature_columns": feature_columns,
        "feature_builder": feature_builder,
        "target": artifact.get("target", "log10_mic"),
        "output_columns": ["pred_log_mic", "pred_mic"],
        "manifest": manifest,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Saved MIC inference bundle: {output_path}")
    print(f"Saved MIC inference manifest: {manifest_path}")


def build_manifest(
    artifact: dict[str, Any],
    model_name: str,
    source_model: Path,
    metrics_path: Path,
    feature_builder: str,
) -> dict[str, Any]:
    """Create a JSON-serializable manifest for the inference bundle."""
    manifest = {
        "schema_version": MIC_INFERENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "source_model": str(source_model),
        "feature_builder": feature_builder,
        "target": artifact.get("target", "log10_mic"),
        "prediction_outputs": {
            "pred_log_mic": "predicted log10(MIC)",
            "pred_mic": "10 ** pred_log_mic",
        },
        "feature_count": len(artifact["feature_columns"]),
        "required_input_columns": ["sequence"],
        "optional_context_columns": [
            "gram_status",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
        ],
        "metrics": read_metrics(metrics_path),
        "metadata": {
            key: value
            for key, value in artifact.items()
            if key
            not in {
                "model",
                "feature_columns",
                "mlp_training_history",
                "numeric_imputation_medians",
            }
            and is_jsonable(value)
        },
    }
    if "embedding_path" in artifact:
        manifest["embedding_cache"] = {
            "path": artifact["embedding_path"],
            "model": artifact.get("embedding_model"),
            "dimension": artifact.get("embedding_dim"),
        }
        manifest["notes"] = [
            "Rows with novel sequences require ESM2 embeddings in the configured cache.",
            "Use scripts/make_plm_embeddings.py on the inference CSV before prediction if the cache is missing sequences.",
        ]
    return manifest


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Read train/validation metrics from a metrics CSV if available."""
    if not path.exists():
        return {}
    metrics = pd.read_csv(path, index_col="split")
    selected_columns = [column for column in ("mae", "rmse", "r2") if column in metrics]
    return {
        split: {
            column: float(metrics.loc[split, column])
            for column in selected_columns
            if pd.notna(metrics.loc[split, column])
        }
        for split in metrics.index
    }


def is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


if __name__ == "__main__":
    main()
