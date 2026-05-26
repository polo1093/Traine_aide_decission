"""Offline prediction helper for the bootstrap model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_bootstrap_model import CATEGORICAL_FEATURES, CARD_FEATURES, FEATURE_ALIASES, NUMERIC_FEATURES, feature_payload


REQUIRED_NUMERIC_FEATURES = ("features.pot", "features.to_call")


class BootstrapPredictionError(ValueError):
    """Raised when offline prediction input is invalid."""


def predict_bootstrap_model(
    *,
    model_dir: str | Path,
    input_payload: Mapping[str, Any],
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    try:
        model, feature_schema, label_mapping = load_prediction_artifacts(model_dir)
        validate_input(input_payload, feature_schema)
        warnings = prediction_warnings(input_payload, feature_schema)
        feature_order = feature_schema.get("feature_order") or feature_schema.get("numeric_features", []) + feature_schema.get("categorical_features", []) + feature_schema.get("card_features", [])
        features = feature_payload(input_payload, feature_columns=feature_order)
        prediction = str(model.predict([features])[0])
        labels = set(label_mapping.get("labels", []))
        if prediction not in labels:
            raise BootstrapPredictionError(f"unknown_predicted_class:{prediction}")
        result = {
            "status": "ok",
            "prediction": prediction,
            "probabilities": predict_probabilities(model, features),
            "warnings": ["pipeline_smoke_only_not_for_production", *warnings],
        }
    except BootstrapPredictionError as exc:
        result = {"status": "failed", "error": str(exc), "warnings": ["pipeline_smoke_only_not_for_production"]}

    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def load_prediction_artifacts(model_dir: str | Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    directory = Path(model_dir)
    model_path = directory / "model.joblib"
    feature_schema_path = directory / "feature_schema.json"
    label_mapping_path = directory / "label_mapping.json"
    if not model_path.exists():
        raise BootstrapPredictionError("model_missing")
    if not feature_schema_path.exists():
        raise BootstrapPredictionError("feature_schema_missing")
    if not label_mapping_path.exists():
        raise BootstrapPredictionError("label_mapping_missing")
    model = joblib.load(model_path)
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    label_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    return model, feature_schema, label_mapping


def validate_input(payload: Mapping[str, Any], feature_schema: Mapping[str, Any]) -> None:
    required = list(REQUIRED_NUMERIC_FEATURES) + list(feature_schema.get("categorical_features", CATEGORICAL_FEATURES))
    missing = [field for field in required if not _has_feature(payload, field)]
    if missing:
        raise BootstrapPredictionError(f"missing_field:{_display_field(missing[0])}")


def prediction_warnings(payload: Mapping[str, Any], feature_schema: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    known_values = feature_schema.get("categorical_values", {})
    for field in feature_schema.get("categorical_features", CATEGORICAL_FEATURES):
        value = str(_payload_value(payload, field) or "UNKNOWN")
        allowed = set(known_values.get(field, []))
        if allowed and value not in allowed:
            warnings.append(f"unknown_category:{field}:{value}")
    for field in feature_schema.get("numeric_features", NUMERIC_FEATURES):
        if not _has_feature(payload, field) and field not in REQUIRED_NUMERIC_FEATURES:
            warnings.append(f"optional_numeric_field_missing:{field}")
    for field in feature_schema.get("card_features", CARD_FEATURES):
        if not _has_feature(payload, field):
            warnings.append(f"optional_card_field_missing:{field}")
    return warnings


def _has_feature(payload: Mapping[str, Any], field: str) -> bool:
    return field in payload or any(alias in payload for alias in FEATURE_ALIASES.get(field, ()))


def _payload_value(payload: Mapping[str, Any], field: str) -> Any:
    if field in payload:
        return payload.get(field)
    for alias in FEATURE_ALIASES.get(field, ()):
        if alias in payload:
            return payload.get(alias)
    return None


def _display_field(field: str) -> str:
    return FEATURE_ALIASES.get(field, (field,))[0]


def predict_probabilities(model: Any, features: Mapping[str, Any]) -> dict[str, float] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba([features])[0]
    classes = [str(value) for value in model.classes_]
    return {label: round(float(probability), 6) for label, probability in zip(classes, probabilities)}


def load_input(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        value = json.loads(args.input_json)
    elif args.input_file:
        value = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    else:
        raise BootstrapPredictionError("input_missing")
    if not isinstance(value, dict):
        raise BootstrapPredictionError("input_must_be_json_object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-json")
    parser.add_argument("--input-file")
    parser.add_argument("--output-json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = load_input(args)
        result = predict_bootstrap_model(model_dir=args.model_dir, input_payload=payload, output_json=args.output_json)
    except (BootstrapPredictionError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "error": str(exc), "warnings": ["pipeline_smoke_only_not_for_production"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
