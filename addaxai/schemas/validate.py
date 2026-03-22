"""JSON schema validation for AddaxAI configuration and output files.

Manual validation without jsonschema dependency — checks types, unknown keys,
and required fields against the defined schema files.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

_SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_schema(name: str) -> Dict[str, Any]:
    """Load a schema JSON file from the schemas directory."""
    path = os.path.join(_SCHEMA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_global_vars(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate global_vars.json data against schema.

    Args:
        data: The loaded global_vars dictionary to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    schema = _load_schema("global_vars.schema.json")
    errors = []
    props = schema.get("properties", {})
    required_keys = schema.get("required", [])

    # Check for missing required keys
    for key in required_keys:
        if key not in data:
            errors.append(f"Required key missing: '{key}'")

    # Check for unknown keys
    for key in data:
        if key not in props:
            errors.append(f"Unknown key: '{key}'")

    # Check types of known keys
    for key, prop_def in props.items():
        if key in data:
            expected_type = prop_def.get("type")
            value = data[key]
            _check_type(key, value, expected_type, errors, prop_def)

    return (len(errors) == 0, errors)


def validate_model_vars(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate model variables.json data against schema.

    Args:
        data: The loaded model variables dictionary to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    schema = _load_schema("model_vars.schema.json")
    errors = []
    props = schema.get("properties", {})

    # Check for unknown keys
    for key in data:
        if key not in props:
            errors.append(f"Unknown key: '{key}'")

    # Check types of known keys
    for key, prop_def in props.items():
        if key in data:
            expected_type = prop_def.get("type")
            value = data[key]
            _check_type(key, value, expected_type, errors, prop_def)

    return (len(errors) == 0, errors)


def validate_recognition_output(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate recognition output JSON against schema.

    Args:
        data: The recognition output dictionary to validate.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    schema = _load_schema("recognition_output.schema.json")
    errors = []
    required_keys = schema.get("required", [])

    # Check for required keys
    for key in required_keys:
        if key not in data:
            errors.append(f"Required key missing: '{key}'")

    # Check type of 'images' field
    if "images" in data:
        images = data["images"]
        if not isinstance(images, list):
            errors.append(f"'images' must be an array, got {type(images).__name__}")
        else:
            for idx, img in enumerate(images):
                if not isinstance(img, dict):
                    errors.append(f"images[{idx}]: expected object, got {type(img).__name__}")
                else:
                    if "file" not in img:
                        errors.append(f"images[{idx}]: required key 'file' missing")
                    if "detections" not in img:
                        errors.append(f"images[{idx}]: required key 'detections' missing")
                    if "detections" in img and not isinstance(img["detections"], list):
                        errors.append(
                            f"images[{idx}]['detections']: expected array, "
                            f"got {type(img['detections']).__name__}"
                        )

    # Check type of 'detection_categories' field
    if "detection_categories" in data:
        categories = data["detection_categories"]
        if not isinstance(categories, dict):
            errors.append(
                f"'detection_categories' must be an object, got {type(categories).__name__}"
            )
        else:
            for key, value in categories.items():
                if not isinstance(value, str):
                    errors.append(
                        f"detection_categories['{key}']: expected string, "
                        f"got {type(value).__name__}"
                    )

    return (len(errors) == 0, errors)


def _check_type(
    key: str,
    value: Any,
    expected_type: str,
    errors: List[str],
    prop_def: Dict[str, Any],
) -> None:
    """Check that a value matches the expected type and constraints.

    Args:
        key: Property key name (for error messages).
        value: The value to check.
        expected_type: Expected type string ('string', 'number', 'integer', 'boolean', 'array', 'object').
        errors: List to append error messages to.
        prop_def: Full property definition (for additional constraints like minimum/maximum).
    """
    if expected_type == "string":
        if not isinstance(value, str):
            errors.append(f"'{key}': expected string, got {type(value).__name__}")
        else:
            enum = prop_def.get("enum")
            if enum is not None and value not in enum:
                errors.append(f"'{key}': value '{value}' not in allowed values {enum}")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"'{key}': expected number, got {type(value).__name__}")
        else:
            _check_numeric_constraints(key, value, prop_def, errors)
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"'{key}': expected integer, got {type(value).__name__}")
        else:
            _check_numeric_constraints(key, value, prop_def, errors)
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            errors.append(f"'{key}': expected boolean, got {type(value).__name__}")
    elif expected_type == "array":
        if not isinstance(value, list):
            errors.append(f"'{key}': expected array, got {type(value).__name__}")
        else:
            items_schema = prop_def.get("items", {})
            item_type = items_schema.get("type") if isinstance(items_schema, dict) else None
            if item_type:
                for idx, item in enumerate(value):
                    _check_type(f"{key}[{idx}]", item, item_type, errors, items_schema)
    elif expected_type == "object":
        if not isinstance(value, dict):
            errors.append(f"'{key}': expected object, got {type(value).__name__}")


def _check_numeric_constraints(
    key: str,
    value: float,
    prop_def: Dict[str, Any],
    errors: List[str],
) -> None:
    """Check numeric constraints (minimum, maximum, enum)."""
    minimum = prop_def.get("minimum")
    maximum = prop_def.get("maximum")
    enum = prop_def.get("enum")

    if enum is not None and value not in enum:
        errors.append(f"'{key}': value {value} not in allowed values {enum}")
    if minimum is not None and value < minimum:
        errors.append(f"'{key}': value {value} is below minimum {minimum}")
    if maximum is not None and value > maximum:
        errors.append(f"'{key}': value {value} is above maximum {maximum}")
