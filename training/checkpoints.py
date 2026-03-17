from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


COMPILED_MODULE_PREFIX = "_orig_mod."


def extract_model_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping) and "model" in payload and isinstance(payload["model"], Mapping):
        return payload["model"]
    if isinstance(payload, Mapping):
        return payload
    raise TypeError("Checkpoint payload must be a state_dict mapping or contain a 'model' mapping.")


def normalize_model_state_dict_keys(
    state_dict: Mapping[str, Any],
    reference_keys: Iterable[str],
) -> dict[str, Any]:
    normalized = dict(state_dict)
    state_keys = [key for key in normalized.keys() if isinstance(key, str)]
    expected_keys = [key for key in reference_keys if isinstance(key, str)]
    if not state_keys or not expected_keys:
        return normalized
    if set(state_keys) == set(expected_keys):
        return normalized

    state_is_compiled = all(key.startswith(COMPILED_MODULE_PREFIX) for key in state_keys)
    expected_is_compiled = all(key.startswith(COMPILED_MODULE_PREFIX) for key in expected_keys)
    if state_is_compiled == expected_is_compiled:
        return normalized

    if state_is_compiled:
        return {
            key.removeprefix(COMPILED_MODULE_PREFIX): value
            for key, value in normalized.items()
        }

    return {
        f"{COMPILED_MODULE_PREFIX}{key}": value
        for key, value in normalized.items()
    }
