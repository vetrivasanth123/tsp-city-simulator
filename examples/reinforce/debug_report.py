
# ============================================================
# DEBUG CODE STARTED
# File: examples/reinforce/debug_report.py
# Purpose: Collect compact, auditable REINFORCE debug details.
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def debug_to_jsonable(value: Any) -> Any:
    """Convert tensors, NumPy values, and non-finite floats to JSON-safe data."""
    if isinstance(value, torch.Tensor):
        return debug_to_jsonable(value.detach().cpu().tolist())

    if isinstance(value, np.ndarray):
        return debug_to_jsonable(value.tolist())

    if isinstance(value, np.generic):
        return debug_to_jsonable(value.item())

    if isinstance(value, dict):
        return {
            str(key): debug_to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [debug_to_jsonable(item) for item in value]

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if np.isnan(value):
            return "NaN"
        if np.isposinf(value):
            return "Infinity"
        if np.isneginf(value):
            return "-Infinity"

        return value

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def debug_tensor_details(tensor: torch.Tensor) -> dict:
    """
    Capture actual tensor values and essential metadata.

    Deliberately omits min/max/mean/std statistics to reduce
    JSON size. The original tensor values are preserved.
    """
    tensor_cpu = tensor.detach().cpu()

    return {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "numel": int(tensor_cpu.numel()),
        "values": debug_to_jsonable(tensor_cpu),
    }


def debug_module_snapshot(module: nn.Module) -> dict:
    """
    Capture parameters, their current gradients, and buffers.

    This is intended for initial/final network snapshots.
    Per-update recording is handled by reinforce.py.
    """
    parameters = {}
    buffers = {}

    for name, parameter in module.named_parameters():
        parameters[name] = {
            **debug_tensor_details(parameter),
            "requires_grad": bool(parameter.requires_grad),
            "gradient": (
                debug_tensor_details(parameter.grad)
                if parameter.grad is not None
                else None
            ),
        }

    for name, buffer in module.named_buffers():
        buffers[name] = debug_tensor_details(buffer)

    return {
        "parameters": parameters,
        "buffers": buffers,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        "total_parameter_count": sum(
            parameter.numel()
            for parameter in module.parameters()
        ),
    }


class ReinforceDebugReport:
    """Temporary recorder for one linear or MLP REINFORCE run."""

    def __init__(
        self,
        policy_type: str,
        output_path: str | Path,
    ):
        if policy_type not in {"linear", "mlp"}:
            raise ValueError(
                "policy_type must be 'linear' or 'mlp'."
            )

        self.policy_type = policy_type
        self.output_path = Path(output_path)

        self.data = {
            "debug_report_version": 2,
            "policy_type": policy_type,
            "experiment": {},
            "network_initialization": {},
            "training_episodes": [],
            "training_returns": [],
            "training_episode_updates": [],
            "training_updates": [],
            "policy_events": [],
            "evaluation": {},
            "visualization_rollout": {},
            "final_network": {},
            "diagnostic_forward_pass": {},
        }

    def record(self, section: str, details: Any) -> None:
        """Store diagnostics without printing large values to the console."""
        if section not in self.data:
            raise KeyError(
                f"Unknown debug-report section: {section}"
            )

        if isinstance(self.data[section], list):
            self.data[section].append(debug_to_jsonable(details))
        else:
            self.data[section] = debug_to_jsonable(details)

    def snapshot_network(self, policy: nn.Module) -> dict:
        """Capture the complete current network state."""
        return debug_module_snapshot(policy)

    def save(self) -> Path:
        """Save the separate JSON report."""
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                debug_to_jsonable(self.data),
                file,
                indent=2,
                allow_nan=False,
            )

        return self.output_path


# ============================================================
# DEBUG CODE ENDED
# ============================================================
