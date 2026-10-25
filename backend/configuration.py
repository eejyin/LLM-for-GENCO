"""Editable GraphKit fields, mapped to the installed source schema."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
import yaml

PAPER = "https://arxiv.org/pdf/2608.09921#page=7"
MODEL_SOURCE = "gridfm_graphkit/models/gnn_heterogeneous_gns.py"
LOSS_SOURCE = "gridfm_graphkit/training/loss.py"

# Public key, YAML path, UI group, bounds, integer, explanation.
SPEC = [
    ("hidden_size", ["model", "hidden_size"], "projection", 1, 512, True, "H is the node and branch projection width. After attention, each node has width H × heads. Increasing H raises parameter count and memory use."),
    ("num_layers", ["model", "num_layers"], "correction", 1, 48, True, "N is the number of heterogeneous message-passing and physics-feedback refinements. Increasing N costs training and inference time. Checkpoint monitoring follows the last layer."),
    ("attention_head", ["model", "attention_head"], "correction", 1, 32, True, "Number of parallel attention heads; affects bus-to-bus and generator-to-bus aggregation and decoder input width."),
    ("dropout", ["model", "dropout"], "correction", 0, .9, False, "Attention dropout in TransformerConv. Regularizes training and is disabled for inference."),
    ("epochs", ["training", "epochs"], "optimizer", 1, 10000, True, "Maximum complete passes through the training set. Each batch backpropagates; validation guides early stopping and checkpoint selection."),
    ("batch_size", ["training", "batch_size"], "optimizer", 1, 4096, True, "Graph samples per gradient update; affects gradient noise, memory, and updates per epoch."),
    ("learning_rate", ["optimizer", "learning_rate"], "optimizer", 1e-8, 1, False, "Optimizer step size for learned weights, not the physics residual weight in the refinement loop."),
    ("seed", ["seed"], "data", 0, 2147483647, True, "Seed for initialization and data processing, useful for reproducible comparisons."),
    ("workers", ["data", "workers"], "data", 0, 32, True, "DataLoader worker count. Local debugging commonly uses 0."),
    ("val_ratio", ["data", "val_ratio"], "data", .01, .8, False, "Validation split proportion, used for tuning and checkpoint selection, not gradient updates."),
    ("test_ratio", ["data", "test_ratio"], "data", .01, .8, False, "Independent test split reserved for final evaluation."),
    ("patience", ["callbacks", "patience"], "optimizer", 1, 1000, True, "Epochs to wait without validation improvement before early stopping."),
]

def get(data, path):
    for part in path:
        data = data[part]
    return data

def put(data, path, value):
    for part in path[:-1]:
        data = data[part]
    data[path[-1]] = value

def descriptors(config):
    result = []
    genco = config.get("model", {}).get("type") == "GNS_heterogeneous"
    for key, path, group, low, high, integer, help_text in SPEC:
        if group in ("projection", "correction") and not genco:
            continue
        try:
            value = get(config, path)
        except (KeyError, IndexError):
            if key == "dropout" and genco:
                value = 0.0
            else:
                continue
        result.append(dict(key=key, path=path, yaml_path=".".join(map(str, path)), group=group,
                           min=low, max=high, integer=integer, value=value, help=help_text))
    training = config["training"]
    for index, name in enumerate(training.get("losses", [])):
        if index < len(training.get("loss_weights", [])):
            path = ["training", "loss_weights", index]
            result.append(dict(key=f"loss_weight_{index}", path=path, yaml_path=f"training.loss_weights[{index}]",
                               group="loss", min=0, max=100, integer=False, value=get(config, path),
                               help=f"{name} mixing coefficient, written in the current loss order; other losses are not renormalized."))
        if name == "LayeredWeightedPhysics" and "base_weight" in training["loss_args"][index]:
            path = ["training", "loss_args", index, "base_weight"]
            result.append(dict(key="base_weight", path=path, yaml_path=f"training.loss_args[{index}].base_weight",
                               group="loss", min=0, max=1, integer=False, value=get(config, path),
                               help="Stage i has a weight proportional to λ^(N−i), normalized across stages. Smaller λ emphasizes later refinements; this does not set the total physics-loss coefficient."))
    return result

def apply_values(original, changes):
    config = copy.deepcopy(original)
    if "LayeredWeightedPhysics" in config["training"].get("losses", []):
        # Older hello-world YAML includes this unused duplicate. The loss reads loss_args.
        config["training"].pop("base_weight", None)
    allowed = {field["key"]: field for field in descriptors(config)}
    if not isinstance(changes, dict):
        raise ValueError("Parameter changes must be an object")
    for key, value in changes.items():
        if key not in allowed:
            raise ValueError(f"Current configuration does not support parameter {key}")
        field = allowed[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if not field["min"] <= value <= field["max"] or (field["integer"] and int(value) != value):
            raise ValueError(f"{key} must be in the range {field['min']}–{field['max']}" + (" and must be an integer" if field["integer"] else ""))
        put(config, field["path"], int(value) if field["integer"] else value)
    if config["data"].get("val_ratio", 0) + config["data"].get("test_ratio", 0) >= 1:
        raise ValueError("Validation and test split proportions must sum to less than 1")
    if not any(config["training"].get("loss_weights", [])):
        raise ValueError("At least one loss weight must be positive")
    if "num_layers" in changes:
        for key, value in config.get("callbacks", {}).items():
            if isinstance(value, str) and re.search(r"layer_\d+_residual", value):
                config["callbacks"][key] = re.sub(r"layer_\d+_residual", f"layer_{config['model']['num_layers']-1}_residual", value)
    return config

def describe(config):
    fields = descriptors(config)
    layers = config["model"].get("num_layers", 1)
    decay = next((f["value"] for f in fields if f["key"] == "base_weight"), None)
    raw = [decay ** (layers-i-1) for i in range(layers)] if decay is not None else []
    weights = [w / sum(raw) for w in raw] if raw else []
    return {"fields": fields, "values": {f["key"]: f["value"] for f in fields},
            "task": config["task"]["task_name"], "model": config["model"]["type"],
            "networks": config["data"].get("networks", []), "scenarios": config["data"].get("scenarios", []),
            "losses": config["training"].get("losses", []), "layer_weights": weights,
            "yaml": yaml.safe_dump(config, sort_keys=False), "paper": PAPER,
            "sources": [MODEL_SOURCE, LOSS_SOURCE, "gridfm_graphkit/io/param_handler.py"],
            "architecture_supported": config["model"]["type"] == "GNS_heterogeneous"}
