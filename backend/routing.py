"""Validate model intent before any action; keep questions strictly read-only."""
ROUTES = ["QUESTION_GENERAL", "QUESTION_PARAMETER", "PARAMETER", "GENERATION", "ANALYSIS", "PLANNER"]
READ_ONLY = {"QUESTION_GENERAL", "QUESTION_PARAMETER", "ANALYSIS"}
ALIASES = {"lr": "learning_rate", "heads": "attention_head", "num_heads": "attention_head",
           "layers": "num_layers", "lambda": "base_weight", "λ": "base_weight"}

def intent_schema(fields):
    patch = {"type": "object", "properties": {key: {"type": "number"} for key in fields}, "additionalProperties": False}
    step = {"type": "object", "properties": {"route": {"type": "string", "enum": ROUTES[:-1]},
            "changes": patch, "instruction": {"type": "string"}}, "required": ["route", "changes", "instruction"], "additionalProperties": False}
    return {"type": "object", "properties": {"route": {"type": "string", "enum": ROUTES},
            "changes": patch, "steps": {"type": "array", "items": step, "maxItems": 8}},
            "required": ["route", "changes", "steps"], "additionalProperties": False}


def changes(value):
    if not isinstance(value, dict):
        raise ValueError("changes must be an object of parameter values")
    result = {}
    for key, number in value.items():
        key = ALIASES.get(key, key)
        if key in result and result[key] != number:
            raise ValueError(f"{key} has conflicting values")
        result[key] = number
    return result


def validate_intent(parsed, requested, validate):
    route = requested or parsed.get("route")
    if route not in ROUTES:
        raise ValueError("Model returned an unknown route")
    # Even a misbehaving model cannot change parameters in a question/analysis route.
    patch = {} if route in READ_ONLY else changes(parsed.get("changes", {}))
    steps = []
    if route == "PLANNER":
        raw = parsed.get("steps")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
            raise ValueError("Plan must contain 1–8 steps")
        accumulated = {}
        for item in raw:
            if not isinstance(item, dict) or item.get("route") not in ROUTES[:-1]:
                raise ValueError("Plan contains an unknown step")
            instruction = item.get("instruction", "")
            if not isinstance(instruction, str) or len(instruction) > 8000:
                raise ValueError("Step instruction must be text of at most 8000 characters")
            delta = {} if item["route"] in READ_ONLY else changes(item.get("changes", {}))
            steps.append({"route": item["route"], "changes": delta, "instruction": instruction})
        # A top-level patch must be visible as a parameter step, never silently lost.
        if patch:
            if steps[0]["route"] == "PARAMETER":
                steps[0]["changes"] = {**patch, **steps[0]["changes"]}
            elif len(steps) < 8:
                steps.insert(0, {"route": "PARAMETER", "changes": patch, "instruction": "Apply explicitly specified parameters"})
            else:
                raise ValueError("Merge parameter changes into a PARAMETER plan step")
        for step in steps:
            if step["route"] == "PARAMETER" and not step["changes"]:
                raise ValueError("Parameter step has no explicit values; provide parameter names and numbers")
            accumulated.update(step["changes"])
            validate(accumulated)
        if requested != "PLANNER" and all(s["route"] == "PARAMETER" for s in steps):
            seen, conflicts = {}, False
            for step in steps:
                for key, value in step["changes"].items():
                    if key in seen and seen[key] != value: conflicts = True
                    seen[key] = value
            if not conflicts:
                return {"route": "PARAMETER", "changes": seen, "steps": []}
    else:
        if route == "PARAMETER" and not patch:
            raise ValueError("No explicit parameter change recognized; for example, write: set epochs to 3")
        validate(patch)
    return {"route": route, "changes": patch, "steps": steps}
