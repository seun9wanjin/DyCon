import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

def save_json(data: Any, file_name: str) -> None:
    """Save data to file_name as indented JSON."""
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

def to_float_safe(x) -> float:
    """Safely convert a value to float."""
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(Decimal(str(x)))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Cannot convert to float: {x!r}")

def is_match(pred, gt, tol=1e-4) -> bool:
    """Check if prediction matches ground truth within tolerance."""
    pred_f, gt_f = to_float_safe(pred), to_float_safe(gt)
    return math.isclose(pred_f, gt_f, abs_tol=tol) 