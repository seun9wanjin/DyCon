from typing import Dict, Any, List, Tuple
import json
import os
from pydantic import BaseModel
from .llm import ask_llm

class Context(BaseModel):
    personal_context: str

class Refine(BaseModel):
    refine_rationale: str
    refined_personal_context: str

class UpdateContext(BaseModel):
    update_rationale: str
    updated_personal_context: str

def generate_context(correlation: str, target: str, config) -> str:
    """Generate a personal context narrative given a correlation string and sensor data."""
    prompt = config.context_gen_prompt.format(correlation=correlation, target=target)
    data = ask_llm(prompt, Context)
    return data['personal_context']

def update_context(
    target: str,
    prediction: float,
    gt_value: float,
    analysis: str,
    momentary_context: str,
    longitudinal_context: str,
    threshold: float,
    config
) -> Tuple[str, str]:
    """Update longitudinal context based on momentary context and prediction results."""
    prompt = config.context_update_prompt.format(
        target=target,
        prediction=prediction,
        gt_value=gt_value,
        prediction_analysis=analysis,
        momentary_context=momentary_context,
        longitudinal_context=longitudinal_context,
        threshold=threshold
    )
    data = ask_llm(prompt, UpdateContext)
    return data["update_rationale"], data["updated_personal_context"]

def refine_context(
    momentary_context: str,
    prediction: float,
    gt_value: float,
    analysis: str,
    target: str,
    correlations: str = None,
    config = None
) -> Tuple[str, str]:
    """Refine context based on prediction results."""
    prompt = config.context_refine_prompt.format(
        momentary_context=momentary_context,
        prediction=prediction,
        gt_value=gt_value,
        prediction_analysis=analysis,
        target=target,
        correlation=correlations
    )
    data = ask_llm(prompt, Refine)
    return data["refine_rationale"], data["refined_personal_context"]
