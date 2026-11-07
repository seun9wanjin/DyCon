from typing import Dict, Any, Tuple
from pydantic import BaseModel
from .llm import ask_llm

class AnxietyPrediction(BaseModel):
    prediction_analysis: str
    anxiety: float

class DepressionPrediction(BaseModel):
    prediction_analysis: str
    depression: float

def predict_score(
    momentary_context: str,
    data: Dict[str, Any],
    session: int,
    days: int,
    target: str,
    longitudinal_context: str = None,
    config = None
) -> Tuple[str, float]:
    """Return (analysis, predicted_score) using enforced JSON output."""
    if session == 0:
        prompt = config.predict_prompt_session_1.format(
            target=target,
            days=days,
            data=data,
            momentary_context=momentary_context
        )
    else:
        prompt = config.predict_prompt_after_session_1.format(
            target=target,
            days=days,
            data=data,
            momentary_context=momentary_context,
            longitudinal_context=longitudinal_context
        )

    if target == "depression":
        data = ask_llm(prompt, DepressionPrediction)
    else:
        data = ask_llm(prompt, AnxietyPrediction)

    return data["prediction_analysis"], data[target]
