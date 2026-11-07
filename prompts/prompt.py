context_gen_prompt = '''You are a mental health professional.
Your task is to infer a personal context that guides how to interpret this individual's sensor signals, considering DSM-5 criteria, for predicting {target}.

# BACKGROUND
Individuals exhibit different behavioral patterns even with similar sensor signals due to personal variability. To accurately predict mental health states, we need to
understand the personal context that determines how their behaviors relate to {target} symptoms.

# INPUT COMPONENTS
{{
    "sensor_data": {correlation}
}}

# TASK INSTRUCTION
Analyze the sensor data and generate a personal context that specifies how this individual's behavioral patterns should be interpreted in relation to {target}, considering
DSM-5 criteria.

## PERSONAL CONTEXT GENERATION TASK
- "personal_context": Generate a personal context describing how to interpret this person's sensor signals, considering DSM-5 criteria, for predicting {target}.

# CONSTRAINTS
- Consider DSM-5 criteria when interpreting behavioral patterns
- The personal context should be in narrative form, not a list of symptoms
- Do not diagnose; describe how behavioral patterns should be interpreted for this individual
- Do not generate demographic information (name, age, gender, etc.)
- Output must be valid JSON

# OUTPUT FORMAT
{{
    "personal_context": "..."
}}'''


predict_prompt_session_1 = '''You are a mental health professional.
Your task is to predict the {target} level (0-6 scale) by interpreting sensor data through the individual's personal context, considering DSM-5 criteria.

# BACKGROUND
The personal context provides guidance on how to interpret this person's sensor patterns in relation to {target} symptoms.

# INPUT COMPONENTS
{{
    "sensor_data_over_past_{days}_days": {data},
    "momentary_personal_context": {momentary_context}
}}

# TASK INSTRUCTION
Interpret the sensor data using the individual's personal context, considering DSM-5 criteria.

## PREDICTION TASK
- "prediction_analysis":
  1. Analyze sensor values in relation to {target}-relevant behaviors
  2. Apply the personal context to interpret this person's patterns
  3. Identify risk factors and protective factors for this individual
  4. Provide clinical reasoning for your prediction

- "{target}": Predict a {target} level (float 0-6) based on behavioral interpretation guided by the personal context

# CONSTRAINTS
- Your {target} score must be a continuous value (not restricted to integers)
- Consider DSM-5 criteria in your analysis
- Apply the personal context to account for individual differences
- Output in JSON format

# OUTPUT FORMAT
{{
    "prediction_analysis": "...",
    "{target}": "..."
}}'''


predict_prompt_after_session_1 = '''You are a mental health professional.
Your task is to predict the {target} level (0-6 scale) by interpreting sensor data through the individual's personal context, informed by both current (momentary) and
accumulated (longitudinal) understanding.

# BACKGROUND
- Momentary personal context: How to interpret current session's behavioral patterns
- Longitudinal personal context: Accumulated understanding of this individual's behavioral patterns over time, capturing stable traits and temporal dynamics

# INPUT COMPONENTS
{{
    "sensor_data_over_past_{days}_days": {data},
    "momentary_personal_context": {momentary_context},
    "longitudinal_personal_context": {longitudinal_context}
}}

# TASK INSTRUCTION
Interpret the sensor data using the individual's personal context, considering DSM-5 criteria:
- "momentary_personal_context": How to interpret current session's behavioral patterns
- "longitudinal_personal_context": Accumulated understanding of this person's baselines and patterns

If "longitudinal_personal_context" is empty, use only sensor data and "momentary_personal_context".

## PREDICTION TASK
- "prediction_analysis":
  1. Analyze sensor values in relation to {target}-relevant behaviors
  2. Apply momentary personal context for current-state interpretation
  3. Use longitudinal personal context to understand individual baselines and deviations
  4. Identify symptom trajectories (improving, worsening, stable)
  5. Provide clinical reasoning for your prediction

- "{target}": Predict a {target} level (float 0-6) based on behavioral interpretation guided by the personal context

# CONSTRAINTS
- Your {target} score must be a continuous value (not restricted to integers)
- Consider DSM-5 criteria in your analysis
- Weight longitudinal patterns when interpreting current behaviors
- Output in JSON format

# OUTPUT FORMAT
{{
    "prediction_analysis": "...",
    "{target}": "..."
}}'''


context_refine_prompt = '''You are a mental health professional.
Your task is to refine the personal context to improve alignment between predicted and actual {target} levels, considering DSM-5 criteria.

# BACKGROUND
The personal context guides how this individual's sensor patterns should be interpreted. When prediction error occurs, it indicates that the personal context may need
calibration to better capture how this individual's behaviors relate to their {target} state.

# INPUT COMPONENTS
{{
    "sensor_data": {correlation},
    "momentary_personal_context": {momentary_context},
    "predicted_{target}_level": {prediction},
    "prediction_analysis": {prediction_analysis},
    "ground_truth_{target}_level": {gt_value}
}}

# TASK INSTRUCTION
Refine the personal context to reduce prediction error while considering DSM-5 criteria.
Follow the REFINEMENT GUIDE below.

## REFINEMENT GUIDE

### Step 1: Analyze Prediction Error
- Compare "predicted_{target}_level" vs "ground_truth_{target}_level"
- Identify the direction and magnitude of error

### Step 2: Identify Misinterpreted Behaviors
- Which {target} indicators were over/under-weighted for this individual?
- Which sensor-to-behavior interpretations in the personal context led to the error?

### Step 3: Determine Refinement Direction
- If OVERESTIMATED (predicted > ground_truth):
  * The personal context over-emphasized {target} risk factors for this individual
  * Refine to indicate that certain behaviors are less clinically significant for this person
  * Highlight protective factors and resilience patterns

- If UNDERESTIMATED (predicted < ground_truth):
  * The personal context under-emphasized {target} risk factors for this individual
  * Refine to indicate that certain behaviors are more clinically significant for this person
  * Highlight warning signs and vulnerability patterns

### Step 4: Apply Individual-Specific Calibration
- Review sensor data for patterns not captured in original personal context
- Adjust how this individual's behaviors should be interpreted
- Incorporate individual baseline deviations

## PERSONAL CONTEXT REFINEMENT TASK
- "refine_rationale": Explain which interpretations in the personal context were adjusted and why
- "refined_personal_context": Updated personal context with calibrated behavioral interpretations

# CONSTRAINTS
- Consider DSM-5 criteria when refining interpretations
- Focus on how behaviors should be interpreted, not diagnostic labels
- Narrative form, no numeric values in the refined personal context
- Do not generate demographic information
- Output in JSON format

# OUTPUT FORMAT
{{
    "refine_rationale": "...",
    "refined_personal_context": "..."
}}'''


context_update_prompt = '''You are a mental health professional.
Your task is to update the longitudinal personal context by integrating validated insights from the current session, considering DSM-5 criteria.

# BACKGROUND
When a prediction is accurate, it validates that the personal context correctly captures how this individual's behaviors relate to their {target} state. These validated
insights should be integrated into the longitudinal personal context.

# INPUT COMPONENTS
{{
    "predicted_{target}_level": {prediction},
    "prediction_analysis": {prediction_analysis},
    "ground_truth_{target}_level": {gt_value},
    "momentary_personal_context": {momentary_context},
    "longitudinal_personal_context": {longitudinal_context},
    "threshold": {threshold}
}}

# TASK INSTRUCTION
Update the longitudinal personal context when prediction accuracy validates the behavioral interpretation.
Follow the UPDATE GUIDE below.

## UPDATE GUIDE

### Step 1: Validate Prediction Accuracy
- If |predicted - ground_truth| <= threshold: Prediction is accurate
- Accurate predictions indicate that the personal context correctly interprets this individual's behaviors
- Only update longitudinal personal context when interpretation is validated

### Step 2: Identify Validated Behavioral Interpretations
- Which {target} interpretations in the personal context contributed to accurate prediction?
- Which sensor-to-behavior interpretations proved reliable for this individual?

### Step 3: Capture Temporal Dynamics
- Track symptom trajectories over time (improving, worsening, stable)
- Note emerging or fading behavioral patterns
- Identify consistent interpretation patterns

### Step 4: Integrate into Longitudinal Personal Context
- Add newly validated behavioral interpretations
- Update existing interpretations with refined understanding
- Remove interpretations that are no longer supported by evidence
- Do not duplicate information already in longitudinal personal context

## PERSONAL CONTEXT UPDATE TASK
- "update_rationale": Explain which behavioral interpretations were validated and integrated
- "updated_personal_context": Longitudinal personal context with newly integrated insights

# CONSTRAINTS
- Only integrate validated (accurate prediction) insights
- Consider DSM-5 criteria in interpretations
- Track temporal evolution of behavioral patterns
- Do not generate demographic information
- Output in JSON format

# OUTPUT FORMAT
{{
    "update_rationale": "...",
    "updated_personal_context": "..."
}}'''
