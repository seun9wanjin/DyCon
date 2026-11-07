"""Runtime configuration. Credentials are read from the environment only."""
import os

FRAMEWORK_NAME = "DyCon"
FRAMEWORK_VERSION = "1.0.0"
model = ""
temperature = 0
max_tokens = 1000
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
llama_api_key = os.environ.get("LLAMA_API_KEY", "")
max_refines = 10
excluded_cols = ['index', 'pid', 'date', 'year', 'phq4', 'depression', 'anxiety', 'correlation']
context_gen_prompt = ""
context_update_prompt = ""
predict_prompt_session_1 = ""
predict_prompt_after_session_1 = ""
context_refine_prompt = ""
