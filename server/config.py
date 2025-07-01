from server.models.models import Model
from server.inference.prompts.llm_prompts import LlmMetaPrompt, Llama2Prompt, Llama4Prompt
import os

# TODO: Define model locations - these should be configured via environment variable

MODEL_HOME = os.environ.get("MODEL_HOME", "/models")

# The 'MODELS' env variable should be a string in the form:
# LLM,WHISPER,OCR,QA,TRANSCRIPTION,MATCHING
MODEL_CONFIG_STRING = os.environ.get("MODELS", "")
MODEL_CONFIG = [model.strip() for model in MODEL_CONFIG_STRING.split(",")]

# always load
LLM_MODEL_TO_LOAD = Model.LLAMA4


LLAMA2_LOCATION = f"{MODEL_HOME}/meta-llama/Llama-2-7b"
LLAMA4_LOCATION = f"{MODEL_HOME}/meta-llama/Llama-4-Scout-17B-16E"

MODEL_MAP: dict[Model, tuple[str, type[LlmMetaPrompt]]] = {
    Model.LLAMA2: (LLAMA2_LOCATION, Llama2Prompt),
    Model.LLAMA4: (LLAMA4_LOCATION, Llama4Prompt),
}