
from pydantic import BaseModel
class GenerationParams(BaseModel):
    temperature: float
    top_p: float | None
    top_k: int | None
    max_tokens: int
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    regex: str | None = None
    ignore_eos: bool | None = False
    choices: list[str] | None = None
    dtype: str | None = None