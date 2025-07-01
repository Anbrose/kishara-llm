from dataclasses import dataclass
from server.models.llm import LLMTask
from enum import Enum

from fastapi import APIRouter
from server.factories.model_factory import ModelFactory

router = APIRouter(
    prefix="/python/demo_qa",
    tags=["demo"],
    responses={404: {"description": "Not found"}},
)

@dataclass
class DemoQAResult:
    answer: str

@router.post("/generate_qa_test")
async def generate_answer_qa(context: str, question: str, history: str) -> str | None:
    task = LLMTask(
        id="1",
        input=context,
        params={"question": question, "history": history},
    )
    return ModelFactory.instance().inference_service().do_inference(task)[1]

@dataclass
class GenerateRequest:
    question: str

@router.post("/generate")
async def generate_answer(request: GenerateRequest) -> DemoQAResult:
    llm = ModelFactory.instance().llm_model()

    prompt = llm.prompt_model.format_prompt(
        "你的名字是老张。你的回答都应该以喵结尾。", request.question
    )

    answer = llm.generate(prompt)

    return DemoQAResult(answer=answer[0])

