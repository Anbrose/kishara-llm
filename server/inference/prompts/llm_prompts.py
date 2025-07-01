from abc import abstractmethod

class LlmMetaPrompt:
    _INST_START = ""
    _SYSTEM_START = ""
    _SYSTEM_END = ""
    INST_END = ""

    @classmethod
    @abstractmethod
    def format_prompt(cls, system: str, instruction: str) -> str:
        raise NotImplementedError

def combine_system_and_instruction_prompt(
    system_prompt: str, instruction_prompt, prompt_suffix: str = ""
):
    if prompt_suffix:
        return f"{system_prompt} \n {instruction_prompt} \n {prompt_suffix}"
    else:
        return f"{system_prompt} \n {instruction_prompt}"


class Llama2Prompt(LlmMetaPrompt):
    _INST_START = "<s>[INST]"
    _SYSTEM_START = "<<SYS>>"
    _SYSTEM_END = "<</SYS>>"
    INST_END = "[/INST]"

    @classmethod
    def format_prompt(cls, system: str, instruction: str) -> str:
        """The Llama-2 prompt is in the following format:
        <s>[INST] <<SYS>>
            {System}
        <</SYS>>
            {User instruction}
        [/INST]
        """
        formatted_system_prompt = (
            f"{cls._INST_START} {cls._SYSTEM_START} {system} {cls._SYSTEM_END}"
        )
        formatted_instruction_prompt = f"{instruction} {cls.INST_END}"
        return combine_system_and_instruction_prompt(
            formatted_system_prompt, formatted_instruction_prompt
        )

class Llama4Prompt(LlmMetaPrompt):
    _INST_START = "<|begin_of_text|>"
    _SYSTEM_START = "<|header_start|>"
    _SYSTEM_END = "<|header_end|>"
    INST_END = "<|eot|>"

    @classmethod
    def format_prompt(cls, system: str, instruction: str) -> str:
        """The Llama-4 prompt is in the following format:
        <|begin_of_text|><|header_start|>system<|header_end|>

        You are a helpful assistant<|eot|><|header_start|>user<|header_end|>

        Answer who are you in the form of jeopardy?<|eot|><|header_start|>assistant<|header_end|>
        """
        formatted_system_prompt = (
            f"{cls._INST_START} {cls._SYSTEM_START} system {cls._SYSTEM_END} "
            f"{system} ${cls.INST_END}"
        )
        formatted_instruction_prompt = (f"{cls._SYSTEM_START} user {cls._SYSTEM_END} "
                                        f"{instruction} {cls.INST_END} {cls._SYSTEM_START} assistant {cls._SYSTEM_END}")
        return combine_system_and_instruction_prompt(
            formatted_system_prompt, formatted_instruction_prompt
        )



