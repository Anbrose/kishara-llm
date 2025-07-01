import logging
import torch
from server.models.models import Model, ModelEngine
from server.config import MODEL_MAP
from server.models.dtos import GenerationParams
from torch import dtype

# vllm doesn't work on m1, so we import it this way to get around the issue of running
# unit tests locally on macs
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


DEFAULT_DEVICE_MAP = "auto"

logger = logging.getLogger("APP").getChild(__name__)

class Generator:
    """
    A generator that wraps over the implementation details of different LLM engines.
    """

    _MAX_SPLIT = 1

    def __init__(
        self,
        model_type: Model,
        device_map: str = DEFAULT_DEVICE_MAP,
        torch_dtype: dtype = torch.bfloat16,
        model_engine: ModelEngine = ModelEngine.VLLM,
        lora_adaptor: str | None = None,
        stop_on_sequence: str = "\n\n\n\n",
        **kwargs,
    ):
        model_name_or_path, prompt_model = MODEL_MAP[model_type]
        self.prompt_model = prompt_model

        if model_engine == ModelEngine.HUGGING_FACE:
            self._set_hf_model(
                model_name_or_path, device_map, torch_dtype, stop_on_sequence, **kwargs
            )
        elif model_engine == ModelEngine.LORA:
            self._set_lora_model(model_name_or_path, lora_adaptor, **kwargs)
        elif model_engine == ModelEngine.VLLM:
            self._set_vllm_model(model_name_or_path, **kwargs)
        elif model_engine == ModelEngine.SGLANG:
            self._set_sglang_model(
                model_name_or_path, device_map, torch_dtype, stop_on_sequence, **kwargs
            )
        else:
            raise RuntimeError(
                "Invalid model engine, choose between vllm, lora, hf or sglang"
            )

        self.model_engine = model_engine

    def _set_sglang_model(
        self, model_name_or_path, device_map, torch_dtype, stop_on_sequence, **kwargs
    ):
        # mem_fraction_static controls how much mem the model uses for weights + kv cache
        # TODO: check load dtype
        runtime = sgl.Runtime(model_path=model_name_or_path, mem_fraction_static=0.65)
        sgl.set_default_backend(runtime)

        self.tokenizer = runtime.get_tokenizer()

        # prevent OOM issues
        self.MAX_TOKENS = kwargs.get("max_tokens", 2800)

        self.max_tokens_out = kwargs.get("max_new_tokens", 550)
        self.temperature = kwargs.get("temperature", 0.0)

        self.stop_on_sequence = stop_on_sequence

    def _set_hf_model(
        self, model_name_or_path, device_map, torch_dtype, stop_on_sequence, **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_cache=True,
            use_flash_attention_2=False,
        )

        generation_config = self.model.generation_config
        generation_config.top_p = kwargs.get("top_p", 1.0)
        generation_config.top_k = kwargs.get("top_k", 1)

        generation_config.num_return_sequences = kwargs.get("num_return_sequences", 1)
        generation_config.temperature = kwargs.get("temperature", 0.0)
        generation_config.max_new_tokens = kwargs.get("max_new_tokens", 550)
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.do_sample = False
        self.generation_config = generation_config
        # Although models can take upto 32K tokens we limit it to 2800 to
        # prevent OOM issues
        self.MAX_TOKENS = kwargs.get(
            "max_tokens", self.model.config.max_position_embeddings
        )

        self.max_tokens_out = kwargs.get("max_new_tokens", 550)
        self.stop_on_sequence = stop_on_sequence

    def _set_lora_model(self, model_name_or_path, lora_adaptor, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_lora_model(model_name_or_path, model_name=lora_adaptor)

        generation_config = self.model.generation_config
        generation_config.top_p = kwargs.get("top_p", 1.0)
        generation_config.top_k = kwargs.get("top_k", 3)

        generation_config.num_return_sequences = kwargs.get("num_return_sequences", 1)
        generation_config.temperature = kwargs.get("temperature", 0.1)
        generation_config.max_new_tokens = kwargs.get("max_new_tokens", 500)
        generation_config.pad_token_id = self.tokenizer.eos_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.do_sample = True

        self.generation_config = generation_config
        self.MAX_TOKENS = self.model.config.max_position_embeddings
        self.max_tokens_out = kwargs.get("max_new_tokens", 500)

    def _set_vllm_model(self, model_name_or_path, **kwargs):
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k", 3)

        num_return_sequences = kwargs.get("num_return_sequences", 1)
        temperature = kwargs.get("temperature", 0.1)
        max_new_tokens = kwargs.get("max_new_tokens", 500)

        self.sampling_params = SamplingParams(
            n=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        # Load model
        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            dtype="bfloat16",
            gpu_memory_utilization=0.6,
        )
        self.MAX_TOKENS = self.model.llm_engine.model_config.max_model_len
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens_out = self.sampling_params.max_tokens

    def generate(
        self,
        model_input: list[str] | str,
        generation_params: GenerationParams | None = None,
    ) -> list[str]:
        if (
            self.model_engine == ModelEngine.HUGGING_FACE
            or self.model_engine == ModelEngine.LORA
        ):
            return self._generate_hf(model_input, generation_params)

        elif self.model_engine == ModelEngine.VLLM:
            return self._generate_vllm(model_input)
        elif self.model_engine == ModelEngine.SGLANG:
            return self._generate_sglang(model_input, generation_params)

    def _generate_hf(
        self,
        prompts: list[str] | str,
        generation_params: GenerationParams | None = None,
    ) -> list[str]:
        encoding = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            # add_special_tokens=False,
        ).to(self.model.device)

        stopping_criteria = []

        if self.stop_on_sequence:
            stopping_tokens = self.tokenizer.encode(
                self.stop_on_sequence,
                # add_special_tokens=False
            )
            stopping_criteria.append(TokensStopCriteria(stopping_tokens))

        generation_config = copy.deepcopy(self.generation_config)
        # Set generation config using defaults
        # key is our param name, value is the models attribute name for that param
        if generation_params:
            attributes = generation_params.model_dump()
            attributes["max_new_tokens"] = attributes["max_tokens"]

            for param, val in attributes.items():
                # if param in generation_params and generation_params[param] is not None:
                if hasattr(generation_config, param):
                    setattr(generation_config, param, val)
                else:
                    logger.info(f"Ignoring generation param {param}")

            generation_config.do_sample = generation_params.temperature > 0

        outputs = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

        full_response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # if the prompt is not formatted correctly, INST_END may not be present or the model might not output that token
        # but output text after that token
        assistant_section = [
            response.split(self.prompt_model.INST_END, self._MAX_SPLIT)[1]
            if self.prompt_model.INST_END in response
            else response
            for response in full_response
        ]
        return assistant_section

    def _generate_sglang(
        self,
        prompts: list[str] | str,
        generation_params: GenerationParams | None = None,
    ) -> list[str]:
        if generation_params is None:
            generation_params = GenerationParams(
                temperature=self.temperature,
                stop=[self.stop_on_sequence],
                max_tokens=self.max_tokens_out,
                top_p=1.0,
                top_k=3,
            )

        @sgl.function
        def generate(s, prompt):
            s += prompt
            s += sgl.gen(
                "output",
                max_tokens=generation_params.max_tokens,
                temperature=generation_params.temperature,
                stop=generation_params.stop,
                regex=generation_params.regex,
                top_k=generation_params.top_k,
                top_p=generation_params.top_p,
                frequency_penalty=generation_params.frequency_penalty,
                presence_penalty=generation_params.presence_penalty,
                ignore_eos=generation_params.ignore_eos,
                choices=generation_params.choices,
                dtype=generation_params.dtype,
            )

        state = generate.run(prompts)

        return [state.get_var("output")]

    def _generate_vllm(self, prompt: list[str] | str) -> list[str]:
        outputs = self.model.generate(prompt, self.sampling_params)
        full_response = [output.outputs[0].text for output in outputs]

        return full_response