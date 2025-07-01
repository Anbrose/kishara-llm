from typing import Any, Optional
import logging
from server.config import MODEL_CONFIG
from server.inference.llm.generator import Generator

logger = logging.getLogger("app").getChild(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class ModelFactory:
    """This class should be used to get an instance of a model"""

    _instance: Optional["ModelFactory"] = None

    def __init__(self) -> None:
        """
        DO NOT construct this class directly. The intention of the class is to be
        used as a singleton by calling ModelFactory.instance() to get back an
        instance of this class
        """

        key_model_mapping: dict[str, list[str]] = {
            "LLM": ["llm"],
        }

        self.models: dict[str, Any] = {}
        # load models required
        for key in MODEL_CONFIG:
            logger.info("Loading model {}".format(key))
            if key in key_model_mapping:
                for model in key_model_mapping[key]:
                    self.load_model(model)

    @classmethod
    def instance(cls) -> "ModelFactory":
        if cls._instance is None:
            raise RuntimeError("Models have not been initialised call initialise()")
        return cls._instance

    @classmethod
    def initialise(cls):
        cls._instance = cls()
        logger.debug(f"{MODEL_CONFIG=}")

    def load_model(self, model_name):
        # if model already loaded return
        if model_name in self.models:
            return

        if model_name == "llm":
            # Although models can take upto 32K tokens we limit it to 2800 to
            # prevent OOM issues
            self.models[model_name] = Generator(
                LLM_MODEL_TO_LOAD,
                model_engine=MODEL_ENGINE,
                **{"max_tokens": MAX_TOKENS},
            )
            self.models["inference_service"] = InferenceService(self.models[model_name])
            logger.info("Loaded llm model correctly")

    def matching_model(self) -> MatchingModel:
        return self.models["matching_model"]

    def llm_model(self) -> Generator:
        return self.models["llm"]

    def inference_service(self) -> InferenceService:
        return self.models["inference_service"]

    def whisper_model(self) -> "WhisperModel":
        return self.models["whisper"]

    def wav2vec_model(self) -> tuple[Any, dict[str, Any]]:
        return self.models["wav2vec"]

    def pyannote_model(self) -> Pipeline:
        return self.models["pyannote"]

    def ocr_model(self) -> OCRPipeline:
        return self.models["ocr"]

    def table_extraction_model(self) -> TableExtractionPipeline:
        return self.models["table_extraction"]

    def embeddings_model(self) -> SentenceTransformer:
        return self.models["embeddings"]

    def deepgram_session(self) -> AudioTranscriber:
        return self.models["deepgram"]

    @classmethod
    def _create_whisper_model(cls) -> "WhisperModel":
        return WhisperModel(
            FASTER_WHISPER_MODEL_LOCATION,
            device="cuda",
            compute_type="float16",
        )

    @classmethod
    def _create_deepgram_session(cls) -> AudioTranscriber:
        return AudioTranscriber(api_key=DEEPGRAM_KEY)