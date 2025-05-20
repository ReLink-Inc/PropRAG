from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel, CacheNVEmbedV2EmbeddingModel
from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(config: BaseConfig, use_cache: bool = True):
    embedding_model_name = config.embedding_model_name
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return CacheNVEmbedV2EmbeddingModel.from_experiment_config(config) if use_cache else NVEmbedV2EmbeddingModel(global_config=config, embedding_model_name=config.embedding_model_name)
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"
