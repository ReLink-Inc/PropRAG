import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    return CacheOpenAI.from_experiment_config(config)
    