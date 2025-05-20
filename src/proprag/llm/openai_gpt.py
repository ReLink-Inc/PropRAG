import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from packaging import version

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response_optimized(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        lock_file = self.cache_file_name + ".lock"
        use_cache_flag = kwargs.pop("use_cache", True)

        # 1. Try reading from cache WITHOUT file lock (but ensure table exists)
        # Ensure table exists (brief lock if needed, only first time)
        # This initial check could also be done once during __init__
        with FileLock(lock_file):
             conn = sqlite3.connect(self.cache_file_name, timeout=180.0) # Add timeout
             try:
                 c = conn.cursor()
                 c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                 conn.commit()
             finally:
                 conn.close()


        cached_result = None
        if use_cache_flag:
            try:
                # Use shared cache mode if possible, helps with concurrent reads
                conn = sqlite3.connect(f"file:{self.cache_file_name}?mode=ro&cache=shared", uri=True, timeout=10.0)
                try:
                    c = conn.cursor()
                    c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                    row = c.fetchone()
                    if row is not None:
                        message, metadata_str = row
                        metadata = json.loads(metadata_str)
                        cached_result = (message, metadata, True) # Mark as hit
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                # Handle cases like DB locked (though ro mode should minimize this) or file not found
                logger.warning(f"Cache read error (key: {key_hash[:8]}...): {e}. Will proceed without cache for this call.")
            except Exception as e:
                logger.error(f"Unexpected cache read error (key: {key_hash[:8]}...): {e}. Will proceed without cache.")


        if cached_result:
             # logging.debug(f"Cache hit for key {key_hash[:8]}") # Optional debug log
             return cached_result

        # --- Cache miss ---
        # logging.debug(f"Cache miss for key {key_hash[:8]}") # Optional debug log
        result = func(self, *args, **kwargs)
        message, metadata = result

        # 2. Write to cache WITH file lock
        try:
            with FileLock(lock_file, timeout=180): # Add timeout to lock acquisition
                conn = sqlite3.connect(self.cache_file_name, timeout=180.0)
                try:
                    c = conn.cursor()
                    # Ensure table exists again (minimal overhead)
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS cache (
                            key TEXT PRIMARY KEY,
                            message TEXT,
                            metadata TEXT
                        )
                    """)
                    metadata_str = json.dumps(metadata)
                    c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                              (key_hash, message, metadata_str))
                    conn.commit()
                    # logging.debug(f"Cache written for key {key_hash[:8]}") # Optional debug log
                except sqlite3.Error as e:
                    logger.error(f"Cache write SQLite error (key: {key_hash[:8]}...): {e}")
                finally:
                    conn.close()
        except TimeoutError: # From FileLock timeout
             logger.warning(f"Could not acquire cache lock for writing key {key_hash[:8]}... within timeout.")
        except Exception as e:
             logger.error(f"Unexpected cache write error (key: {key_hash[:8]}...): {e}")


        return message, metadata, False # Mark as miss

    return wrapper


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation

            if kwargs.get("use_cache", True):
                c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                if row is not None:
                    message, metadata_str = row
                    metadata = json.loads(metadata_str)
                    # return cached result and mark as hit
                    return message, metadata, True
            conn.close()
        kwargs.pop("use_cache", None)
        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper


class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, **config_dict)

    def __init__(self, cache_dir, cache_filename: str = None,
                 llm_name: str = "gpt-4o-mini", api_key: str = None, llm_base_url: str = None, 
                 high_throughput: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url

        # self.llm_base_url = "https://api.openai.com/v1"
        # self.llm_name = "gpt-4o-mini"
        self._init_llm_config(**kwargs)
        # if high_throughput:
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=500)
        client = httpx.Client(limits=limits, timeout=httpx.Timeout(3*60, read=2*60))
        # else:
        #     client = None
        self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=api_key, http_client=client)

    def _init_llm_config(self, **kwargs) -> None:
        config_dict = {
            "llm_name": self.llm_name,
            "llm_base_url": self.llm_base_url,
            "generate_params": {
                "model": self.llm_name,
                "max_completion_tokens": kwargs.get("max_new_tokens", 8192),
                "n": kwargs.get("num_gen_choices", 1),
                "seed": kwargs.get("seed", 0),
                "temperature": kwargs.get("temperature", 0.0),
            }
        }
        logger.info(f"LLM generate params: {config_dict['generate_params']}")
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response_optimized
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")
        params['extra_body'] = {"provider": {"allow_fallbacks": False, "order": ["Nebius"]}}

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')
        response_checker = params.pop('response_checker', None)
        while True:
            try:
                response = self.openai_client.chat.completions.create(**params)
            except openai.APITimeoutError:
                logger.error(f"API timeout, try again: {params['messages']}")
                continue
            except Exception as e:
                logger.error(f"HTTP timeout, try again: {str(e)}")
                continue

            try:
                response_message = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
            except Exception as e:
                logger.error(f"Error getting response message, try again: {e}")
                logger.error(f"Response: {response}")
                logger.error(f"Error causing passage: {params['messages'][-1]}")
                continue
                
            if response_checker is not None and not response_checker(response_message, finish_reason, params):
                continue
            break

        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata


