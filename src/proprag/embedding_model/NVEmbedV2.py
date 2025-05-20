from copy import deepcopy
import atexit
import functools
import hashlib
import json
import os
import sqlite3
import threading
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel
from filelock import FileLock

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

# Global connection registry for persistent SQLite connections
_db_connections = {}
_connection_locks = {}
_registry_lock = threading.Lock()

def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Get a persistent SQLite connection for the given database path"""
    with _registry_lock:
        # Create connection lock if it doesn't exist
        if db_path not in _connection_locks:
            _connection_locks[db_path] = threading.Lock()
        
        # Get the lock for this specific connection
        lock = _connection_locks[db_path]
    
    # Lock to ensure only one thread accesses this connection at a time
    with lock:
        # Check if connection exists and is valid
        needs_new_connection = True
        if db_path in _db_connections and _db_connections[db_path] is not None:
            try:
                # Test the connection with a simple query
                _db_connections[db_path].execute("SELECT 1").fetchone()
                needs_new_connection = False
            except sqlite3.Error:
                # Connection is invalid or broken
                logger.debug(f"Existing SQLite connection for {db_path} is broken, creating new one")
                try:
                    _db_connections[db_path].close()
                except:
                    pass
                _db_connections[db_path] = None
        
        # Create connection if needed
        if needs_new_connection:
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create connection with optimized settings
            conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)  # Longer timeout for busy DB
            
            # Performance optimizations
            conn.execute("PRAGMA journal_mode = WAL")       # Write-ahead logging for better concurrency
            conn.execute("PRAGMA synchronous = NORMAL")     # Reduced disk sync frequency
            conn.execute("PRAGMA cache_size = 10000")       # Larger cache
            conn.execute("PRAGMA page_size = 8192")         # Larger page size for BLOBs
            conn.execute("PRAGMA temp_store = MEMORY")      # Store temp tables in memory
            conn.execute("PRAGMA busy_timeout = 30000")     # Wait up to 30 seconds if DB is locked
            
            # Create table if it doesn't exist
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON embedding_cache(key)")
            conn.commit()
            
            # Store the connection
            _db_connections[db_path] = conn
            
            logger.debug(f"Created persistent SQLite connection for {db_path}")
        
        return _db_connections[db_path]


def close_db_connection(db_path: str):
    """Close a specific database connection"""
    with _registry_lock:
        if db_path in _connection_locks:
            lock = _connection_locks[db_path]
            
            with lock:
                if db_path in _db_connections and _db_connections[db_path] is not None:
                    try:
                        # First, try to commit any pending changes
                        try:
                            _db_connections[db_path].commit()
                            logger.debug(f"Committed pending changes for {db_path}")
                        except sqlite3.Error as se:
                            logger.warning(f"Could not commit pending changes for {db_path}: {se}")
                        
                        # Then close the connection
                        logger.debug(f"Closing SQLite connection for {db_path}")
                        _db_connections[db_path].close()
                    except Exception as e:
                        logger.error(f"Error closing SQLite connection for {db_path}: {e}")
                    finally:
                        _db_connections[db_path] = None

def close_all_db_connections():
    """Close all database connections (called at exit)"""
    logger.info("Closing all SQLite cache connections...")
    
    # Make a copy of the keys to avoid modification during iteration
    with _registry_lock:
        db_paths = list(_db_connections.keys())
    
    # Close each connection
    for db_path in db_paths:
        try:
            close_db_connection(db_path)
        except Exception as e:
            logger.error(f"Error closing connection for {db_path}: {e}")
    
    logger.info("All SQLite connections closed")

# # Register cleanup for all connections at program exit
# atexit.register(close_all_db_connections)

# Also register signal handlers for common exit signals
# try:
#     import signal
    
#     def signal_handler(sig, frame):
#         logger.info(f"Signal {sig} received, closing database connections...")
#         close_all_db_connections()
#         # Re-raise the signal to allow the default handler to run
#         signal.signal(sig, signal.SIG_DFL)
#         os.kill(os.getpid(), sig)
    
#     # Register for common termination signals
#     signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
#     signal.signal(signal.SIGTERM, signal_handler)  # kill command
    
#     logger.debug("Registered signal handlers for database connection cleanup")
# except Exception as e:
#     logger.warning(f"Could not register signal handlers: {e}")


def cache_embeddings(func):
    @functools.wraps(func)
    def wrapper(self, texts, disable_tqdm=False, **kwargs):
        import time
        # total_start_time = time.time()
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Prepare parameters from embedding config
        # params_start_time = time.time()
        params = deepcopy(self.embedding_config.encode_params) if hasattr(self, "embedding_config") else {}
        if kwargs:
            params.update(kwargs)
        
        # Special handling for instruction
        instruction = ""
        if "instruction" in params:
            instruction = params["instruction"]
        # params_time = time.time() - params_start_time
        
        # Initialize cache key data
        cache_hits = []
        cache_misses = []
        cache_results = [None] * len(texts)
        key_hashes = []
        
        # Generate hash for each text
        # hash_start_time = time.time()
        for i, text in enumerate(texts):
            key_data = {
                "text": text,
                "instruction": instruction,
                "model": self.embedding_model_name,
                "norm": getattr(self.embedding_config, "norm", False)
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
            key_hashes.append(key_hash)
        # hash_time = time.time() - hash_start_time
        
        db_lookup_time = 0
        db_write_time = 0
        model_inference_time = 0
        
        # If cache file exists, check for hits
        if hasattr(self, "cache_file_name"):
            lock_file = self.cache_file_name + ".lock"
            
            # db_lookup_start = time.time()
            with FileLock(lock_file):
                # Get persistent connection (creates if doesn't exist)
                conn = get_db_connection(self.cache_file_name)
                c = conn.cursor()
                
                # Check for each hash in the cache - use a single query with IN clause
                if key_hashes:  # Only query if we have hashes to look up
                    placeholders = ', '.join(['?'] * len(key_hashes))
                    c.execute(f"SELECT key, embedding FROM embedding_cache WHERE key IN ({placeholders})", key_hashes)
                    rows = c.fetchall()
                    
                    # Process results
                    found_keys = {}
                    for key, embedding_bytes in rows:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        embedding = embedding.reshape(1, -1)  # Reshaping to 2D array
                        found_keys[key] = embedding
                    
                    # Process hits and misses
                    for i, key_hash in enumerate(key_hashes):
                        if key_hash in found_keys:
                            cache_results[i] = found_keys[key_hash]
                            cache_hits.append(i)
                        else:
                            cache_misses.append(i)
                else:
                    # No texts to process
                    cache_misses = []
                
                # Don't close the connection - it's persistent
            # db_lookup_time = time.time() - db_lookup_start
        else:
            # If no cache file is configured, all are misses
            cache_misses = list(range(len(texts)))
        
        # Process cache misses
        if cache_misses:
            # Create a new list with only the texts that weren't in the cache
            miss_texts = [texts[i] for i in cache_misses]
            
            # Call the original function to generate embeddings for misses
            # model_start_time = time.time()
            miss_results = func(self, miss_texts, disable_tqdm, **kwargs)
            # model_inference_time = time.time() - model_start_time
            
            # Update cache with new embeddings
            if hasattr(self, "cache_file_name"):
                # db_write_start = time.time()
                
                # Prepare batch insertion
                batch_data = []
                for i, miss_idx in enumerate(cache_misses):
                    # Get the embedding for this miss
                    if isinstance(miss_results, list):
                        embedding = miss_results[i]
                    else:
                        embedding = miss_results[i:i+1]
                    
                    # Convert to numpy if it's a tensor
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    
                    # Store in our results array
                    cache_results[miss_idx] = embedding
                    
                    # Store in cache
                    key_hash = key_hashes[miss_idx]
                    embedding_bytes = embedding.tobytes()
                    batch_data.append((key_hash, embedding_bytes))
                
                # Write to persistent database connection
                with FileLock(lock_file):
                    # Get the persistent connection
                    conn = get_db_connection(self.cache_file_name)
                    c = conn.cursor()
                    
                    # Use executemany for batch insertion
                    c.executemany("INSERT OR REPLACE INTO embedding_cache (key, embedding) VALUES (?, ?)", batch_data)
                    conn.commit()
                    # Don't close the connection - it's persistent
                # db_write_time = time.time() - db_write_start
            else:
                # If no cache, just use the original results
                for i, miss_idx in enumerate(cache_misses):
                    if isinstance(miss_results, list):
                        cache_results[miss_idx] = miss_results[i]
                    else:
                        cache_results[miss_idx] = miss_results[i:i+1]
        
        # Combine all results
        # combine_start = time.time()
        result = None
        if all(isinstance(r, np.ndarray) for r in cache_results):
            result = np.vstack(cache_results)
        elif all(isinstance(r, torch.Tensor) for r in cache_results):
            result = torch.cat(cache_results, dim=0)
        else:
            # Convert any numpy arrays to torch tensors for consistency
            result = np.vstack([r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in cache_results])
        # combine_time = time.time() - combine_start
        
        # total_time = time.time() - total_start_time
        
        # Log performance metrics
        # hit_ratio = len(cache_hits) / len(texts) if texts else 0
        # logger.info(f"Embedding cache stats for {len(texts)} texts:")
        # logger.info(f"  Hit ratio: {hit_ratio:.2f} ({len(cache_hits)}/{len(texts)})")
        # logger.info(f"  Total time: {total_time:.4f}s")
        # logger.info(f"  Time breakdown:")
        # logger.info(f"    Params setup: {params_time:.4f}s")
        # logger.info(f"    Hashing: {hash_time:.4f}s")
        # logger.info(f"    DB lookup: {db_lookup_time:.4f}s")
        # logger.info(f"    Model inference: {model_inference_time:.4f}s")
        # logger.info(f"    DB write: {db_write_time:.4f}s")
        # logger.info(f"    Combining results: {combine_time:.4f}s")
        
        return result

    return wrapper


class NVEmbedV2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_dim = self.embedding_model.config.hidden_size
        
        # Print model's data type to check if it's using fp16 or fp32
        for name, param in self.embedding_model.named_parameters():
            print(f"Model parameter dtype: {param.dtype}")
            break  # Just print the first parameter's dtype as an example

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        
        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "torch_dtype": "float16",  # Use half-precision (16-bit) to reduce memory usage
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }
        print("config_dict: ", config_dict)
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    # def _add_eos(self, texts: List[str]) -> List[str]:
    #     # Adds EOS token to each text
    #     return [text + self.embedding_model.tokenizer.eos_token for text in texts]

    def batch_encode(self, texts: List[str], disable_tqdm: bool = False, **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        batch_size = params.pop("batch_size", 16)
        # print("batch_size: ", batch_size)

        print(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["prompts"] = texts  # self._add_eos(texts=texts)
            results = self.embedding_model.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding", disable=disable_tqdm)
            results = []
            for i in range(0, len(texts), batch_size):
                params["prompts"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results


class CacheNVEmbedV2EmbeddingModel(NVEmbedV2EmbeddingModel):
    """NVEmbedV2 embedding model with caching capabilities."""
    
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheNVEmbedV2EmbeddingModel":
        config_dict = global_config.__dict__
        cache_dir = os.path.join(global_config.save_dir, "embedding_cache")
        return cls(cache_dir=cache_dir, global_config=global_config, embedding_model_name=global_config.embedding_model_name)
    
    def __init__(self, cache_dir: str, cache_filename: str = None, 
                 global_config: Optional[BaseConfig] = None,
                 embedding_model_name: Optional[str] = None) -> None:
        
        super().__init__(global_config=global_config, embedding_model_name=embedding_model_name)
        
        # Setup cache directory and file
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if cache_filename is None:
            cache_filename = f"{self.embedding_model_name.replace('/', '_')}_cache.sqlite"
        
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)
        logger.debug(f"Initialized embedding cache at: {self.cache_file_name}")
        
        # Initialize the persistent connection
        _ = get_db_connection(self.cache_file_name)
    
    def close_connection(self):
        """Explicitly close the database connection"""
        if hasattr(self, 'cache_file_name'):
            close_db_connection(self.cache_file_name)
    
    def __del__(self):
        """Handle cleanup when object is deleted"""
        try:
            # Connection will be closed by atexit handler if not closed explicitly
            pass
        except Exception as e:
            logger.error(f"Error during embedding model cleanup: {e}")
    
    @cache_embeddings
    def batch_encode(self, texts: List[str], disable_tqdm: bool = False, **kwargs) -> None:
        """Overrides the parent class method to add caching functionality"""
        return super().batch_encode(texts, disable_tqdm, **kwargs)
