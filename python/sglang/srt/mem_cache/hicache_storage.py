import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch

from dataclasses import dataclass
import threading
import time
import rocksdb_binding as rocksdb
from sglang.srt.mem_cache.safetensor_helper import SafetensorHelper
import functools
import threading

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# Thread-local context to track batch nesting
_stats_context = threading.local()
_stats_context.depth = 0

def record_stats(op_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Init if not set
            if not hasattr(_stats_context, "depth"):
                _stats_context.depth = 0

            top_level = _stats_context.depth == 0
            _stats_context.depth += 1

            try:
                if not top_level:
                    return func(self, *args, **kwargs)

                stats = self.statistics
                count_attr = f"n_{op_name}s"
                time_attr = f"t_{op_name}s"

                start = time.perf_counter()
                result = func(self, *args, **kwargs)
                duration = time.perf_counter() - start

                count = len(args[0]) if isinstance(args[0], list) else 1
                setattr(stats, count_attr, getattr(stats, count_attr) + count)
                setattr(stats, time_attr, getattr(stats, time_attr) + duration)

                if op_name == "exist" and result is True:
                    stats.n_exists_true += 1
                elif op_name == "set" and result is True:
                    stats.n_sets_success += count
                elif op_name == "get":
                    stats.n_gets_success += sum(r is not None for r in result)

                return result
            finally:
                _stats_context.depth -= 1
        return wrapper
    return decorator

logger = logging.getLogger(__name__)

def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, potentially pass model and TP configs into storage backend
    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    class Statistics:
        def __init__(self):
            self.n_gets = 0
            self.t_gets = 0.0
            self.n_gets_success = 0
            self.n_sets = 0
            self.t_sets = 0.0
            self.n_sets_success = 0
            self.n_exists = 0
            self.n_exists_true = 0
            self.t_exists = 0.0

        def __str__(self):
            return (
                f"\n\n[HiCacheStorage] Statistics\n"
                f"[Gets] Count: {self.n_gets}, Avg Time: {self.t_gets / max(1, self.n_gets):.6f}s, Get Success: {self.n_gets_success}\n"
                f"[Sets] Count: {self.n_sets}, Avg Time: {self.t_sets / max(1, self.n_sets):.6f}s, Set Success: {self.n_sets_success}\n"
                f"[Exists] Count: {self.n_exists}, Avg Time: {self.t_exists / max(1, self.n_exists):.6f}s, Exists True: {self.n_exists_true}\n"
            )

    def __init_subclass__(cls):
        super().__init_subclass__()
        method_map = {
            "get": "get",
            "batch_get": "get",
            "set": "set",
            "batch_set": "set",
            "exists": "exist",
        }

        for method_name, op_name in method_map.items():
            orig = getattr(cls, method_name, None)
            if callable(orig) and not getattr(orig, "_is_wrapped", False):
                wrapped = record_stats(op_name)(orig)
                wrapped._is_wrapped = True
                setattr(cls, method_name, wrapped)
                

    def _start_stats_thread(self, interval: int = 1):
        def _loop():
            while True:
                time.sleep(interval)
                logger.info(self.statistics.__str__())
        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
    
    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool | dict:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass


class HiCacheFile(HiCacheStorage):

    def __init__(self, file_path: str = "/tmp/hicache"):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")
        self.statistics = HiCacheStorage.Statistics()
        self._start_stats_thread()

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            # Load directly into target_location's memory buffer
            with open(tensor_path, "rb") as f:
                target_location.set_(
                    torch.frombuffer(f.read(), dtype=target_location.dtype)
                    .reshape(target_location.shape)
                    .untyped_storage()
                )
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            os.remove(tensor_path)
        except FileNotFoundError:
            logger.warning(f"Key {key} does not exist. Cannot delete.")
            return

    def clear(self) -> None:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
