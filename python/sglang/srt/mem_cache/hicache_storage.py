import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from dataclasses import dataclass
import threading
import time
import rocksdb_binding as rocksdb
from sglang.srt.mem_cache.safetensor_helper import SafetensorHelper
import functools

logger = logging.getLogger(__name__)


from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()

def record_stats(op_name, is_batch=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            stats = self.statistics
            count_attr = f"n_{op_name}s"
            time_attr = f"t_{op_name}s"

            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            duration = time.perf_counter() - start

            # Determine count
            if is_batch:
                count = len(args[0]) if args else 1  # keys or inputs list
            else:
                count = 1

            if hasattr(stats, count_attr):
                setattr(stats, count_attr, getattr(stats, count_attr) + count)
            if hasattr(stats, time_attr):
                setattr(stats, time_attr, getattr(stats, time_attr) + duration)

            if op_name == "exist" and not is_batch and result is True:
                stats.n_exists_true += 1

            return result
        return wrapper
    return decorator



class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, translate tensor object access for different TP ranks
    # potentially pass model and TP configs into storage backend
    # todo, the page size of storage backend does not have to be the same as the same as host memory pool
    class Statistics:
        def __init__(self):
            self.n_gets = 0
            self.t_gets = 0.0
            self.n_sets = 0
            self.t_sets = 0.0
            self.n_exists = 0
            self.n_exists_true = 0
            self.t_exists = 0.0

        def __str__(self):
            return (
                f"\n\n[HiCacheStorage] Statistics\n"
                f"[Gets] Count: {self.n_gets}, Avg Time: {self.t_gets / max(1, self.n_gets):.6f}s\n"
                f"[Sets] Count: {self.n_sets}, Avg Time: {self.t_sets / max(1, self.n_sets):.6f}s\n"
                f"[Exists] Count: {self.n_exists}, Avg Time: {self.t_exists / max(1, self.n_exists):.6f}s, Exists True: {self.n_exists_true}\n"
            )

    def __init_subclass__(cls):
        super().__init_subclass__()
        method_map = {
            "get": ("get", False),
            "batch_get": ("get", True),
            "set": ("set", False),
            "batch_set": ("set", True),
            "exists": ("exist", False),
        }

        for method_name, (op_name, is_batch) in method_map.items():
            orig = getattr(cls, method_name, None)
            if callable(orig) and not getattr(orig, "_is_wrapped", False):
                wrapped = record_stats(op_name, is_batch)(orig)
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
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    @abstractmethod
    def batch_get(
        self, keys: List[str], target_locations: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(self, key, value) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
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
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            # todo: fixing the target_location logic to enable in-place loading
            loaded_tensor = torch.load(tensor_path)
            if isinstance(loaded_tensor, torch.Tensor):
                return loaded_tensor
            else:
                logger.error(f"Loaded data for key {key} is not a tensor.")
                return None
        except FileNotFoundError:
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(self, key: str, value: torch.Tensor) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            torch.save(value, tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
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

class HiCacheLSM(HiCacheStorage):
    def __init__(self, db_path: str = "db", tensor_filename: str = "tensor"):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        self.db_path = db_path
        self.tensor_filename = tensor_filename
        self.file_count = 0
        
        self.db = rocksdb.RocksDB()
        print(f"Opening RocksDB at '{self.db_path}'", flush=True)
        open_status = self.db.open(self.db_path)
        assert open_status
        
        self.safetensor_helper = SafetensorHelper(storage_dir = self.db_path)

        self.statistics = HiCacheStorage.Statistics()
        self._start_stats_thread()

    def _get_filename(self, fileno) -> str:
        return f"{self.tensor_filename}_{fileno}"

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix

    def _int_tobytes(self, key: List[int] | int) -> bytes:
        if isinstance(key, int):
            return key.to_bytes(4, byteorder="little", signed=False)
        return b"".join(
            k.to_bytes(4, byteorder="little", signed=False) for k in key
        )

    def _int_frombytes(self, key: bytes) -> List[int] | int:
        if len(key) % 4 != 0:
            raise ValueError("Byte length must be a multiple of 4.")
        if len(key) == 4:
            return int.from_bytes(key, byteorder="little", signed=False)
        return [
            int.from_bytes(key[i : i + 4], byteorder="little", signed=False)
            for i in range(0, len(key), 4)
        ]
        
    def db_value_tobytes(self, fileno: int, offset: int) -> bytes:
        return self._int_tobytes(fileno) + self._int_tobytes(offset)
    def db_value_frombytes(self, value: bytes) -> tuple[int, int]:
        if len(value) != 8:
            raise ValueError("Value must be 8 bytes long.")
        fileno = int.from_bytes(value[:4], byteorder="little", signed=False)
        offset = int.from_bytes(value[4:8], byteorder="little", signed=False)
        return fileno, offset

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        db_value = self.db.get(key.encode("utf-8"))
        if db_value is None:
            return None
        # byte to int
        fileno, offset = self.db_value_frombytes(db_value)
        kv_caches = self.safetensor_helper.load_kv_caches(
            self._get_filename(fileno), offsets=[offset]
        )
        kv_tensor = torch.stack(kv_caches[0], dim=0)
        return kv_tensor
        

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        db_keys = [self._get_suffixed_key(key).encode("utf-8") for key in keys]
        db_values = self.db.multiget(db_keys)

        location_dict = {}
        for i, db_value in enumerate(db_values):
            offset, idx = self.db_value_frombytes(db_value)
            if fileno not in location_dict:
                location_dict[fileno] = [[],[]]
            location_dict[fileno][0].append(offset)
            location_dict[fileno][1].append(idx)
            
        results = [None] * len(keys)
        for fileno, (offsets, idx) in location_dict.items():
            kv_caches = self.safetensor_helper.load_kv_caches(
                self._get_filename(fileno), offsets=offsets
            )
            kv_tensors = [torch.stack(kv_cache, dim=0) for kv_cache in kv_caches]
            for i, idx in enumerate(idx):
                results[idx] = kv_tensors[i]
        return results

    def set(self, key: str, value: torch.Tensor) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        key = self._get_suffixed_key(key)
        self.safetensor_helper.save_kv_caches(self._get_filename(self.file_count), [(value[0], value[1])])
        status = self.db.put(
            key.encode("utf-8"), 
            self.db_value_tobytes(self.file_count, 0)
        )
        self.file_count += 1
        return status
        

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        db_keys = [self._get_suffixed_key(key).encode("utf-8") for key in keys]
        tensors = [(value[0], value[1]) for value in values]
        db_values = [
            self.db_value_tobytes(self.file_count, i) for i in range(len(tensors))
        ]
        self.safetensor_helper.save_kv_caches(
            self._get_filename(self.file_count), db_values
        )
        self.file_count += 1
        status = self.db.batch_put(
            db_keys, db_values
        )
        return status


    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        return self.db.probe(key.encode("utf-8"))