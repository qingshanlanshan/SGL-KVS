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
import threading

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

                return result
            finally:
                _stats_context.depth -= 1
        return wrapper
    return decorator

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
        try:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        except Exception as e:
            logger.warning(f"Failed to get tensor model parallel rank/size: {e}. Defaulting to 0 and 1.")
            tp_rank = 0
            tp_size = 1
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        self.db_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", db_path)
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
            raise ValueError(f"Value must be 8 bytes long, got {len(value)}.")
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
        for i, (db_key, db_value) in enumerate(db_values.items()):
            fileno, offset = self.db_value_frombytes(db_value)
            if fileno not in location_dict:
                location_dict[fileno] = [[],[]]
            location_dict[fileno][0].append(offset)
            location_dict[fileno][1].append(i)
            
        results = [None] * len(keys)
        for fileno, (offsets, idxs) in location_dict.items():
            kv_caches = self.safetensor_helper.load_kv_caches(
                self._get_filename(fileno), offsets=offsets
            )
            kv_tensors = [torch.stack(kv_cache, dim=0) for kv_cache in kv_caches]
            for i, idx in enumerate(idxs):
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
        db_keys = []
        tensors = []
        db_values = []
        value_offset = 0
        for i, key in enumerate(keys):
            if self.exists(key):
                continue
            db_keys.append(self._get_suffixed_key(key).encode("utf-8"))
            tensors.append((values[i][0], values[i][1]))
            db_values.append(self.db_value_tobytes(self.file_count, value_offset))
            value_offset += 1
        if len(db_keys) == 0:
            return True
        self.safetensor_helper.save_kv_caches(
            self._get_filename(self.file_count), tensors
        )
        self.file_count += 1
        status = self.db.batch_put(
            db_keys, db_values
        )
        return status


    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        return self.db.probe(key.encode("utf-8"))

class HiCacheBlob(HiCacheStorage):
    def __init__(self, db_path: str = "db", tensor_filename: str = "tensor"):
        try:
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
        except Exception as e:
            logger.warning(f"Failed to get tensor model parallel rank/size: {e}. Defaulting to 0 and 1.")
            tp_rank = 0
            tp_size = 1
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        self.db_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", db_path)
        
        self.db = rocksdb.RocksDB(blobdb=True)
        print(f"Opening RocksDB at '{self.db_path}'", flush=True)
        open_status = self.db.open(self.db_path)
        assert open_status

        self.statistics = HiCacheStorage.Statistics()
        self._start_stats_thread()
        
    def _compress(
        self,
        kv_tensor: torch.Tensor,
    ) -> bytes:
        max_val = torch.abs(kv_tensor).max().item()
        if max_val == 0:
            scale_factor = 1.0
        else:
            scale_factor = 127.0 / max_val
        quantized_tensor = (kv_tensor * scale_factor).clamp(-127, 127).round().to(torch.int8)
        quantized_bytes = quantized_tensor.cpu().contiguous().numpy().tobytes()
        scale_bytes = torch.tensor(scale_factor, dtype=torch.float16).cpu().contiguous().numpy().tobytes()
        
        return quantized_bytes + scale_bytes
    
    def _decompress(
        self,
        compressed_value: bytes,
    ):
        if compressed_value is None:
            return None
        quantized_tensor = torch.frombuffer(
            bytearray(compressed_value[:-2]),
            dtype=torch.int8,
        ).to(torch.float16)
        scale = torch.frombuffer(
            bytearray(compressed_value[-2:]),
            dtype=torch.float16,
        ).item()
        return quantized_tensor / scale

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix
    
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key).encode("utf-8")
        compressed_value = self.db.get(key)
        if compressed_value is None:
            return None
        return self._decompress(compressed_value)
    
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        db_keys = [self._get_suffixed_key(key).encode("utf-8") for key in keys]
        result_dict = self.db.multiget(db_keys)
        
        results = [self._decompress(compressed_value) for compressed_value in result_dict.values()]
        return results

    def set(self, key: str, value: torch.Tensor) -> bool:
        key = self._get_suffixed_key(key).encode("utf-8")
        if self.exists(key):
            return True
        compressed_value = self._compress(value)
        status = self.db.put(key, compressed_value)
        return status
    
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        db_keys = []
        compressed_values = []
        for key, value in zip(keys, values):
            key = self._get_suffixed_key(key).encode("utf-8")
            if self.exists(key):
                continue
            db_keys.append(key)
            compressed_values.append(self._compress(value))
        if len(db_keys) == 0:
            return True
        status = self.db.batch_put(db_keys, compressed_values)
        return status

    def exists(self, key: str) -> bool:
        if isinstance(key, str):
            key = self._get_suffixed_key(key).encode("utf-8")
        assert isinstance(key, bytes), "Key must be a bytes object"
        return self.db.probe(key)


if __name__ == "__main__":
    import random
    # Example usage
    os.system("rm -rf db/*")
    
    # storage = HiCacheLSM("db")
    storage = HiCacheBlob("db")
    
    key = "example_key"
    value = torch.rand(4, dtype=torch.float16)
    storage.set(key, value)
    
    keys = [f"key_{i}" for i in range(256)]
    values = [torch.rand(4, dtype=torch.float16) for _ in range(len(keys))]
    for i in range(len(keys)):
        storage.batch_set(keys[i:i+1], values[i:i+1])
    
    retrieved_value = storage.get(key)
    assert torch.allclose(retrieved_value, value, rtol=1e-1), "single put & get failed"
    
    assert all(
        storage.exists(k) for k in keys
    ), "some keys do not exist"
    
    retrieved_values = storage.batch_get(keys)
    assert all(
        torch.allclose(retrieved_values[i], values[i], atol=1e-1)
        for i in range(len(keys))
    ), "batch put & get failed"
    
    print(storage.statistics)
    os.system("rm -rf db")
    print("All tests passed!")