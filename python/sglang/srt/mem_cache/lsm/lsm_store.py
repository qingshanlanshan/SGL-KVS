import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import torch
from dataclasses import dataclass
import rocksdb_binding as rocksdb
from sglang.srt.mem_cache.safetensor_helper import SafetensorHelper
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
logger = logging.getLogger(__name__)

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
        if isinstance(key, str):
            return (key + self.tp_suffix).encode("utf-8")
        elif isinstance(key, bytes):
            return key + self.tp_suffix.encode("utf-8")
        else:
            raise TypeError(f"Key must be str or bytes, got {type(key)}")
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
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        db_value = self.db.get(key)
        if db_value is None:
            return None
        # byte to int
        fileno, offset = self.db_value_frombytes(db_value)
        kv_caches = self.safetensor_helper.load_kv_caches(
            self._get_filename(fileno), offsets=[offset]
        )
        target_location.set_(
            kv_caches[0]
            .reshape(target_location.shape)
            .untyped_storage()
        )
        return target_location
        

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        db_keys = [self._get_suffixed_key(key) for key in keys]
        db_values = self.db.multiget(db_keys)

        location_dict: dict[int, list[list[int]]] = {}
        for i, (db_key, db_value) in enumerate(db_values.items()):
            fileno, offset = self.db_value_frombytes(db_value)
            if fileno not in location_dict:
                location_dict[fileno] = [[], []]
            location_dict[fileno][0].append(offset)
            location_dict[fileno][1].append(i)
        
        results = [None] * len(keys)
        for fileno, (offsets, idxs) in location_dict.items():
            kv_caches = self.safetensor_helper.load_kv_caches(
                self._get_filename(fileno), offsets=offsets
            )
            for i, idx in enumerate(idxs):
                target_locations[idx].set_(
                    kv_caches[i]
                    .reshape(target_locations[idx].shape)
                    .untyped_storage()
                )
                results[idx] = target_locations[idx]
        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        key = self._get_suffixed_key(key)
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        self.safetensor_helper.save_kv_caches(self._get_filename(self.file_count), [value])
        status = self.db.put(
            key, 
            self.db_value_tobytes(self.file_count, 0)
        )
        self.file_count += 1
        return status
        

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        db_keys = []
        tensors = []
        db_values = []
        value_offset = 0
        for i, key in enumerate(keys):
            key = self._get_suffixed_key(key)
            if self.exists(key):
                continue
            db_keys.append(key)
            tensors.append(values[i])
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
        if isinstance(key, str):
            key = self._get_suffixed_key(key)
        assert isinstance(key, bytes), "Key must be a bytes object"
        return self.db.probe(key)

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
        if isinstance(key, str):
            return (key + self.tp_suffix).encode("utf-8")
        elif isinstance(key, bytes):
            return key + self.tp_suffix.encode("utf-8")
        else:
            raise TypeError(f"Key must be str or bytes, got {type(key)}")
    
    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        compressed_value = self.db.get(key)
        if compressed_value is None:
            return None
        decompressed_tensor = self._decompress(compressed_value)
        target_location.set_(
            decompressed_tensor
            .reshape(target_location.shape)
            .untyped_storage()
        )
        return target_location
    
    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        db_keys = [self._get_suffixed_key(key) for key in keys]
        result_dict = self.db.multiget(db_keys)
        
        results = [None] * len(keys)
        for i, compressed_value in enumerate(result_dict.values()):
            if compressed_value is None:
                continue
            target_locations[i].set_(
                self._decompress(compressed_value)
                .reshape(target_locations[i].shape)
                .untyped_storage()
            )
            results[i] = target_locations[i]
        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        key = self._get_suffixed_key(key)
        if self.exists(key):
            return True
        compressed_value = self._compress(value)
        status = self.db.put(key, compressed_value)
        return status
    
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:        
        db_keys = []
        compressed_values = []
        for key, value in zip(keys, values):
            key = self._get_suffixed_key(key)
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
            key = self._get_suffixed_key(key)
        assert isinstance(key, bytes), "Key must be a bytes object"
        return self.db.probe(key)


if __name__ == "__main__":
    import random
    # Example usage
    os.system("rm -rf db/*")
    
    storage = HiCacheLSM("db")
    # storage = HiCacheBlob("db")
    
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