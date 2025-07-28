import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import rocksdb_binding as rocksdb
from safetensor_helper import SafetensorHelper

import torch

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

    @abstractmethod
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        raise NotImplementedError("[HiCacheStorage] get method not implemented")

    @abstractmethod
    def batch_get(
        self, keys: List[str], target_locations: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        raise NotImplementedError("[HiCacheStorage] batch_get method not implemented")

    @abstractmethod
    def set(self, key, value) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        raise NotImplementedError("[HiCacheStorage] set method not implemented")

    @abstractmethod
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        raise NotImplementedError("[HiCacheStorage] batch_set method not implemented")

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        raise NotImplementedError("[HiCacheStorage] exists method not implemented")


class HiCacheFile(HiCacheStorage):

    def __init__(self, file_path: str = "/tmp/hicache"):
        self.file_path = file_path
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 else ""
        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        print(f"[HiCacheFile] get key: {key}, target_location: {target_location}")
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
        
        
        self.db = rocksdb.RocksDB()
        print(f"Opening RocksDB at '{self.db_path}'")
        open_status = self.db.open(self.db_path)
        assert open_status
        
        self.safetensor_helper = SafetensorHelper(self.db_path)
        self.file_offset = 0

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix
    
    def str_tobytes(self, key: str) -> bytes:
        if isinstance(key, str):
            return key.encode("utf-8")
        raise TypeError("Key must be a string.")

    def int_tobytes(self, key: List[int] | int) -> bytes:
        if isinstance(key, list):
            assert all(
                isinstance(k, int) for k in key
            ), "All elements in the list must be integers."
            return b"".join(
                k.to_bytes(4, byteorder="little", signed=False) for k in key
            )
        assert isinstance(key, int), "Key must be an integer or a list of integers."
        return key.to_bytes(4, byteorder="little", signed=False)

    def int_frombytes(self, key: bytes) -> List[int]:
        if len(key) % 4 != 0:
            raise ValueError("Byte length must be a multiple of 4.")
        return [
            int.from_bytes(key[i : i + 4], byteorder="little", signed=False)
            for i in range(0, len(key), 4)
        ]

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        print(f"[HiCacheLSM] get key: {key}, target_location: {target_location}")
        key = self._get_suffixed_key(key)
        key = self.str_tobytes(key)
        offset = self.db.get(key)
        if offset is None:
            return None
        # byte to int
        offsets = self.int_frombytes(offset)
        kv_caches = self.safetensor_helper.load_kv_caches(self.tensor_filename, offsets)
        kv_tensor = torch.stack(kv_caches[0], dim=0)
        return kv_tensor
        

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
        key = self.str_tobytes(key)
        self.safetensor_helper.save_kv_caches(self.tensor_filename, [(value[0], value[1])])
        status = self.db.put(key, self.int_tobytes(self.file_offset))
        self.file_offset += 1
        return status
        

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        key = self.int_tobytes(key)
        return self.db.probe(key)
