import struct
import torch
import numpy as np
import rocksdb_binding as rocksdb
from typing import Tuple, Dict, Optional, List
import os
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
from typing import Optional
import threading
from dataclasses import dataclass, field
import time
import queue

class KVStorage:
    _instance = None
    _lock = threading.Lock()

    @dataclass
    class Statistics:
        n_prefix_gets: int = 0
        t_prefix_get: float = 0.0 

        n_prefix_puts: int = 0
        t_prefix_put: float = 0.0   

        n_wait_for_kv: int = 0
        t_wait_for_kv: float = 0.0 

        n_executor_gets: int = 0
        t_executor_get: float = 0.0

        n_db_probes: int = 0
        t_db_probe: float = 0.0

        n_db_puts: int = 0
        t_db_put: float = 0.0

    def __init__(
        self,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        executor_worker_num: int = 16,
        db_path: str = "db",
        compress: bool = True,
    ):
        print(f"[KVStorage::__init__] Initializing KVStorage with dtype={dtype}, kvtensor shape=({2}, {layer_num}, seq_len, {head_num}, {head_dim})")

        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.db_path = os.path.expanduser(db_path)
        self.do_compress = compress

        self.db = rocksdb.RocksDB()
        print(f"Opening RocksDB at '{self.db_path}' with compression={self.do_compress}")
        open_status = self.db.open(self.db_path)
        assert open_status

        self.executor = ThreadPoolExecutor(max_workers=executor_worker_num)
        self.db_put_queue: queue.Queue[Tuple[List[int], torch.Tensor]] = queue.Queue(maxsize=20)
        threading.Thread(
            target=self._db_put_worker, daemon=True, name="DB Put Worker"
        ).start()
        self.statistics = self.Statistics()

    def statistics_str(self):
        return (
            f"[KVStorage] Statistics:\n"
            f"[Put] Prefix put count: {self.statistics.n_prefix_puts}, "
            f"Avg prefix put time: {self.statistics.t_prefix_put / max(1, self.statistics.n_prefix_puts):.6f} seconds\n"
            f"[Get] Prefix get count: {self.statistics.n_prefix_gets}, "
            f"Avg prefix get time: {self.statistics.t_prefix_get / max(1, self.statistics.n_prefix_gets):.6f} seconds\n"
            f"[Wait] Wait for KV count: {self.statistics.n_wait_for_kv}, "
            f"Avg wait time: {self.statistics.t_wait_for_kv / max(1, self.statistics.n_wait_for_kv):.6f} seconds\n"
            f"[Probe] DB probe count: {self.statistics.n_db_probes}, "
            f"Avg probe time: {self.statistics.t_db_probe / max(1, self.statistics.n_db_probes):.6f} seconds\n"
            f"[DB Put] Count: {self.statistics.n_db_puts}, "
            f"Avg time: {self.statistics.t_db_put / max(1, self.statistics.n_db_puts):.6f} seconds\n"
            f"[Executer Get] Count: {self.statistics.n_executor_gets}, "
            f"Avg time: {self.statistics.t_executor_get / max(1, self.statistics.n_executor_gets):.6f} seconds\n"
        )

    def _make_key(self, key: List[int]) -> bytes:
        assert isinstance(key, list), "Key must be a list of integers"
        assert isinstance(key[0], int), "List keys must contain integers"
        return np.array(key, dtype=np.int32).tobytes()

    def put_prefix_kv(
        self, 
        key: List[int],
        # shape: [2, layer_num, pre_len, head_num, head_dim]
        kv_tensor: torch.Tensor,
        block: bool = False,
    ):
        self.statistics.n_prefix_puts += 1
        start = time.perf_counter()

        required_shape= (2, self.layer_num, len(key), self.head_num, self.head_dim)
        assert (
            kv_tensor.shape[:2] == required_shape[:2]
            and kv_tensor.shape[2] <= len(key)
            and kv_tensor.shape[3:] == required_shape[3:]
        ), f"kv_tensor shape {kv_tensor.shape} does not match required shape {required_shape}"

        if self.dtype not in [torch.float16, torch.float32, torch.float64]:
            kv_tensor = kv_tensor.to(torch.float32)
        if block:
            self._rocksdb_put(key, kv_tensor)
        else:
            try:
                self.db_put_queue.put((key, kv_tensor), block=False)
            except queue.Full:
                pass
        end = time.perf_counter()   
        self.statistics.t_prefix_put += (end - start)
        

    def _db_put_worker(self):
        while True:
            key, kv_tensor = self.db_put_queue.get()
            self.statistics.n_db_puts += 1
            start = time.perf_counter()

            key_bytes = self._make_key(key)
            if kv_tensor.shape[2] > 1:
                exist_key_len = self._probe_max_prefix(
                    key_bytes, min_length=0, max_length=len(key)
                )
            else:
                # if the kv_tensor has only one element, force put
                exist_key_len = 0
            if exist_key_len == len(key):
                continue

            # need to put [max(exist_key_len, len(key) - kv_tensor.shape[2]) : len(key)]
            start_put_idx = max(exist_key_len, len(key) - kv_tensor.shape[2])

            db_keys = [key_bytes[: (L + 1) * 4] for L in range(start_put_idx, len(key))]

            if self.do_compress:
                db_values = [
                    self.compress(
                        kv_tensor[:, :, L - (len(key) - kv_tensor.shape[2]), :, :]
                    )
                    for L in range(start_put_idx, len(key))
                ]
            else:
                db_values = [
                    kv_tensor[:, :, L - (len(key) - kv_tensor.shape[2]), :, :]
                    .cpu()
                    .contiguous()
                    .numpy()
                    .data.tobytes()
                    for L in range(start_put_idx, len(key))
                ]
            assert len(db_keys) == len(db_values), \
                f"Length mismatch: {len(db_keys)=} != {len(db_values)=}"
            assert all(db_value is not None for db_value in db_values), \
                "All db_values must be non-None"
            status = self.db.batch_put(db_keys, db_values)
            assert status, "Batch put failed"
            end = time.perf_counter()
            self.statistics.t_db_put += end - start

    def compress(
        self,
        kv_tensor: torch.Tensor,  # shape: [2, layer_num, head_num, head_dim]
        num_bits: int = 8,
    ) -> bytes:
        data = kv_tensor.cpu().contiguous()
        group_dim = 1
        B: int = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim, keepdim=True)[0]
        scale = B / (mx - mn + 1e-8)
        data = (data - mn) * scale
        data = data.clamp(0, B).round().to(torch.uint8)
        # pack data, mn, scale
        data = data.cpu().contiguous().numpy().data.tobytes()
        mn = mn.cpu().contiguous().numpy().data.tobytes()
        scale = scale.cpu().contiguous().numpy().data.tobytes()
        return data + mn + scale

    def decompress(
        self,
        compressed_value: bytes,
    ):
        # compute sizes
        shape_data = torch.Size((2, self.layer_num, self.head_num, self.head_dim))
        shape_mn_scale = torch.Size((2, 1, self.head_num, self.head_dim))

        data = torch.frombuffer(
            bytearray(compressed_value), 
            count=shape_data.numel(), 
            dtype=torch.uint8,
        ).reshape(shape_data)
        mn = torch.frombuffer(
            bytearray(compressed_value),
            count=shape_mn_scale.numel(),
            offset=shape_data.numel() * torch.uint8.itemsize,
            dtype=self.dtype,
        ).reshape(shape_mn_scale)
        scale = torch.frombuffer(
            bytearray(compressed_value), 
            count=shape_mn_scale.numel(),
            offset=shape_data.numel() * torch.uint8.itemsize + shape_mn_scale.numel() * self.dtype.itemsize,
            dtype=self.dtype,
        ).reshape(shape_mn_scale)

        # dequantize
        tensor = data.to(self.dtype) / scale + mn
        return tensor

    def _probe_max_prefix(
        self,
        key: List[int] | bytes,
        min_length: int,
        max_length: int
    ) -> int:
        start = time.perf_counter()
        matched_pre_len = min_length
        if isinstance(key, list):
            key = self._make_key(key)
        if False:
            # binary search for the longest prefix
            low, high = min_length, max_length
            while low < high:
                mid = (low + high + 1) // 2
                db_key = key[:mid * 4]
                exist = self.db.probe(db_key)
                self.statistics.n_db_probes += 1
                if exist:
                    matched_pre_len = mid
                    low = mid
                else:
                    high = mid - 1
        else:
            for pre_len in range(max_length, min_length, -1):
                db_key = key[:pre_len * 4]  # Each int32 is 4 bytes
                exist = self.db.probe(db_key)
                self.statistics.n_db_probes += 1
                if exist:
                    matched_pre_len = pre_len
                    break
        end = time.perf_counter()
        self.statistics.t_db_probe += (end - start)
        return matched_pre_len

    def get_prefix_kv(
        self, 
        key: list[int], 
        min_length: int,
        max_length: int
    ) -> Tuple[int, Optional[Future]]:
        assert min_length <= max_length <= len(key), \
            f"Invalid lengths: {min_length=} {max_length=} {len(key)=}"
        if min_length == max_length:
            return min_length, None
        self.statistics.n_prefix_gets += 1
        start = time.perf_counter()
        matched_pre_len = self._probe_max_prefix(
            key,
            min_length=min_length,
            max_length=max_length
        )
        print(f"[KVStorage::get_prefix_kv] Matched prefix length: {matched_pre_len} for {len(key)=}[{min_length}:{max_length}]")
        kv_future: Optional[Future] = None
        if matched_pre_len > min_length:
            matched_key = key[:matched_pre_len]
            # issue a worker thread to perform _rocksdb_get
            # the return value serves prefix [min_length, matched_pre_len]
            kv_future: Future = self.executor.submit(
                self._rocksdb_get,
                matched_key,
                min_length,
            )
        end = time.perf_counter()
        self.statistics.t_prefix_get += (end - start)
        return matched_pre_len, kv_future

    def _rocksdb_put(
        self,
        key: List[int],
        kv_tensor: torch.Tensor,
    ):
        kv_tensor = kv_tensor.cpu().numpy()
        exist_key_len = self._probe_max_prefix(
            key,
            min_length=0,
            max_length=len(key)
        ) 
        for L in range(exist_key_len, len(key)):
            prefix_ids = key[: L + 1]  # Prefix of length L
            prefix_tensor = kv_tensor[:, :, L, :, :]
            db_key = self._make_key(prefix_ids)
            value = prefix_tensor.cpu().numpy().tobytes()
            put_status = self.db.put(db_key, value)
            assert put_status

    def _rocksdb_get(
        self,
        matched_key: List[int],
        min_length: int,
    ) -> torch.Tensor:
        matched_prefix_length = self._probe_max_prefix(
            matched_key,
            min_length=min_length,
            max_length=len(matched_key)
        )
        assert matched_prefix_length == len(matched_key), \
            f"Matched prefix length {matched_prefix_length} does not match key length {len(matched_key)}"
        self.statistics.n_executor_gets += 1
        start = time.perf_counter()
        dtype = self.dtype if self.dtype in [torch.float16, torch.float32, torch.float64] else torch.float32
        matched_key_bytes = self._make_key(matched_key)
        db_keys = [matched_key_bytes[:(L + 1) * 4] for L in range(min_length, len(matched_key))]
        kv_cpu_raws = self.db.multiget(db_keys)
        kv_tensor = torch.stack(
            [
                self.decompress(kv_cpu_raws[db_key]) if self.do_compress else
                torch.frombuffer(
                    bytearray(kv_cpu_raws[db_key]),
                    dtype=dtype,
                    count=2 * self.layer_num * self.head_num * self.head_dim,
                ).reshape(2, self.layer_num, self.head_num, self.head_dim)
                for db_key in db_keys
            ],
            dim=2,
        ).to(self.dtype)
        end = time.perf_counter()
        self.statistics.t_executor_get += (end - start)
        return kv_tensor

    def wait_for_kv(
        self,
        kv_future: Future,
        timeout: Optional[float] = None,  # seconds; None = wait forever
    ) -> torch.Tensor:
        self.statistics.n_wait_for_kv += 1
        start = time.perf_counter()
        kv_tensor : torch.Tensor = kv_future.result(timeout=timeout)
        kv_tensor = kv_tensor.cuda()
        required_shape = (2, self.layer_num, kv_tensor.shape[2], self.head_num, self.head_dim)
        assert kv_tensor.shape == required_shape, f"{kv_tensor.shape=} does not match {required_shape=}"
        end = time.perf_counter()
        self.statistics.t_wait_for_kv += (end - start)
        return kv_tensor


if __name__ == "__main__":
    if os.path.exists(os.path.expanduser("~/test_db")):
        print("Deleting existing RocksDB at ~/test_db")
        os.system("rm -rf ~/test_db")
    head_num = 2
    head_dim = 4
    layer_num = 8
    kvs = KVStorage(
        dtype=torch.float16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        executor_worker_num=4,
        db_path="~/test_db",
        compress=False
    )

    key = list(range(4096))
    kv_tensor = torch.arange(
        2 * layer_num * len(key) * head_num * head_dim,
        dtype=torch.float16,
    ).reshape(2, layer_num, len(key), head_num, head_dim)
    kvs.put_prefix_kv(key, kv_tensor)
    time.sleep(2)
    print(f"Stored kv_tensor with shape {kv_tensor.shape} for {len(key)=}")
    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key, 
        min_length=0, 
        max_length=len(key)
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)

    assert matched_pre_len == len(key), "Matched prefix length does not match the original key length"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()), "Fetched kv_tensor does not match the original kv_tensor"

    matched_pre_len, kv_future = kvs.get_prefix_kv(
        key[:3], 
        min_length=0, 
        max_length=3
    )
    fetched_kv_tensor = kvs.wait_for_kv(kv_future)
    assert matched_pre_len == 3, "Matched prefix length should be 3 for key [1, 2, 3]"
    assert torch.equal(fetched_kv_tensor.cpu(), kv_tensor.cpu()[:, :, :3, :, :]), "Fetched kv_tensor does not match the original kv_tensor for prefix [1, 2, 3]"

    print("=" * 40)
    print("KVStorage test passed successfully!")
    print(kvs.statistics_str())
