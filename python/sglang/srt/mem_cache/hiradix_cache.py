import heapq
import logging
import threading
import time
from typing import List, Optional

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.kv_storage import KVStorage

logger = logging.getLogger(__name__)


class HiRadixCache(RadixCache):

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_cache_group: torch.distributed.ProcessGroup,
        page_size: int,
        hicache_ratio: float,
        hicache_size: int,
        hicache_write_policy: str,
        hicache_io_backend: str,
        kvstore: Optional[KVStorage] = None,
    ):
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        if isinstance(self.kv_cache, MHATokenToKVPool):
            self.token_to_kv_pool_host = MHATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        elif isinstance(self.kv_cache, MLATokenToKVPool):
            self.token_to_kv_pool_host = MLATokenToKVPoolHost(
                self.kv_cache, hicache_ratio, hicache_size, page_size
            )
        else:
            raise ValueError(f"HiRadixCache only supports MHA and MLA yet")

        self.tp_group = tp_cache_group

        self.load_cache_event = threading.Event()
        self.cache_controller = HiCacheController(
            token_to_kv_pool_allocator,
            self.token_to_kv_pool_host,
            page_size,
            load_cache_event=self.load_cache_event,
            write_policy=hicache_write_policy,
            io_backend=hicache_io_backend,
        )

        # record the nodes with ongoing write through
        self.ongoing_write_through = {}
        # record the node segments with ongoing load back
        self.ongoing_load_back = {}
        # todo: dynamically adjust the threshold
        self.write_through_threshold = (
            1 if hicache_write_policy == "write_through" else 3
        )
        self.load_back_threshold = 10
        super().__init__(
            req_to_token_pool, token_to_kv_pool_allocator, page_size, disable=False, kvstore=kvstore
        )
        self.host_indices_to_kv_futures = {}

    def reset(self):
        TreeNode.counter = 0
        self.cache_controller.reset()
        self.token_to_kv_pool_host.clear()
        super().reset()

    def get_height(self, node: TreeNode):
        height = 0
        while node != self.root_node:
            node = node.parent
            height += 1
        return height

    def write_backup(self, node: TreeNode, write_back=False):
        host_indices = self.cache_controller.write(
            device_indices=node.value,
            node_id=node.id,
        )
        if host_indices is None:
            self.evict_host(len(node.value))
            host_indices = self.cache_controller.write(
                device_indices=node.value,
                node_id=node.id,
            )
        if host_indices is not None:
            node.host_value = host_indices
            self.ongoing_write_through[node.id] = node
            if not write_back:
                # no need to lock nodes if write back
                self.inc_lock_ref(node)
        else:
            return 0

        return len(host_indices)

    def inc_hit_count(self, node: TreeNode):
        if node.backuped or self.cache_controller.write_policy == "write_back":
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)
            node.hit_count = 0

    def writing_check(self, write_back=False):
        if write_back:
            # blocking till all write back complete
            while len(self.ongoing_write_through) > 0:
                ack_id = self.cache_controller.ack_write_queue.get()
                del self.ongoing_write_through[ack_id]
            return
        queue_size = torch.tensor(
            self.cache_controller.ack_write_queue.qsize(), dtype=torch.int
        )
        if torch.distributed.get_world_size(group=self.tp_group) > 1:
            # synchrnoize TP workers to make the same update to radix cache
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        for _ in range(queue_size.item()):
            ack_id = self.cache_controller.ack_write_queue.get()
            self.dec_lock_ref(self.ongoing_write_through[ack_id])
            del self.ongoing_write_through[ack_id]

    def loading_check(self):
        while not self.cache_controller.ack_load_queue.empty():
            try:
                ack_id = self.cache_controller.ack_load_queue.get_nowait()
                start_node, end_node = self.ongoing_load_back[ack_id]
                self.dec_lock_ref(end_node)
                while end_node != start_node:
                    assert end_node.loading
                    end_node.loading = False
                    end_node = end_node.parent
                # clear the reference
                del self.ongoing_load_back[ack_id]
            except Exception:
                break

    def evictable_size(self):
        return self.evictable_size_

    def _merge_kv_cache_cpu(self, kv_cache_cpu):
        # kv_cache_cpu: List[layer_num][chunk_id][2]  -> [k_cpu, v_cpu]

        layer_num = len(kv_cache_cpu)
        k_list = []
        v_list = []

        for layer in kv_cache_cpu:
            k_chunks = []
            v_chunks = []
            for k_cpu, v_cpu in layer:
                k_chunks.append(k_cpu)
                v_chunks.append(v_cpu)
            # concat chunks along sequence dimension
            k_cat = torch.cat(k_chunks, dim=0)
            v_cat = torch.cat(v_chunks, dim=0)
            k_list.append(k_cat)
            v_list.append(v_cat)

        # Stack across layer dimension -> shape: (layer_num, seq_len, ...)
        k_stack = torch.stack(k_list, dim=0)
        v_stack = torch.stack(v_list, dim=0)

        # Final shape: (2, layer_num, seq_len, ...)
        return torch.stack([k_stack, v_stack], dim=0)

    def _backup_to_disk(
        self,
        node: TreeNode,
    ):
        # if node.evicted:
        #     len_value = len(node.host_value)
        # else:
        #     len_value = len(node.value)
        # if len_value > 512:
        #     return
        prefix: List[int] = node.key
        cur_node = node
        # get the whole key
        while cur_node != self.root_node:
            cur_node = cur_node.parent
            prefix = cur_node.key + prefix
            
        # put the current node only
        if False:
            if node.host_value is not None:
                kv_tensor = self.token_to_kv_pool_host.kv_buffer[
                    :, :, node.host_value, :, :
                ]
            else:
                kv_tensor = self.token_to_kv_pool_allocator.get_cpu_copy(node.value)
                kv_tensor = self._merge_kv_cache_cpu(kv_tensor)
            self.kvstore.put_prefix_kv(prefix, kv_tensor)
            return
        else:
            max_prefix_length = self.kvstore._probe_max_prefix(
                prefix, min_length=0, max_length=len(prefix)
            )
            
            # put to db for each missing prefix
            node_list = []
            cur_node = node
            prefix_len = len(prefix)
            while cur_node != self.root_node and len(prefix) > max_prefix_length:
                node_list.insert(0, cur_node)
                prefix_len -= len(cur_node.key)
                cur_node = cur_node.parent

            for node in node_list:
                prefix_len += len(node.key)
                if node.evicted:
                    kv_tensor = self.token_to_kv_pool_host.kv_buffer[
                        :, :, node.host_value, :, :
                    ]
                else:
                    kv_tensor = self.token_to_kv_pool_allocator.get_cpu_copy(node.value)
                    kv_tensor = self._merge_kv_cache_cpu(kv_tensor)
                    
                assert kv_tensor.device == torch.device("cpu"), \
                    f"KV tensor must be on CPU, got {kv_tensor.device}"

                self.kvstore.put_prefix_kv(prefix[:prefix_len], kv_tensor)

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves_device()
        heapq.heapify(leaves)

        num_evicted = 0
        write_back_nodes = []
        while num_evicted < num_tokens and len(leaves):
            x: TreeNode = heapq.heappop(leaves)

            if x.lock_ref > 0:
                continue

            if self.kvstore:
                self._backup_to_disk(x)

            if not x.backuped:
                if self.cache_controller.write_policy == "write_back":
                    # write to host if the node is not backuped
                    num_evicted += self.write_backup(x, write_back=True)
                    write_back_nodes.append(x)
                else:
                    num_evicted += self._evict_regular(x)
            else:
                num_evicted += self._evict_backuped(x)

            for child in x.parent.children.values():
                if child in write_back_nodes:
                    continue
                if not child.evicted:
                    break
            else:
                # all children are evicted or no children
                heapq.heappush(leaves, x.parent)

        if self.cache_controller.write_policy == "write_back":
            self.writing_check(write_back=True)
            for node in write_back_nodes:
                assert node.backuped
                self._evict_backuped(node)

    def _evict_backuped(self, node: TreeNode):
        # evict a node already written to host
        num_evicted = self.cache_controller.evict_device(node.value, node.host_value)
        assert num_evicted > 0
        self.evictable_size_ -= num_evicted
        node.value = None
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        # evict a node not initiated write to host
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted

    def evict_host(self, num_tokens: int):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x : TreeNode = heapq.heappop(leaves)
            if x == self.root_node:
                break
            # only evict the host value of evicted nodes
            if not x.evicted:
                continue

            if self.kvstore and False:
                self._backup_to_disk(x)

            num_evicted += self.cache_controller.evict_host(x.host_value)

            for k, v in x.parent.children.items():
                if v == x:
                    break
            del x.parent.children[k]

            if len(x.parent.children) == 0 and x.parent.evicted:
                heapq.heappush(leaves, x.parent)

    def load_back(
        self, node: TreeNode, mem_quota: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        # todo: more loading policies

        last_hit_node = node
        nodes_to_load = []
        while node.evicted:
            assert (
                node.backuped
            ), "No backup available on evicted nodes, should not happen"
            nodes_to_load.insert(0, node)
            node = node.parent
        else:
            ancester_node = node

        # protect the ancestor nodes from eviction
        delta = self.inc_lock_ref(ancester_node)

        # load it all or not at all
        host_indices = torch.cat([n.host_value for n in nodes_to_load])
        if len(host_indices) < self.load_back_threshold or (
            len(host_indices) > mem_quota + delta if mem_quota is not None else False
        ):
            # skip loading back if the total size is too small or exceeding the memory quota
            self.dec_lock_ref(ancester_node)
            return None

        device_indices = self.cache_controller.load(
            host_indices=host_indices, node_id=last_hit_node.id
        )
        if device_indices is None:
            self.evict(len(host_indices))
            device_indices = self.cache_controller.load(
                host_indices=host_indices, node_id=last_hit_node.id
            )
        self.dec_lock_ref(ancester_node)
        if device_indices is None:
            # no sufficient GPU memory to load back KV caches
            return None

        self.ongoing_load_back[last_hit_node.id] = (ancester_node, last_hit_node)
        offset = 0
        for node in nodes_to_load:
            node.value = device_indices[offset : offset + len(node.host_value)]
            offset += len(node.host_value)
            node.loading = True
        self.evictable_size_ += len(device_indices)
        self.inc_lock_ref(last_hit_node)

        return device_indices

    def _load_disk_to_cpu(
        self,
        node: TreeNode,
    ):
        assert node.evicted, "Node must be evicted to load from disk"
        for indice in node.host_value:
            indice = int(indice.item())
            kv_future, index = self.host_indices_to_kv_futures.pop(indice, (None, None))
            if kv_future is None:
                return
            kv_tensor = self.kvstore.wait_for_kv(kv_future)
            self.token_to_kv_pool_host.kv_buffer[:, :, indice, :, :] = kv_tensor[
                :, :, index, :, :
            ].cpu()

    def init_load_back(
        self,
        last_node: TreeNode,
        host_hit_length: int,
        mem_quota: Optional[int] = None,
    ):
        _ = host_hit_length  # unused, but kept for compatibility
        if last_node.evicted:
            if not self.kvstore:
                loading_values = self.load_back(last_node, mem_quota)
            else:
                # gpu nodes - cpu nodes - (last_cpu_node) - disk nodes - (last_node)
                loading_values = None
                last_cpu_node = last_node
                while last_cpu_node.evicted and int(last_cpu_node.host_value[0]) in self.host_indices_to_kv_futures:
                    last_cpu_node = last_cpu_node.parent

                # load cpu nodes
                if last_cpu_node.evicted:
                    loading_values = self.load_back(last_cpu_node, mem_quota)

                if last_cpu_node.evicted:
                    # no sufficient GPU memory to load back KV caches
                    assert loading_values is None, "Loading values should be None if loading back failed"
                elif last_cpu_node.id != last_node.id:
                    # load disk to cpu
                    disk_node = last_node
                    while disk_node != last_cpu_node:
                        self._load_disk_to_cpu(disk_node)
                        disk_node = disk_node.parent
                    # load cpu to gpu
                    new_values = self.load_back(last_node, mem_quota)
                    if new_values is None:
                        last_node = last_cpu_node
                    if loading_values is None:
                        loading_values = new_values
                    elif new_values is not None:
                        loading_values = torch.cat([loading_values, new_values]) 

            if loading_values is not None:
                logger.debug(
                    f"loading back {len(loading_values)} tokens for node {last_node.id}"
                )
                assert not last_node.evicted, \
                    f"Node {last_node.id} should not be evicted after loading back"
                return loading_values, last_node

            while last_node.evicted:
                last_node = last_node.parent

        return (
            torch.empty((0,), dtype=torch.int64, device=self.device),
            last_node,
        )

    def ready_to_load_host_cache(self):
        producer_index = self.cache_controller.layer_done_counter.next_producer()
        self.load_cache_event.set()
        return producer_index

    def check_hicache_events(self):
        self.writing_check()
        self.loading_check()

    def match_prefix(self, key: List[int], **kwargs):
        empty_value = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=empty_value,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = empty_value

        # fetch kv tensor from disk to host
        if self.kvstore:
            host_hit_length = 0
            last_host_node = last_node
            while last_node.evicted:
                host_hit_length += len(last_node.host_value)
                last_node = last_node.parent

            tree_hit_length = host_hit_length + value.shape[0]
            last_node = last_host_node

            # get kv tensor
            disk_match_length, kv_future = self.kvstore.get_prefix_kv(
                key,
                tree_hit_length,
                len(key),
            )
            # insert into radix tree
            if disk_match_length > tree_hit_length:
                # allocate memory
                need_size = disk_match_length - tree_hit_length
                kv_indices = self.token_to_kv_pool_host.alloc(need_size)
                if kv_indices is None:
                    self.evict_host(need_size)
                    kv_indices = self.token_to_kv_pool_host.alloc(need_size)

                assert kv_indices is not None, \
                    f"Failed to allocate {need_size} indices from host kv pool"
                assert len(kv_indices) == need_size, \
                    f"Allocated {len(kv_indices)} indices, expected {need_size}"
                assert kv_future is not None, \
                    f"KV future for key {key} not found in kvstore"

                if kv_indices is not None:
                    for i, indice in enumerate(kv_indices):
                        indice = int(indice.item())
                        self.host_indices_to_kv_futures[indice] = (kv_future, i)
                    child_key = key[tree_hit_length:disk_match_length]
                    new_node = TreeNode()
                    new_node.parent = last_node
                    new_node.key = child_key
                    new_node.value = None
                    new_node.host_value = kv_indices
                    last_node.children[self.get_child_key_fn(child_key)] = new_node
                    # self.evictable_size_ += len(kv_indices)
                    if self.cache_controller.write_policy != "write_back":
                        self.inc_hit_count(new_node)
                    last_node = new_node

        host_hit_length = 0
        last_host_node = last_node
        while last_node.evicted:
            host_hit_length += len(last_node.host_value)
            last_node = last_node.parent

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_host_node,
            host_hit_length=host_hit_length,
        )

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                self.inc_hit_count(new_node)
                if not new_node.evicted:
                    value.append(new_node.value)
                node = new_node
                break
            else:
                self.inc_hit_count(child)
                if not child.evicted:
                    value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # child node split into new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.loading = child.loading
        new_node.hit_count = child.hit_count

        # split value and host value if exists
        if child.evicted:
            new_node.value = None
        else:
            new_node.value = child.value[:split_len]
            child.value = child.value[split_len:]
        if child.backuped:
            new_node.host_value = child.host_value[:split_len]
            child.host_value = child.host_value[split_len:]
        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)

            if prefix_len == len(node.key):
                if node.evicted:
                    # change the reference if the node is evicted
                    # this often happens in the case of KV cache recomputation
                    node.value = value[:prefix_len]
                    self.token_to_kv_pool_host.update_synced(node.host_value)
                    self.evictable_size_ += len(node.value)
                else:
                    self.inc_hit_count(node)
                    total_prefix_length += prefix_len
            else:
                # partial match, split the node
                new_node = self._split_node(node.key, node, prefix_len)
                if new_node.evicted:
                    new_node.value = value[:prefix_len]
                    self.token_to_kv_pool_host.update_synced(new_node.host_value)
                    self.evictable_size_ += len(new_node.value)
                else:
                    self.inc_hit_count(new_node)
                    total_prefix_length += prefix_len
                node = new_node

            key = key[prefix_len:]
            value = value[prefix_len:]

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            if self.cache_controller.write_policy != "write_back":
                self.inc_hit_count(new_node)
        return total_prefix_length

    def _collect_leaves_device(self):
        def is_leaf(node):
            if node.evicted:
                return False
            if node == self.root_node:
                return False
            if len(node.children) == 0:
                return True
            for child in node.children.values():
                if not child.evicted:
                    return False
            return True

        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            if is_leaf(cur_node):
                ret_list.append(cur_node)
            else:
                for cur_child in cur_node.children.values():
                    if not cur_child.evicted:
                        stack.append(cur_child)
        return ret_list
