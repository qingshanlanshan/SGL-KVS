import shutil
import torch
import numpy as np
from safetensors.torch import save_file, safe_open
import os
from pathlib import Path

class SafetensorHelper:
    def __init__(self, storage_dir="./kv_cache_storage"):
        print(f"Initializing SafetensorHelper with storage directory: {storage_dir}")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _quantize_tensor(self, tensor, scale_factor=None):
        """将float16 tensor量化到int8"""
        if scale_factor is None:
            # 动态计算缩放因子：使用tensor的最大绝对值
            max_val = torch.abs(tensor).max().item()
            if max_val == 0:
                scale_factor = 1.0
            else:
                scale_factor = 127.0 / max_val
        
        quantized = (tensor * scale_factor).clamp(-127, 127).round().to(torch.int8)
        return quantized, scale_factor
    
    def _dequantize_tensor(self, quantized_tensor, scale_factor):
        """将int8 tensor反量化到float16"""
        return (quantized_tensor.float() / scale_factor).to(torch.float16)
    
    def _dequantize_tensor_batch(self, quantized_tensor_batch, scale_batch):
        """批量将int8 tensor反量化到float16 - 安全版本"""
        results = []
        for i in range(quantized_tensor_batch.size(0)):
            quantized_single = quantized_tensor_batch[i]
            scale_single = scale_batch[i].item()
            dequantized = self._dequantize_tensor(quantized_single, scale_single)
            results.append(dequantized)
        return results
    
    def _group_consecutive_offsets(self, offsets):
        """将连续的offsets分组，返回分组信息"""
        if not offsets:
            return []
        
        # 创建包含原始索引的元组列表，然后按offset值排序
        indexed_offsets = list(enumerate(offsets))
        indexed_offsets.sort(key=lambda x: x[1])
        
        groups = []
        current_group = []
        
        for original_idx, offset in indexed_offsets:
            if not current_group or offset == current_group[-1][1] + 1:
                # 连续的offset，加入当前组
                current_group.append((original_idx, offset))
            else:
                # 不连续，开始新组
                if current_group:
                    groups.append(current_group)
                current_group = [(original_idx, offset)]
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def save_kv_caches(self, filename, kv_caches):
        if not kv_caches:
            raise ValueError("kv_caches cannot be empty")
        
        quantized_tensors = []
        tensor_scales = []
        
        for kv_tensor in kv_caches:
            # 确保输入是float16类型的tensor
            if not isinstance(kv_tensor, torch.Tensor):
                raise ValueError("Each kv_cache item must be a torch.Tensor")
            
            # 转换为float16（如果不是的话）
            kv_tensor = kv_tensor.to(torch.float16)
            
            # 量化并保存缩放因子
            quantized, scale = self._quantize_tensor(kv_tensor)
            quantized_tensors.append(quantized)
            tensor_scales.append(scale)

        # 将所有的keys和values分别concat成大tensor
        stacked_tensors = torch.stack(quantized_tensors, dim=0)

        # 保存缩放因子
        scale_tensor = torch.tensor(tensor_scales, dtype=torch.float32)
        
        # 保存到safetensor文件
        file_path = self.storage_dir / filename
        tensors_dict = {
            "tensors": stacked_tensors,
            "scales": scale_tensor,
            "num_caches": torch.tensor(len(kv_caches), dtype=torch.int32)
        }
        
        save_file(tensors_dict, str(file_path))
        
        return {
            "filename": filename,
            "num_caches": len(kv_caches),
        }
    
    def load_kv_caches(self, filename, offsets):
        """优化版本：合并连续offsets，批量读取和解压缩"""
        file_path = self.storage_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        if not offsets:
            return []
        
        # 按原始顺序初始化结果列表
        results = [None] * len(offsets)
        
        # 将连续的offsets分组
        offset_groups = self._group_consecutive_offsets(offsets)
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            tensor_slice = f.get_slice("tensors")
            scale_tensor = f.get_tensor("scales")
            
            for group in offset_groups:
                if len(group) == 1:
                    # 单个offset，直接处理
                    original_idx, offset = group[0]
                    tensor_quantized = tensor_slice[offset]
                    scale = scale_tensor[offset].item()
                    tensor_dequantized = self._dequantize_tensor(tensor_quantized, scale)
                    results[original_idx] = tensor_dequantized
                else:
                    # 连续的offsets，批量处理
                    start_offset = group[0][1]
                    end_offset = group[-1][1]
                    
                    # 批量读取连续的tensor数据
                    tensor_quantized_batch = tensor_slice[start_offset:end_offset+1]
                    scale_batch = scale_tensor[start_offset:end_offset+1]
                    
                    # 批量反量化，返回张量列表
                    tensor_dequantized_list = self._dequantize_tensor_batch(
                        tensor_quantized_batch, scale_batch
                    )
                    
                    # 将结果分配到原始位置
                    for i, (original_idx, offset) in enumerate(group):
                        results[original_idx] = tensor_dequantized_list[i]
        
        return results
    
    def cleanup_file(self, filename):
        """删除safetensor文件"""
        file_path = self.storage_dir / filename
        if file_path.exists():
            os.remove(file_path)
            return True
        return False
    
    def list_files(self):
        """列出所有存储的文件"""
        return [f.name for f in self.storage_dir.glob("*.safetensors")]