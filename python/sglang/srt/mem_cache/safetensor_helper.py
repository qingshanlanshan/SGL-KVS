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
        file_path = self.storage_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        results = []
        
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            # 获取tensor slices和缩放因子
            tensor_slice = f.get_slice("tensors")
            scale_tensor = f.get_tensor("scales")
            
            # 按offsets读取数据
            for offset in offsets:
                # 读取第offset个cache
                tensor_quantized = tensor_slice[offset]
                
                # 获取对应的缩放因子
                scale = scale_tensor[offset].item()
                
                # 反量化
                tensor_dequantized = self._dequantize_tensor(tensor_quantized, scale)

                results.append(tensor_dequantized)

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
