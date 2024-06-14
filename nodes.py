
import torch
import torch.nn.functional as F
from torchvision import transforms
from huggingface_hub import hf_hub_download
import os
from contextlib import nullcontext

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

from .depth_anything_v2.dpt import DepthAnythingV2

class DownloadAndLoadDepthAnythingV2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                        'depth_anything_v2_vits_fp16.safetensors',
                        'depth_anything_v2_vits_fp32.safetensors',
                        'depth_anything_v2_vitb_fp16.safetensors',
                        'depth_anything_v2_vitb_fp32.safetensors',
                        'depth_anything_v2_vitl_fp16.safetensors',
                        'depth_anything_v2_vitl_fp32.safetensors',
                    ],
                    {
                    "default": 'depth_anything_v2_vitl_fp32.safetensors'
                    }),
            },
        }

    RETURN_TYPES = ("DAMODEL",)
    RETURN_NAMES = ("da_v2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = """
Models autodownload to `ComfyUI\models\depthanything` from   
https://huggingface.co/Kijai/DepthAnythingV2-safetensors/tree/main   
   
fp16 reduces quality by a LOT, not recommended.
"""

    def loadmodel(self, model):
        device = mm.get_torch_device()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            #'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        custom_config = {
            'model_name': model,
        }
        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            download_path = os.path.join(folder_paths.models_dir, "depthanything")
            model_path = os.path.join(download_path, model)

            if not os.path.exists(model_path):
                print(f"Downloading model to: {model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/DepthAnythingV2-safetensors", 
                                  allow_patterns=[f"*{model}*"],
                                  local_dir=download_path, 
                                  local_dir_use_symlinks=False)

            print(f"Loading model from: {model_path}")

            if "vitl" in model:
                encoder = "vitl"
            elif "vitb" in model:
                encoder = "vitb"
            elif "vits" in model:
                encoder = "vits"

            self.model = DepthAnythingV2(**model_configs[encoder])
            
            state_dict = load_torch_file(model_path)
            self.model.load_state_dict(state_dict)
            dtype = torch.float16 if "fp16" in model else torch.float32

            self.model = self.model.to(dtype).to(device).eval()
            da_model = {
                "model": self.model,
                "dtype": dtype
            }
           
        return (da_model,)
    
class DepthAnything_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "da_model": ("DAMODEL", ),
            "images": ("IMAGE", ),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = """
https://depth-anything-v2.github.io
"""

    def process(self, da_model, images):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da_model['model']
        dtype=da_model['dtype']
        
        B, H, W, C = images.shape

        images = images.to(device)
        images = images.permute(0, 3, 1, 2)

        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images = F.interpolate(images, size=(H, W), mode="bicubic")
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images)
        pbar = ProgressBar(B)
        out = []
        model.to(device)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for img in normalized_images:
                depth = model.infer_image(img.unsqueeze(0))
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                out.append(depth)
                pbar.update(1)
            model.to(offload_device)
            depth_out = torch.cat(out, dim=0)
            depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()
        
        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
            depth_out = F.interpolate(depth_out.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bicubic").permute(0, 2, 3, 1)
        return (depth_out,)
    
NODE_CLASS_MAPPINGS = {
    "DepthAnything_V2": DepthAnything_V2,
    "DownloadAndLoadDepthAnythingV2Model": DownloadAndLoadDepthAnythingV2Model
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnything_V2": "Depth Anything V2",
    "DownloadAndLoadDepthAnythingV2Model": "DownloadAndLoadDepthAnythingV2Model"
}