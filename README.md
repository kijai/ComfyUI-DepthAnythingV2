# ComfyUI nodes to use DepthAnythingV2
https://depth-anything-v2.github.io

Models download from https://huggingface.co/Kijai/DepthAnythingV2-safetensors/tree/main

Set a shared folder in ComfyUI's `extra_model_paths.yaml`:

```yaml
shared_models:
  is_default: true
  depthanything: C:/AI/models/depthanything
```

The loader searches registered folders before downloading to the first path. `is_default: true` puts the shared folder first. `ComfyUI/models/depthanything` remains a fallback.

![image](https://github.com/kijai/ComfyUI-DepthAnythingV2/assets/40791699/6e9f190a-02da-4176-9cca-6e4e4b0ad35d)
