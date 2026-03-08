# 网络结构
+ `Base3DModel`：模型的基类，包含 `__init__`、`load`、`infer_chunk`
    - `__init__()`：模型初始化
    - `load()`：加载模型参数和初始化模型实例
    - `infer_chunk()`：对输入的图像块进行推理并输出结果
+ `VGGTAdapter`：VGGT 模型的适配器，继承自 `Base3DModel`
    - `load()`：加载 VGGT 模型
    - `infer_chunk()`：使用 VGGT 模型对输入的图像块进行推理并输出结果

```python
return {
            'world_points': predictions["world_points"],
            'world_points_conf': predictions["world_points_conf"],
            'extrinsic': predictions["extrinsic"],
            'intrinsic': predictions["intrinsic"],
            'depth': predictions["depth"],
            'depth_conf': predictions["depth_conf"],
            'images': predictions["images"],
            'mask': None
        }
```

+ `Pi3Adapter`：Pi3 模型的适配器，继承自 `Base3DModel`
    - `load()`：加载 Pi3 模型
    - `infer_chunk()`：使用 Pi3 模型对输入的图像块进行推理并输出结果
```python
return {
            'world_points': predictions['points'],
            'world_points_conf': predictions['conf'],
            'extrinsic': predictions['camera_poses'],  # already C2W
            'intrinsic': None,
            'depth': None,
            'depth_conf': None,
            'images': predictions['images'],
            'mask': None
        }
```
+ `MapAnythingAdapter`：
