# AnimeGAN_PaddleInference
## 简介
* 使用 PaddleInference 推理引擎实现 AnimeGAN V1 和 AnimeGAN V2 图像动漫化模型的模型推理
* 并封装为 PaddleHub Module 供使用者进行快速调用
* 原项目：[AnimeGAN V1](https://github.com/TachibanaYoshino/AnimeGAN)、[AnimeGAN V2](https://github.com/TachibanaYoshino/AnimeGANv2)
* 本项目使用到的所有模型均转换至官方项目提供的预训练模型
* 感谢官方项目的开源代码和模型
## 效果展示
* 输入图像：

![输入图像](https://ai-studio-static-online.cdn.bcebos.com/fff48f6cd38e41f5bc931d86d08143c835ef827866364fc587b6d3b4fc7dcc98)

* 输出图像：

![输出图像](https://ai-studio-static-online.cdn.bcebos.com/f8c1acba47a74f23b3fb9bac43a6d91662785872c08148d5a90f22c911a919c7)
## 模型列表
| Model | Style | Epoch | PaddleHub Module |
| -------- | -------- | -------- | -------- |
| AnimeGAN v1 | Hayao | 60 | animegan_v1_hayao_60 |
| AnimeGAN V2 | Hayao | 64 | animegan_v2_hayao_64 |
| AnimeGAN V2 | Hayao | 99 | animegan_v2_hayao_99 |
| AnimeGAN V2 | Paprika | 54 | animegan_v2_paprika_54 |
| AnimeGAN V2 | Paprika | 74 | animegan_v2_paprika_74 |
| AnimeGAN V2 | Paprika | 97 | animegan_v2_paprika_97 |
| AnimeGAN V2 | Paprika | 98 | animegan_v2_paprika_98 |
| AnimeGAN V2 | Shinkai | 33 | animegan_v2_shinkai_33 |
| AnimeGAN V2 | Shinkai | 53 | animegan_v2_shinkai_53 |
## AIStudio 在线体验
* [PaddleInference：风景图像动漫化模型AnimeGAN的推理部署](https://aistudio.baidu.com/aistudio/projectdetail/1201335)
## 使用 PaddleHub 进行快速调用
* 安装 PaddlePaddle >= 2.0.0rc0 ：请参考官网的[安装教程](https://www.paddlepaddle.org.cn)
* 安装 PaddleHub ：
```shell
$ pip install paddlehub
```  
* 快速调用：
```python
import cv2
import paddlehub as hub

# 模型加载
# use_gpu：是否使用GPU进行预测
# name: 模型名称，参考上面的模型列表
model = hub.Module(name='animegan_v2_shinkai_33', use_gpu=False)

# 模型预测
result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])

# or
# result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
```
* API 说明：
```python
def style_transfer(
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False,
    min_size=32,
    max_size=1024
)
```
> 参数：  
> images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]，默认为 None  
> paths (list[str]): 图片的路径，默认为 None   
> batch_size (int): batch 的大小，默认设为 1   
> visualization (bool): 是否将识别结果保存为图片文件，默认设为 False   
> output_dir (str): 图片的保存路径，默认设为 output   
> min_size (int): 输入图片的短边最小尺寸，默认设为 32   
> max_size (int): 输入图片的短边最大尺寸，默认设为 1024  
## 基于 PaddleInference 源码使用
* 克隆项目：
```shell
$ git clone https://github.com/jm12138/AnimeGAN_PaddleInference
$ cd AnimeGAN_PaddleInference
```
* 安装依赖：
```shell
$ pip install -r requirements.txt
```
* 安装 PaddlePaddle >= 2.0.0rc0 ：请参考官网的[安装教程](https://www.paddlepaddle.org.cn)
* 下载预训练模型：[下载链接](https://bj.bcebos.com/v1/ai-studio-online/6f827f241bc14536b335a3f3b5c1ed952618faee9a794348b61e03489271fbb7?responseContentDisposition=attachment%3B%20filename%3DAnimeGAN.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-11-07T15%3A56%3A54Z%2F-1%2F%2Ff24e418e4b134203e8665ba1db53239bb20ba133dce283af640439aed3bf5825)
* 解压预训练模型至项目目录
* 运行图像动漫化主程序：
```shell
$ python main.py \
    --img_path test.jpg \
    --save_path save_imgs \
    --use_gpu True \
    --max_size 512 \
    --min_size 32
```
> 参数：    
> img_path：输入图像路径  
> save_path：输出图像保存目录  
> use_gpu：是否使用GPU进行推理，最好使用GPU进行推理  
> max_size：模型输入的最大尺寸，尺寸越大消耗的内存/显存会越大  
> min_size：模型输入的最小尺寸  
