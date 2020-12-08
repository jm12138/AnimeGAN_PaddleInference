import os
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from paddle.inference import Config
from paddle.inference import create_predictor

def load_model(modelpath, use_gpu):
    # 加载模型参数
    config = Config(modelpath)

    # 设置参数
    if use_gpu:   
        config.enable_use_gpu(100, 0)
    else:
        config.disable_gpu()
        config.enable_mkldnn()
    config.disable_glog_info()
    config.switch_ir_optim(True)
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)

    # 通过参数加载模型预测器
    predictor = create_predictor(config)

    # 获取模型的输入输出
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handle = predictor.get_input_handle(input_names[0])
    output_handle = predictor.get_output_handle(output_names[0])
    
    # 返回模型预测器和输入输出
    return predictor, input_handle, output_handle


def preprocess(img_path, max_size=512, min_size=32):
    # 读取图片
    img = cv2.imread(img_path)

    # 格式转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 缩放图片
    h, w = img.shape[:2]
    if max(h,w)>max_size:
        img = cv2.resize(img, (max_size, int(h/w*max_size))) if h<w else cv2.resize(img, (int(w/h*max_size), max_size))
    elif min(h,w)<min_size:
        img = cv2.resize(img, (min_size, int(h/w*min_size))) if h>w else cv2.resize(img, (int(w/h*min_size), min_size))

    # 裁剪图片
    h, w = img.shape[:2]
    img = img[:h-(h%32), :w-(w%32), :]

    # 归一化
    img = img/127.5 - 1.0

    # 新建维度
    img = np.expand_dims(img, axis=0).astype('float32')
    
    # 返回输入数据
    return img


def postprocess(outputs, output_dir, pretrained_models):
    for im_id, output in enumerate(outputs):
        # 反归一化
        image = (output.squeeze() + 1.) / 2 * 255

        # 限幅
        image = np.clip(image, 0, 255).astype(np.uint8)

        # 格式转换
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 检查输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入输出图片
        cv2.imwrite(os.path.join(output_dir, '%s.jpg' % pretrained_models[im_id]), image)

def main(args):
    pretrained_models = [
        'animegan_v1_hayao_60', 
        'animegan_v2_hayao_64', 
        'animegan_v2_hayao_99', 
        'animegan_v2_paprika_54', 
        'animegan_v2_paprika_74', 
        'animegan_v2_paprika_97', 
        'animegan_v2_paprika_98', 
        'animegan_v2_shinkai_33',
        'animegan_v2_shinkai_53'
    ]

    models = []
    for pretrained_model in tqdm(pretrained_models):
        modelpath = os.path.join('AnimeGAN', 'pertrained_models', pretrained_model, 'inference_model')
        models.append(load_model(modelpath, args.use_gpu))

    input_data = preprocess(args.img_path, args.max_size, args.min_size)

    outputs = []
    for model in tqdm(models):
        predictor, input_handle, output_handle = model
        input_handle.copy_from_cpu(input_data)
        predictor.run()
        output = output_handle.copy_to_cpu()
        outputs.append(output)

    postprocess(outputs, args.save_path, pretrained_models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--img_path", type=str, default='AnimeGAN/test.jpg', help="the input img path.")
    parser.add_argument("--save_path", type=str, default='AnimeGAN/save_imgs', help="the save path of output imgs.")
    parser.add_argument("--use_gpu", type=bool, default=False, help="the save path of output imgs.")
    parser.add_argument("--max_size", type=int, default=512, help="the save path of output imgs.")
    parser.add_argument("--min_size", type=int, default=32, help="the save path of output imgs.")
 
    args = parser.parse_args()
    main(args)
