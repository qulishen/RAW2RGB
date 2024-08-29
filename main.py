
import argparse

import numpy as np
import torch
from torchvision import utils
from TorchTools.ArgsTools.base_args import BaseArgs
from datasets import process
from TorchTools.DataTools.FileTools import save_tensor_to_cv2img
import torchvision.transforms.functional as TF

def process_single_raw(src_path, height=3072, width=4096):
    # 1. 读取RAW图像数据，使用其他读取方式代替 rawpy
    def load_raw_image(file_path, height, width):
        img = np.fromfile(file_path, dtype=np.uint16)
        img = img.reshape((height, width))  # 重塑为正确的高度和宽度
        return img

    raw_image = load_raw_image(src_path, height, width)

    # 裁剪
    raw_image = raw_image[:,:]
    # raw_image = raw_image[:,1:-1]
    # raw_image = raw_image[1:-1,:1:-1]
    # raw_image = raw_image[:,:]



    # 2. 估计黑电平和白电平
    def estimate_black_white_levels(raw_image):
        black_level = 63.937500  # 根据实际信息中的黑电平
        white_level = np.percentile(raw_image, 99.9)  # 使用高百分位数估计白电平
        return black_level, white_level

    black_level, white_level = estimate_black_white_levels(raw_image)

    # 3. 黑白电平校正
    normalized_image = (raw_image - black_level) / (white_level - black_level)
    normalized_image = np.clip(normalized_image, 0, 1)  # 确保像素值在 [0, 1] 范围内




    colormatrix = np.asarray([[1.429688, -0.468750, 0.039062],
                              [-0.210938, 1.203125, 0.007812],
                              [-0.007812, -0.632812, 1.640625]]).astype(np.float32)





    # 使用实际的白平衡增益
    red_gain = np.array([1.754883])  # 根据实际信息中的 redGain
    blue_gain = np.array([1.717773])  # 根据实际信息中的 blueGain

    metadata = {
        'colormatrix': colormatrix,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }

    return normalized_image, metadata



def main():



    # 图片路径
    input_paths = 'IMG20230101030649_rawdump_4096x3072_02.raw'
    gt_path = 'output2.png'

    # read image
    rggb, matainfo = process_single_raw(input_paths)
    ccm, red_g, blue_g = process.metadata2tensor(matainfo)
    ccm, red_g, blue_g = ccm.to(args.device), red_g.to(args.device), blue_g.to(args.device)

    raw_image_in = torch.unsqueeze(TF.to_tensor(rggb.astype(np.float32)), dim=0).to(args.device)

    if 'raw' in args.in_type:
        print("raw_image_in---------------------------", raw_image_in.shape)
        B, C, H, W = raw_image_in.shape
        raw_image_in = raw_image_in.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous().view(B, 4,
                                                                                                                 H // 2,
                                                                                                                 W // 2)
        scale = args.scale * 2
        print("scale--------------", scale)
    else:
        scale = args.scale

    linrgb = torch.stack((raw_image_in[:, 0, :, :], raw_image_in[:, 1, :, :] / 2 + raw_image_in[:, 2, :, :] / 2,
                          raw_image_in[:, 3, :, :]), dim=1)

    linrgb = process.rgb2srgb(linrgb, red_g, blue_g, ccm)



    save_tensor_to_cv2img(linrgb, gt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args

    main()


