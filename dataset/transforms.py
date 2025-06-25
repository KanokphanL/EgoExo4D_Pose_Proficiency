import cv2
import numpy as np

import torch
import torchvision.transforms.functional as F
import random
from typing import Union, Tuple

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # 禁用Albumentations的版本检查

import albumentations as A

class AlbumentationsAug:
    def __init__(self):
        # 定义transforms列表
        self.transforms = A.Compose([
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
            A.CoarseDropout(
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ])

    def __call__(self, image):
        # 对输入图像应用transforms
        return self.transforms(image=image)['image']

class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def _get_hsv_gains(self):
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta, self.saturation_delta, self.value_delta
        ]
        # random selection of h, s, v
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        return hsv_gains

    def __call__(self, img):
        # 假设img是cv2.RGB图像
        # img = results['img']
        hsv_gains = self._get_hsv_gains()
        # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        # cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2RGB, dst=img)

        # results['img'] = img
        # return results
        return img

def vertical_flip(image, prob=0.5, keypoints=None):
    """
    Perform vertical flip on the given images and corresponding keypoints.
    Args:
        image (ndarray): images to perform vertical flip, the dimension is
             `height` x `width` x `channel`.
        prob (float): probility to flip the images.
        keypoints (ndarray or None): optional. Corresponding 3D keypoints to images.
            Dimension is `num joints` x 3.
    Returns:
        images (ndarray): flipped images with dimension of
            `height` x `width` x `channel`.
        flipped_keypoints (ndarray or None): the flipped keypoints with dimension of
            `num keypoints` x 3.
    """
    if keypoints is None:
        flipped_keypoints = None
    else:
        flipped_keypoints = keypoints.copy()

    if np.random.uniform() < prob:
        # images = images.flip((-1))
        image = np.flip(image, axis=0)

        # if len(images.shape) == 3:
        #     width = images.shape[2]
        # elif len(images.shape) == 4:
        #     width = images.shape[3]
        # else:
        #     raise NotImplementedError("Dimension does not supported")
        if keypoints is not None:
            # flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1
            flipped_keypoints[:, 1] = - flipped_keypoints[:, 1]

    return image, flipped_keypoints

def horizontal_flip(image, prob=0.5, keypoints=None):
    """
    Perform horizontal flip on the given images and corresponding keypoints.
    Args:
        image (ndarray): images to perform horizontal flip, the dimension is
             `height` x `width` x `channel`.
        prob (float): probility to flip the images.
        keypoints (ndarray or None): optional. Corresponding 3D keypoints to images.
            Dimension is `num joints` x 3.
    Returns:
        images (ndarray): flipped images with dimension of
            `height` x `width` x `channel`.
        flipped_keypoints (ndarray or None): the flipped keypoints with dimension of
            `num keypoints` x 3.
    """
    if keypoints is None:
        flipped_keypoints = None
    else:
        flipped_keypoints = keypoints.copy()

    if np.random.uniform() < prob:
        image = np.flip(image, axis=1)

        if keypoints is not None:
            flipped_keypoints[:, 0] =  - flipped_keypoints[:, 0]

    return image, flipped_keypoints


def random_horizontal_flip(video: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    对视频clip进行随机水平翻转数据增强
    
    参数:
        video: 输入视频张量，形状为 (C, T, H, W)
        p: 翻转概率，默认0.5
        
    返回:
        翻转后的视频张量，形状保持 (C, T, H, W)
    """
    # 确保输入张量维度正确
    assert video.dim() == 4, "输入张量必须为4维，形状应为 (C, T, H, W)"
    
    # 生成随机数决定是否翻转（在输入张量的设备上生成，保持一致性）
    do_flip = torch.rand(1) < p
    
    if do_flip:
        # 在宽度维度（最后一维）上进行翻转
        flipped_video = torch.flip(video, dims=[-1])  # 等价于 dims=[3]（当形状为(C,T,H,W)时）
    else:
        flipped_video = video
    
    return flipped_video



class VideoRandomResizedCrop:
    """
    对视频clip进行随机缩放裁剪数据增强的可配置类
    适用于形状为 (C, T, H, W) 的视频张量
    """
    
    def __init__(self, 
                 size: Union[int, Tuple[int, int]],
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (3./4., 4./3.),
                 interpolation_mode: str = 'bilinear'):
        """
        初始化随机缩放裁剪增强器
        
        参数:
            size: 输出尺寸，可以是整数或元组 (height, width)
            scale: 裁剪区域占原图的比例范围 (min_area, max_area)
            ratio: 裁剪区域的宽高比范围 (min_ratio, max_ratio)
            interpolation_mode: 插值模式，可选 'nearest', 'bilinear', 'bicubic'
        """
        # 转换size为(height, width)格式
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
            
        self.scale = scale
        self.ratio = ratio
        self.interpolation_mode = interpolation_mode
        
        # 映射插值模式名称到torchvision的枚举值
        self._interpolation_modes = {
            'nearest': F.InterpolationMode.NEAREST,
            'bilinear': F.InterpolationMode.BILINEAR,
            'bicubic': F.InterpolationMode.BICUBIC,
        }
        
        # 验证插值模式参数
        if interpolation_mode not in self._interpolation_modes:
            raise ValueError(f"不支持的插值模式: {interpolation_mode}")

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        对输入视频应用随机缩放裁剪增强
        
        参数:
            video: 输入视频张量，形状为 (C, T, H, W)
            
        返回:
            增强后的视频张量，形状为 (C, T, size[0], size[1])
        """
        # 确保输入张量维度正确
        assert video.dim() == 4, "输入张量必须为4维，形状应为 (C, T, H, W)"
        
        # 获取原始视频的尺寸
        _, _, height, width = video.shape
        
        # 尝试10次寻找合适的裁剪区域
        for _ in range(10):
            # 随机生成裁剪区域的比例和宽高比
            area = height * width
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            # 计算裁剪区域的宽和高
            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))
            
            # 确保宽和高在有效范围内
            if 0 < w <= width and 0 < h <= height:
                # 随机生成裁剪的左上角坐标
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                
                # 对视频的每一帧应用相同的裁剪和缩放
                interpolation = self._interpolation_modes[self.interpolation_mode]
                return F.resized_crop(video, i, j, h, w, self.size, interpolation)
        
        # 如果尝试10次后仍未找到合适的裁剪区域，使用中心裁剪
        in_ratio = float(width) / float(height)
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        
        interpolation = self._interpolation_modes[self.interpolation_mode]
        return F.resized_crop(video, i, j, h, w, self.size, interpolation)

    def __repr__(self) -> str:
        """返回类的字符串表示"""
        return (f"{self.__class__.__name__}("
                f"size={self.size}, "
                f"scale={self.scale}, "
                f"ratio={self.ratio}, "
                f"interpolation_mode={self.interpolation_mode})")