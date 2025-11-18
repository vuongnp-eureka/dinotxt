# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Sequence

import torch
from torchvision import transforms

# Compatibility layer for torchvision versions without v2
try:
    from torchvision.transforms import v2
    HAS_V2 = True
except ImportError:
    # Fallback to old transforms API for older torchvision versions
    HAS_V2 = False
    # Create a compatibility shim
    class V2Compat:
        """Compatibility shim for torchvision.transforms.v2 when not available"""
        @staticmethod
        def Normalize(mean, std):
            return transforms.Normalize(mean=mean, std=std)
        
        @staticmethod
        def ToDtype(dtype, scale=False):
            # In v2: ToImage() -> uint8 [0,255], ToDtype(scale=True) -> float32 [0,1]
            # In old API: ToTensor() -> float32 [0,1] already
            # So if scale=True and dtype=float32, it's already correct (no-op)
            class ToDtypeTransform:
                def __init__(self, dtype, scale):
                    self.dtype = dtype
                    self.scale = scale
                
                def __call__(self, img):
                    # img should already be a tensor from ToImage/ToTensor
                    if isinstance(img, torch.Tensor):
                        # If it's already float32 [0,1] and we want float32 with scale=True, it's correct
                        if img.dtype == torch.float32 and self.dtype == torch.float32 and img.max() <= 1.0:
                            return img  # Already in correct format
                        # Otherwise convert to target dtype
                        return img.to(self.dtype)
                    else:
                        # Fallback: convert PIL to tensor (shouldn't happen if ToImage is used)
                        img = transforms.ToTensor()(img)
                        return img.to(self.dtype)
            return ToDtypeTransform(dtype, scale)
        
        @staticmethod
        def Compose(transforms_list):
            return transforms.Compose(transforms_list)
        
        @staticmethod
        def Resize(size, max_size=None, interpolation=None):
            if interpolation is None:
                interpolation = transforms.InterpolationMode.BICUBIC
            elif isinstance(interpolation, str):
                interpolation = getattr(transforms.InterpolationMode, interpolation)
            if max_size is not None:
                # For max_size, we need to handle it differently
                class ResizeMaxSize:
                    def __init__(self, max_size, interpolation):
                        self.max_size = max_size
                        self.interpolation = interpolation
                    
                    def __call__(self, img):
                        w, h = img.size
                        if max(w, h) > self.max_size:
                            if w > h:
                                new_w, new_h = self.max_size, int(h * self.max_size / w)
                            else:
                                new_w, new_h = int(w * self.max_size / h), self.max_size
                            img = transforms.Resize((new_h, new_w), interpolation=self.interpolation)(img)
                        return img
                return ResizeMaxSize(max_size, interpolation)
            return transforms.Resize(size, interpolation=interpolation)
        
        @staticmethod
        def CenterCrop(size):
            return transforms.CenterCrop(size)
        
        @staticmethod
        def ToImage():
            # In v2, ToImage() converts PIL to tensor without scaling
            # In old API, ToTensor() converts PIL to tensor and scales to [0,1]
            # We'll use ToTensor() which is compatible
            return transforms.ToTensor()
        
        class InterpolationMode:
            # Handle both old and new torchvision versions
            pass
    
    # Set InterpolationMode.BICUBIC based on availability
    if hasattr(transforms, 'InterpolationMode'):
        V2Compat.InterpolationMode.BICUBIC = transforms.InterpolationMode.BICUBIC
    else:
        # Fallback for very old versions (use integer constant)
        V2Compat.InterpolationMode.BICUBIC = 3  # PIL.Image.BICUBIC
    
    v2 = V2Compat()


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        size = (resize_size, resize_size)
        transform = v2.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        transform = v2.Resize(size=None, max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = v2.Resize(resize_size, interpolation=interpolation)
        return transform


def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    return transform


def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )

