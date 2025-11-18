# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Sequence

import torch
from torchvision.transforms import v2


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

