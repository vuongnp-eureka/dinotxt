# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

__version__ = "0.1.0"

from .hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from .data.transforms import make_classification_eval_transform

__all__ = ["dinov3_vitl16_dinotxt_tet1280d20h24l", "make_classification_eval_transform"]
