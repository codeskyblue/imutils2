#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Created on Tue Nov 21 2023 15:53:41 by codeskyblue
"""

import typing

import cv2
import numpy as np

from .core import AnyImage, imread


def merge_app_screenshots(images: typing.List[AnyImage], draw_merge_line: bool = False, **kwargs) -> np.ndarray:
    """
    App滑动中的截图合并为一张图片

    Args:
        images: 图片列表，图片宽度必须一致
        draw_merge_line: 是否在合并的地方画一条线

    Raises:
        ValueError: 图片宽度不一致
    """
    images = [imread(img) for img in images]
    widths = [img.shape[1] for img in images]
    if len(set(widths)) != 1:
        raise ValueError("images width not equal")

    image_count = len(images)
    if image_count == 0:
        raise ValueError("images is empty")
    if len(images) == 1:
        return images[0]
    current = images[0]
    for i in range(1, image_count):
        img = images[i]
        index1, index2 = cal_common_part(current, img, **kwargs)
        current = np.vstack([current[:index1], img[index2:]])
        if draw_merge_line:
            color = (0, 0, 255)
            cv2.line(current, (0, index1), (current.shape[1], index1), color, thickness=2)
    return current


def cal_common_part(im1: np.ndarray, im2: np.ndarray,
                    colsize: int = 60,
                    skip_head_percent: float = 0.2,
                    compare_percent: float = 0.2,
                    line_min_similarity: float = 0.9,
                    area_min_similarity: float = 0.9) -> typing.Tuple[int, int]:
    """
    计算两张图片中相似的部分，下标通常头部，尾部有部分相同的，中间部分因为滑动的原因有大约50%相同的
    
    Args:
        im1, im2: opencv图片
        colsize: 缩放后的宽度
        skip_head_percent: 跳过开头部分的百分比
        compare_percent: 查找相同部分时的比较长度百分比
        line_min_similarity: 行相似度阈值
        area_min_similarity: 区域相似度阈值
    
    Return:
        第一张图的下标，第二张图的下标
    """
    img1 = cv2.resize(im1, (colsize, im1.shape[0]))
    img2 = cv2.resize(im2, (colsize, im2.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, bin1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    img1_height, img2_height = img1.shape[0], img2.shape[0]
    # 寻找相同头部
    # 考虑到头部和尾部的长度基本不会超过屏幕的50%，每次对比一半就行了
    min_len = min(img1_height, img2_height) // 2
    result = np.sum(bin1[:min_len] == bin2[:min_len], axis=1) > colsize*0.8
    top = np.argmin(result)

    top1 = max(top, img1_height - img2_height) # 确定第一张图的起始查找位置
    top2 = max(top, int(skip_head_percent*img2_height)) # 第二张图的起始查找位置，跳过前20%的区域

    compare_len = int(min(img1_height, img2_height) * compare_percent) # 中间重叠部分的预设长度

    match_results = []
    bin2_part = bin2[top2:top2+compare_len]
    for i in range(top1, img1_height):
        bin1_part = bin1[i:i+compare_len]
        if len(bin1_part) != len(bin2_part):
            break

        # 对比一下相似度
        result = np.sum((np.sum(bin1_part == bin2_part, axis=1) >= colsize*line_min_similarity))
        score = result/compare_len
        match_results.append((i, score))

    match_results.sort(key=lambda v: v[1], reverse=True)
    if match_results and match_results[0][1] >= area_min_similarity:
        index1, index2 = match_results[0][0], top2
    else:
        # 找不到相同的区域，就整体拼接
        index1, index2 = img1_height, 0
    return index1, index2
