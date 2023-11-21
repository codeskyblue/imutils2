#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Created on Tue Nov 21 2023 15:53:41 by codeskyblue
"""

import typing

import cv2
import numpy as np

from .core import AnyImage, imread


def merge_app_screenshots(images: typing.List[AnyImage]) -> np.ndarray:
    """
    App滑动中的截图合并为一张图片

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
    current = None
    for i in range(image_count-1):
        im1, im2 = images[i], images[i+1]
        coord1, coord2 = cal_common_part(im1, im2, top_bottom_min_similarity=0.9)
        head = im1[0:coord1[0]]
        main_top = im1[coord1[0]:coord1[1]]
        main_same = im2[coord2[0]:coord2[1]]
        main_bottom = im2[coord2[1]:coord2[2]]
        tail = im2[coord2[2]:]
        if current is None:
            current = np.vstack([head, main_top, main_same, main_bottom, tail])
        else:
            cut_idx = coord1[1] - im1.shape[0]
            current = np.vstack([current[:cut_idx], main_same, main_bottom, tail])
    return current


def cal_common_part(im1, im2, colsize: int = 60, top_bottom_min_similarity: float = 0.8, 
                    body_min_similarity: float = 0.9,
                   body_search_start: float = 0.2):
    """
    计算两张图片中相似的部分，下标通常头部，尾部有部分相同的，中间部分因为滑动的原因有大约50%相同的
    
    Args:
        im1, im2: opencv图片格式
        colsize: 缩放后的宽度
        top_bottom_min_similarity: 当每一行像素超过该阈值时，认为是一样的
        body_min_similarity: 中间区域相似度阈值
        body_search_start: 相似区域查找起始位置 0-1
        
    当前的算法还是有点问题的，如果相同的头部查找不准确的话，会影响中间部分相同区域的查找。不过好处是速度快。
    如果有图像最长相似区域去查找的话，计算复杂度就是N的平方，虽然更准确一点，但是太慢了。
    """
    img1 = cv2.resize(im1, (colsize, im1.shape[0]))
    img2 = cv2.resize(im2, (colsize, im2.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, bin1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, bin2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    # 寻找相同头部，top为第一张图片的下标
    # 考虑到头部和尾部的长度基本不会超过屏幕的50%，每次对比一半就行了
    min_len = min(len(bin1), len(bin2)) // 2
    result = np.sum(bin1[:min_len] == bin2[:min_len], axis=1) > colsize*top_bottom_min_similarity
    top = np.argmin(result)
    
    # 寻找相同尾部，bottom为第二张图片的下标
    result = np.sum(bin1[-min_len:] == bin2[-min_len:], axis=1) > colsize*top_bottom_min_similarity
    bottom = len(bin2) - np.argmin(result[::-1])

    # 移除头部和尾部的中间部分
    mid_bin1 = bin1[top:bottom]
    mid_bin2 = bin2[top:bottom]
    
    A = []
    mid_len = len(mid_bin1)
    # 从中下开始查找相似的部分
    start_idx, end_idx = int(mid_len*body_search_start), len(mid_bin1)
    
    im1_height, im2_height = im1.shape[0], im2.shape[0]
    expect_overlap_len = int(min(im1_height, im2_height) * 0.2) # 中间重叠部分的最小长度
    
    # 从上到下找相似的部分
    cmp2 = mid_bin2[0:expect_overlap_len] # 只看头部这一块
    for i in range(start_idx, end_idx):
        overlap_len = mid_len - i
        cmp1 = mid_bin1[i:i+expect_overlap_len]
        # 用于加速处理的代码，不过好像速度本身就挺快的
        #cmp1 = shrink_array(cmp1, 4, 50)
        #cmp2 = shrink_array(cmp2, 4, 50)
        if len(cmp1) != len(cmp2):
            break
        # 对比一下相似度
        result = np.sum((np.sum(cmp1 == cmp2, axis=1) >= colsize*0.8))
        score = result/len(cmp1)
        A.append((i, score, overlap_len))
        
    A.sort(key=lambda v: v[1], reverse=True)

    if A and A[0][1] >= body_min_similarity:
        start, overlap_len = A[0][0], A[0][2]
    else:
        start, overlap_len = end_idx, 0
    return (top, top+start, top+start+overlap_len), (top, bottom-overlap_len, bottom)


def shrink_array(arr: np.ndarray, seg_size: int = 3, chunk_size: int = 2) -> np.ndarray:
    length = len(arr)
    if length <= seg_size * chunk_size:
        return arr
    step = length // seg_size
    arrs = np.concatenate([arr[i*step:i*step+chunk_size] for i in range(seg_size)])
    return arrs