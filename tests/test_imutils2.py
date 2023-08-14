# coding: utf-8
#

import pytest

from PIL import Image
import numpy as np
from imutils2 import pil2cv, cv2pil, cv2gray, url_to_image

def test_pil2cv():
    pil_img = Image.new('RGB', (100, 100), color='red')
    cv_img = pil2cv(pil_img)
    assert isinstance(cv_img, np.ndarray)



def test_cv2pil():
    cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
    pil_img = cv2pil(cv_img)
    assert isinstance(pil_img, Image.Image)


def test_cv2gray():
    # 支持彩色图转化
    cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray_img = cv2gray(cv_img)
    assert isinstance(gray_img, np.ndarray)
    assert len(gray_img.shape) == 2

    # 灰色图直接返回
    cv_img = np.zeros((100, 100), dtype=np.uint8)
    gray_img = cv2gray(cv_img)
    assert isinstance(gray_img, np.ndarray)
    assert len(gray_img.shape) == 2


def test_url_to_image():
    url = "https://www.baidu.com/img/bd_logo1.png"
    img = url_to_image(url)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0
    assert img.shape[1] > 0