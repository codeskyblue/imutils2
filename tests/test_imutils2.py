# coding: utf-8
#

import base64
import pathlib

import cv2
import numpy as np
from PIL import Image
from pytest_httpserver import HTTPServer

from imutils2 import cv2gray, cv2pil, imread, pil2cv, url_to_image


def test_pil2cv():
    pil_img = Image.new('RGB', (100, 100), color='red')
    cv_img = pil2cv(pil_img)
    assert isinstance(cv_img, np.ndarray)
    assert len(cv_img.shape) == 3


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
    url = "https://pyimagesearch.com/wp-content/themes/pyi/assets/images/logo.png"
    img = url_to_image(url)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0
    assert img.shape[1] > 0


def test_imread(httpserver: HTTPServer):
    imagedata = pathlib.Path("./tests/testdata/pyimagesearch.png").read_bytes()
    httpserver.expect_request("/test.png").respond_with_data(imagedata)

    def check_image(im):
        assert isinstance(im, np.ndarray)
        assert len(im.shape) == 3
        assert im.shape[2] == 3
        assert im.shape[0] > 0
        assert im.shape[1] > 0
    
    for data in [
        imagedata,
        httpserver.url_for("/test.png"),
        "data:image/png;base64," + base64.b64encode(imagedata).decode("utf-8"),
        base64.b64encode(imagedata).decode("utf-8"),
        pathlib.Path("./tests/testdata/pyimagesearch.png"),
        cv2.imread("./tests/testdata/pyimagesearch.png"),
    ]:
        im = imread(data)
        check_image(im)
    

