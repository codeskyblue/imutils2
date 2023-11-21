# coding: utf-8
#

import base64
import pathlib
import pytest

import cv2
import numpy as np
from PIL import Image
from pytest_httpserver import HTTPServer

from imutils2 import cv2bytes, cv2gray, cv2pil, imread, pil2cv, url_to_image, merge_app_screenshots


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


def test_cv2bytes():
    cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
    bytes_img = cv2bytes(cv_img)
    assert isinstance(bytes_img, bytes)
    
    img2 = imread(bytes_img)
    assert isinstance(img2, np.ndarray)
    assert img2.shape == cv_img.shape

    
def test_url_to_image(httpserver: HTTPServer):
    imagedata = pathlib.Path("./tests/testdata/pyimagesearch.png").read_bytes()
    httpserver.expect_oneshot_request("/test.png").respond_with_data(imagedata)

    url = httpserver.url_for("/test.png")
    img = url_to_image(url)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0
    assert img.shape[1] > 0

    img2 = url_to_image(url)
    assert isinstance(img2, np.ndarray)
    assert len(img2.shape) == 3
    assert img2.shape[2] == 3
    assert img2.shape[0] > 0
    assert img2.shape[1] > 0

    with pytest.raises(ValueError):
        im3 = url_to_image(url, cached=False)
        assert len(img2.shape) == 3


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
        Image.open("./tests/testdata/pyimagesearch.png"),
    ]:
        im = imread(data)
        check_image(im)
    
    for invalid_data in (b'xxxx', None, 1, 'xxxx', pathlib.Path("./xxxx.jpg")):
        with pytest.raises(ValueError):
            imread(invalid_data)
    

@pytest.mark.skip("local test")
def test_merge_app_screenshots2():
    im1 = pathlib.Path("tmp/a1.jpg")
    im2 = pathlib.Path("tmp/a2.jpg")
    im3 = pathlib.Path("tmp/a3.jpg")
    im5 = pathlib.Path("tmp/a5.jpg")
    img = merge_app_screenshots([im1, im2, im3, im5, im5], draw_merge_line=True)
    assert isinstance(img, np.ndarray)
    cv2pil(img).show()


def test_merge_app_screenshots():
    images = [pathlib.Path(f"./tests/testdata/m{i}.jpg") for i in [1,2,3]]
    img = merge_app_screenshots(images, draw_merge_line=True)
    assert isinstance(img, np.ndarray)
    # cv2pil(img).show()

    images = [pathlib.Path(f"./tests/testdata/m{i}.jpg") for i in [1,3]]
    img = merge_app_screenshots(images, draw_merge_line=True)
    assert isinstance(img, np.ndarray)
    # cv2pil(img).show()

    with pytest.raises(ValueError):
        merge_app_screenshots([pathlib.Path("./tests/testdata/m1.jpg"), pathlib.Path("./tests/testdata/pyimagesearch.png")])

