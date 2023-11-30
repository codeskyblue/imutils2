# coding: utf-8
#
from __future__ import annotations

import base64
import functools
import pathlib
import re
import typing

import cv2
import numpy as np
import requests
from PIL import Image
from typeguard import typechecked

AnyImage = typing.Union[str, pathlib.Path, bytes, Image.Image, np.ndarray]
DEFAULT_REQUEST_TIMEOUT = 30

def imread(data: AnyImage, type: str | None = None) -> np.ndarray:
    """ convert any data to opencv type (gray or rgb)

    RGBA will convert to RGB, Alpha part will fill with white

    Support:
    - url: https://www.baidu.com/baidu.png
    - base64: data:image,xlkj1jf=/
    - pathlib.Path("./test.jpg")
    - binary data: b'xlkeRGBxls...'
    - Image.Image

    if param is opencv data, then return it without modify
    """
    if isinstance(data, str):
        if re.match(r"https?://", data):
            url = data
            try:
                r = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT)
                r.raise_for_status()
                return imread(r.content)
            except requests.RequestException as e:
                raise ValueError("Invalid image url", url) from e
        elif data.startswith("data:image"):
            binary_data = base64.b64decode(data.split(",")[1])
            return imread(binary_data)
        else:
            return imread(pathlib.Path(data))
    elif isinstance(data, Image.Image):
        return pil2cv(data)
    elif isinstance(data, pathlib.Path):
        im = cv2.imread(str(data))
        if im is None:
            raise ValueError("Invalid image path", data)
        return im
    elif isinstance(data, (bytes, bytearray)):
        img_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        im = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError("Invalid image data", data)
        if len(im.shape) == 3 and im.shape[2] == 4:
            # convert rgba to rgb
            # 创建一个白色的画布，然后把非透明的部分画上去
            background = np.full_like(im, (255, 255, 255, 255))
            alpha_mask = im[:,:,3] > 0
            background[alpha_mask,:] = im[alpha_mask,:]
            im = background[:,:,:3]
            # im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR) # 这个转化的图像有问题，所以就先不用这个方法了
        return im
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Unknown data: {data}")


@typechecked
def pil2cv(pil_image: Image.Image) -> np.ndarray:
    """ Convert from pillow image to opencv """
    # convert PIL to OpenCV
    pil_image = pil_image.convert('RGB')
    cv2_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image


@typechecked
def cv2pil(cv_img: np.ndarray) -> Image.Image:
    """ Convert opencv to Pillow """
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def cv2gray(cv_img: np.ndarray) -> np.ndarray:
    """ Convert opencv to grayscale """
    if len(cv_img.shape) == 2:
        return cv_img
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)


def cv2bytes(cv2_image: np.ndarray, format="jpg") -> bytes:
    """ convert cv2 image to bytes """
    return cv2.imencode("."+format.lower(), cv2_image)[1].tobytes()


def show_image(images: AnyImage | typing.List[AnyImage]): # pragma: no cover
    """ Show image in jupyter """
    import matplotlib.pyplot as plt
    
    if not isinstance(images, (list, tuple)):
        images = [images]

    # load image using cv2....and do processing.
    for i, img_data in enumerate(images):
        img = imread(img_data)
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(str(i))
        # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()


def url_to_image(url: str, cached: bool = True) -> np.ndarray:
    """ download image from url and convert to opencv image"""
    if not cached:
        return imread(url)
    else:
        im = _cache_imread(url)
        return im.copy()


@functools.lru_cache(maxsize=32)
def _cache_imread(url: str) -> np.ndarray:
    return imread(url)

