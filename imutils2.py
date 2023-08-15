# coding: utf-8
#

import base64
import pathlib
import re
import typing
import numpy as np
import cv2
from PIL import Image
from typeguard import typechecked
import matplotlib.pyplot as plt
import urllib.request


def imread(data: typing.Union[str, pathlib.Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """ convert any data to opencv type
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
            with urllib.request.urlopen(url) as url_response:
                img_pil = Image.open(url_response)
                return imread(img_pil)
        elif data.startswith("data:image"):
            binary_data = base64.b64decode(data.split(",")[1])
            return imread(binary_data)
        else:
            binary_data = base64.b64decode(data)
            return imread(binary_data)
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
            raise ValueError("Invalid image url", url)
        if len(im.shape) == 3 and im.shape[2] == 4:
            # convert rgba to rgb
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        return im
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Unknown data type: {type(data)}")
        
        

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


def show_image(images: typing.List[np.ndarray]):
    """ Show image in jupyter """
    if not isinstance(images, list):
        images = [images]

    # load image using cv2....and do processing.
    for i, img in enumerate(images):
        if isinstance(img, Image.Image):
            img = pil2cv(img)
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(str(i))
        # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()


def url_to_image(url: str) -> np.ndarray:
    return imread(url)