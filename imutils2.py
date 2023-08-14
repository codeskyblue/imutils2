# coding: utf-8
#

import re
import typing
import numpy as np
import cv2
from PIL import Image
from typeguard import typechecked
import matplotlib.pyplot as plt
import urllib.request


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
    assert re.match(r"https?://", url), url
    with urllib.request.urlopen(url) as url_response:
        img_pil = Image.open(url_response).convert("RGB")
    # convert PIL image to OpenCV image
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    #binary_data = get_url_content(url)
    #img_bytes = np.asarray(bytearray(binary_data), dtype=np.uint8)
    #im = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("Invalid image url", url)
    return img_cv