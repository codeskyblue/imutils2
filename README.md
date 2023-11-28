[![PyPI version](https://badge.fury.io/py/imutils2.svg)](https://badge.fury.io/py/imutils2)
[![codecov](https://codecov.io/gh/codeskyblue/imutils2/branch/master/graph/badge.svg?token=XZP5cusLGW)](https://codecov.io/gh/codeskyblue/imutils2)

已经有了一个imutils库,但是这个imutils也不能完全满足我的需求,所以就整了一个imutils2

# 常用函数
- imread 读取或转化图像为opencv格式
- pil2cv Pillow转Opencv格式
- cv2pil Opencv转Pillow格式
- cv2gray Opencv转成灰度图
- url_to_image 下载URL为图像
- show_image 查看图像兼容jupyter, requires matplotlib
- merge_app_screenshots App下滑长截图拼接

```python
from imutils2 import *

im1 = imread("m1.jpg")
im2 = imread("m2.jpg")
im3 = imread("m3.jpg")
output = merge_app_screenshots([im1, im2, im3])
show_image(output)
```

写的不一定全，更多的例子可以去tests/test_imutils2.py中查看