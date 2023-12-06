import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
#matplotlib.use('Qt5Agg')

# 1：读入图片
img = plt.imread("C:\\Users\\hp\\Desktop\\202141594912705.jpg")
# 或者是img=cv2.imread("./img/cat1.jpg")
# 又或者是#img=matplotlib.image.imread("./img/cat1.jpg")


# 2：读取一些图片信息，比如图像的宽，高，通道数，最大像素值，最小像素值
print(img.shape)  # (227, 286, 3)
print(img.shape[0])  # 图片宽度为227
print(img.shape[1])  # 图片高度为286
print(img.shape[2])  # 图片通道数为3
print(img.mean())  # 图片像素平均值
print(img.min(), img.max())

# 3：显示图片
# 这两行代码要连用才能把图片显示出来，单单一行代码是无法将图片显示出来的
plt.imshow(img)
pylab.show()