# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/3/16 18:02
"""
from torchvision.transforms import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np

ValImgTransform = transforms.Compose([
	# transforms.Resize(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])


def ImgSplit(img_big, row_grid, row_overlap, col_grid, col_overlap, show = False):
	row_total, col_total = img_big.shape[0], img_big.shape[1]
	split_dic = {}
	# %% 将大尺寸图像拆分为小图
	# 拆行
	row_over = False
	for i in range(0, 10000000):  # 行裁剪到第几个
		row_start = i * (row_grid - row_overlap)  # row_start拆分的起始行
		row_end = i * (row_grid - row_overlap) + row_grid  # row_end拆分的结束行
		if row_end >= row_total:  # 如果结束行超过或等于图片的最大尺寸
			row_end = row_total  # 则取图片的最下边X行
			row_start = row_end - row_grid
			row_over = True  # 最后裁剪一次即结束
		# 拆列
		col_over = False
		for j in range(0, 10000000):  # 列裁剪到第几个
			col_start = j * (col_grid - col_overlap)  # col_start拆分的起始列
			col_end = j * (col_grid - col_overlap) + col_grid  # col_end拆分的结束列
			if col_end >= col_total:  # 如果结束列超过或等于图片的最大尺寸
				col_end = col_total  # 则取图片的最右边X列
				col_start = col_end - col_grid
				col_over = True  # 最后裁剪一次即结束
			img_sml = img_big[row_start:row_end, col_start:col_end]  # 截图检测
			split_dic[(row_start, col_start)] = img_sml  # 以小图左上角的坐标作为key
			if col_over:  # 一次列搜索完毕，换下一次行
				break
		if row_over:
			break  # 行搜索也完毕
	# %% 验证拆分结果
	if show:
		for (row_start, col_start) in split_dic.keys():
			print(row_start, col_start)
			fig, axes = plt.subplots(1, 2)
			axes[0].imshow(img_big)
			rect = plt.Rectangle((col_start, row_start), col_grid, row_grid, fill=False, edgecolor = 'red', linewidth=3)
			axes[0].add_patch(rect)
			axes[1].imshow(split_dic[(row_start, col_start)])
			plt.show()
	return split_dic

def ImgMerge(split_dic, row_grid, row_overlap, col_grid, col_overlap, show = False):
	# %% 拆分图合并
	big_img = np.zeros((10000, 30000), dtype=np.uint8)
	# big_img = np.zeros((100000, 100000), dtype=np.uint8)
	row_end, col_end = 0, 0  # 合并图像的行末和列尾
	for (row_start, col_start) in split_dic.keys():
		big_img[row_start:row_start+row_grid, col_start:col_start+col_grid] = split_dic[(row_start, col_start)]
		row_end, col_end = row_start + row_grid, col_start + col_grid
	big_img = big_img[0:row_end, 0:col_end]
	if show:
		plt.imshow(big_img)
		plt.show()
	return big_img