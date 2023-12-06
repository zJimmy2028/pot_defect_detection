#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/1/15 9:52
"""
import os, torch, cv2, random
import matplotlib.pyplot as plt
from PIL import Image
from _99Normalization import *
from _03FCN import *
from _41ImagePreprocessing import *

ImgResize = (4096,4096) # image size
# 拆图参数
row_grid = 4096  # 裁剪小图行高
row_overlap = 0  # 裁剪小图行重叠
col_grid = 4096  # 裁剪小图列宽
col_overlap = 0  # 裁剪小图列重叠

# %% Load model
ModelFolder = '3PotData_Input2(4096)_IF8_Epoch2800_Dilation2_kernel5_BigKernel4_SmlKernel4'
Model = Net(InputChannels=2, OutputChannels=1, InitFeatures=8, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
Model.load_state_dict(torch.load(ModelFolder+'/2800.pt', map_location = 'cuda'))
Model.eval()
torch.set_grad_enabled(False)

# %% Testing
TestImageLocation = '../PotData_1/pot'
SaveFolder = 'TestResult_pot'
TestFolders = os.listdir(TestImageLocation)
for TestFolder in TestFolders:
	# %% 载入需要检测图像
	TestFolderPath = os.path.join(TestImageLocation, TestFolder)
	print(TestFolderPath)
	ImgNames = ['img01.png', 'img02.png']
	MultiImgPaths = [os.path.join(TestFolderPath, ImgName) for ImgName in ImgNames]
	MultiImgs = [cv2.imread(Path, cv2.IMREAD_GRAYSCALE) for Path in MultiImgPaths]
	SmlImg1_dic = ImgSplit(MultiImgs[0], row_grid, row_overlap, col_grid, col_overlap, show = False)
	SmlImg2_dic = ImgSplit(MultiImgs[1], row_grid, row_overlap, col_grid, col_overlap, show = False)
	SmlImgDetect_dic = {}
	for key in SmlImg1_dic.keys():
		SmlImg1_array, SmlImg2_array = SmlImg1_dic[key], SmlImg2_dic[key]
		SmlImg1_PIL, SmlImg2_PIL = Image.fromarray(SmlImg1_array), Image.fromarray(SmlImg2_array)
		SmlImg1_tensor = torch.unsqueeze(ValImgTransform(SmlImg1_PIL), dim=0)
		SmlImg2_tensor = torch.unsqueeze(ValImgTransform(SmlImg2_PIL), dim=0)

		# %% 输入神经网络检测
		Input = torch.cat([SmlImg1_tensor, SmlImg2_tensor], dim=1)
		# Input = torch.cat([SmlImg1_tensor,], dim=1)
		InputImg = Input.float().to('cuda')
		OutputImg = Model(InputImg)
		# Generate result image
		OutputImg = OutputImg.cpu().numpy()[0, 0]
		OutputImg = (OutputImg*255).astype(np.uint8)
		SmlImgDetect_dic[key] = OutputImg
	BigOutputImg = ImgMerge(SmlImgDetect_dic, row_grid, row_overlap, col_grid, col_overlap)
	ResultImg = cv2.cvtColor(MultiImgs[1], cv2.COLOR_GRAY2RGB)
	# plt.subplot(121), plt.imshow(ResultImg), plt.title('Original image')
	contour_points = np.argwhere(BigOutputImg > 20)
	ResultImg[contour_points[:, 0], contour_points[:, 1], 2] = 255
	# plt.subplot(122), plt.imshow(ResultImg), plt.title('Detection result')
	# plt.show()

	#ResultImg[:, :, 2] = BigOutputImg
	os.makedirs(os.path.join(ModelFolder, SaveFolder), exist_ok=True)
	cv2.imwrite(os.path.join(ModelFolder, os.path.join(SaveFolder, TestFolder + '.jpg')),ResultImg)

