#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2019/10/24 22:03
"""
import os,glob
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import matplotlib.pyplot as plt

SaveFolder = '1PotData_Input1_IF8_Epoch700_Dilation2'

plt.ion()
BCETrainLosses = []
BCEValidLosses = []
with open(os.path.join(SaveFolder, 'log.txt'), 'r') as f:
	lines = f.readlines()
	for i, line in enumerate(lines):
		if 'Train' in line:
			Epoch = float(line.strip().split('\t')[2].split(':')[1])
			Lr = float(line.strip().split('\t')[3].split(':')[1])
			BCELoss = float(lines[i+1].strip().split('\t')[1].split(':')[1])
			BCETrainLosses.append(np.array([Epoch, Lr, BCELoss]))

		elif 'Valid' in line:
			Epoch = float(line.strip().split('\t')[16].split(':')[1])
			Lr = float(line.strip().split('\t')[17].split(':')[1])
			BCELoss = float(lines[i+1].strip().split('\t')[15].split(':')[1])
			BCEValidLosses.append(np.array([Epoch, Lr, BCELoss]))
exit()

BCETrainLosses = np.vstack(BCETrainLosses)
BCEValidLosses = np.vstack(BCEValidLosses)

def Split(BCETrainLosses):
	# %% 根据Lrs对TrainLosses进行分割
	Lrs = np.unique(BCETrainLosses[..., 1])
	BCENewTrainLosses = []
	for Lr in Lrs:
		Indx = np.where(BCETrainLosses[:,1] == Lr)
		BCETrainLoss = BCETrainLosses[Indx, :][0]
		BCENewTrainLosses.append(BCETrainLoss)
	return BCENewTrainLosses


fig = plt.figure(SaveFolder)
# NewBCETrainLosses = Split(BCETrainLosses)
# NewDCTrainLosses = Split(DCTrainLosses)
# for i, BCETrainLoss in enumerate(NewBCETrainLosses):
# 	if i==0:
# 		plt.plot(BCETrainLoss[..., 0], BCETrainLoss[..., 2], linestyle = '--', label='Train:BCE Loss')
# 	else:
# 		plt.plot(BCETrainLoss[..., 0], BCETrainLoss[..., 2], linestyle = '--')
# for i, DCTrainLoss in enumerate(NewDCTrainLosses):
# 	if i == 0:
# 		plt.plot(DCTrainLoss[..., 0], DCTrainLoss[..., 2], linestyle = '-', label='Train:DC Loss')
# 	else:
# 		plt.plot(DCTrainLoss[..., 0], DCTrainLoss[..., 2], linestyle = '-')
plt.plot(BCETrainLosses[..., 0], BCETrainLosses[..., 2], label='Train:BCE Loss')
plt.plot(BCEValidLosses[..., 0], BCEValidLosses[..., 2], label='Val:BCE Loss')

# plt.ylim(0,0.2)
plt.yscale('log')
plt.legend()
plt.ioff()
plt.show()





