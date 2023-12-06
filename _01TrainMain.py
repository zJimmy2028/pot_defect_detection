# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 9:33
"""
import logging, os, torch
from _99Timer import *
from _02MultiPipeDatasetLoader import *
from _03FCN import *
import warnings
warnings.filterwarnings('ignore')

def Train(SaveFolder, Width):
	# %% InitParameters
	BatchSize = 1
	Epochs = 3000
	Lr = 0.01
	LrDecay = 0.8
	LrDecayPerEpoch = 300

	ValidPerEpoch = 15
	SaveEpoch = [Epochs]        # epochs need to save temporarily
	torch.cuda.set_device(0)
	Device = torch.device('cuda:0')
	BCELossWeightCoefficient = 2

	# %% Load Multi-exposure tube contour dataset (METCD)
	print('\n\n\n**************SaveFolder*****************\n')
	os.makedirs(SaveFolder, exist_ok=SaveFolder)
	logging.basicConfig(filename=os.path.join(SaveFolder, 'log.txt'), filemode='w', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')

	FolderPath = '../PotData_1'
	TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, TrainBatchSize=BatchSize, ValBatchSize=BatchSize,
	                                                                             TrainNumWorkers=5, ValNumWorkers=1, Width=Width, ShowSample = False)
	Model = Net(InputChannels=2, OutputChannels=1, InitFeatures=8, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(Device)
	# Model.load_state_dict(torch.load('init.pt', map_location=Device))
	# torchsummary.summary(Model, input_size=(1, 4096, 4096))
	# %% Init optimizer and learning rate
	CriterionBCELoss = nn.BCELoss().to(Device)
	for Epoch in range(1, Epochs + 1):
		End = timer(8)
		if Epoch == 1:
			Optimizer = torch.optim.Adam(Model.parameters(), lr=Lr)
			LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=LrDecayPerEpoch, gamma=LrDecay)

		# %% Training
		Model.train()
		# torch.cuda.empty_cache()
		BCELoss = 0
		print('Epoch:%d, LR:%.8f ' % (Epoch, LrScheduler.get_lr()[0]), end='>> ', flush=True)
		for Iter, (InputImgs, Label, TMImg, SampleName) in enumerate(TrainDataLoader):
			print(Iter, end=' ', flush=True)
			InputImg = torch.cat(InputImgs, dim=1)
			InputImg = InputImg.float().to(Device)
			Label = Label.float().to(Device)
			Weight = Label * (BCELossWeightCoefficient - 1) + 1
			CriterionBCELoss.weight = Weight
			Optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				OutputImg = Model(InputImg)
				BatchBCELoss = CriterionBCELoss(OutputImg, Label)
				BatchLoss = BatchBCELoss
				BatchLoss.backward()
				Optimizer.step()
				BCELoss += BatchBCELoss.item()
		AveBCELoss = (BCELoss * BatchSize) / TrainDataset.__len__()
		# print(22222, BCELoss, TrainDataset.__len__(), AveBCELoss)
		print('\tTrain_AveBCELoss:{0:04f}'.format(float(AveBCELoss)))
		logging.warning('\tTrain_AveBCELoss:{0:04f}'.format(float(AveBCELoss)))
		# End(SaveFolder+' Epoch')
		End('Epoch')

		# %% Validation
		if Epoch % ValidPerEpoch == 0 or Epoch == 1:
			Model.eval()
			torch.cuda.empty_cache()
			BCELoss = 0
			print('\tValidate:', end='>>', flush=True)
			for Iter, (InputImgs, Label, TMImg, SampleName) in enumerate(ValDataLoader):
				print(Iter, end=' ', flush=True)
				InputImg = torch.cat(InputImgs, dim=1)
				InputImg = InputImg.float().to(Device)
				Label = Label.float().to(Device)
				Weight = Label * (BCELossWeightCoefficient - 1) + 1
				CriterionBCELoss.weight = Weight
				with torch.set_grad_enabled(False):
					OutputImg = Model(InputImg)
					BatchBCELoss = CriterionBCELoss(OutputImg, Label)
					BCELoss += BatchBCELoss.item()
			AveBCELoss = (BCELoss * BatchSize) / ValDataset.__len__()
			# print(44444, BCELoss, ValDataset.__len__(), AveBCELoss)
			print('\t\t\t\tValidat_AveBCELoss:{0:04f}'.format(AveBCELoss))
			logging.warning('\t\tValidate_AveBCELoss:{0:04f}'.format(AveBCELoss))

		# %% Saving
		# if Epoch in SaveEpoch:
		if (Epoch%500) == 0:
			torch.save(Model.state_dict(), os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
			print("Save path:", os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
		LrScheduler.step()

	log = logging.getLogger()
	for hdlr in log.handlers[:]:
		if isinstance(hdlr, logging.FileHandler):
			log.removeHandler(hdlr)

if __name__ == '__main__':
	torch.backends.cudnn.benchmark = True
	Widths = [2]        # seting contour width of labels
	for Width in Widths:
		SaveFolder = '3PotData_Input2(4096)_IF8_Epoch3000_Dilation2_kernel5_BigKernel4_SmlKernel4'
		Train(SaveFolder, Width)


