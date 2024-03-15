import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import scipy.signal as signal


def reflection(im):
	mr, mg, mb = torch.split(im, 1, dim=1)
	r = mr / (mr + mg + mb + 0.0001)
	g = mg / (mr + mg + mb + 0.0001)
	b = mb / (mr + mg + mb + 0.0001)
	return torch.cat([r,g,b],dim=1)

def luminance(s):
	return ((s[:,0,:,:]+ s[:,1,:,:] +s[:,2,:,:])).unsqueeze(1)

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight) / 255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2, 0, 1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	denoise_net = model.denoise_net().cuda()
	light_net = model.light_net().cuda()

	# denoise_net.load_state_dict(torch.load('snapshots_denoise/Epoch24.pth')['net'])
	light_net.load_state_dict(torch.load('snapshots_light/pre_train_1.pth'))
	start = time.time()
	# img_re_no = reflection(data_lowlight)
	# img_lu_no = luminance(data_lowlight)

	# data_denoise = blur_layer(data_lowlight)
	# data_lowlight = torch.where(img_lumin/3 < 0.1, data_denoise, data_lowlight)

	img_normlight_noise, _, _ = light_net(data_lowlight)
	# img_final = denoise_net(img_normlight_noise)

	# enhanced_image = img_lu_de * img_re_de


	end_time = (time.time() - start)

	print(end_time)
	image_path = image_path.replace('test_data','result')
	# image_path = image_path.replace('LSRW', 'result_LSRW')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(img_normlight_noise, result_path)
##################################
	# image_path = image_path.replace('result', 'wo')
	# # image_path = image_path.replace('low', 'result_lol')
	# result_path = image_path
	# if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
	# 	os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
	#
	# torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
		# filePath = 'data/LSRW/'
	
		file_list = os.listdir(filePath)
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*")
			#test_list = glob.glob(filePath + file_name + "/66.jpg")
			for i in range(len(test_list)):
				test_list[i] = test_list[i].replace("\\", "/")
			for image in test_list:
				# image = image

		# test_list = glob.glob(filePath+"/*")
		# 	#test_list = glob.glob(filePath + file_name + "/66.jpg")
		# for i in range(len(test_list)):
		# 	test_list[i] = test_list[i].replace("\\", "/")
		# for image in test_list:
		# 		# image = image


				print(image)
				lowlight(image)

		



