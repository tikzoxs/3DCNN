import h5py
import tensorflow as tf
import numpy as np
import os

class test_generator:
	def __init__(self, batch_size):
		# self.filepath = "/hpc/tkal976/aeye/test"
		self.filepath = "/media/tkal976/Transcend/Tharindu/aeye_data/test"
		self.batch_size = batch_size
		self.channels = 1

	def __call__(self):
		count = 0
		while count < 1000:
			count = count + 1
			for datapath in os.listdir(self.filepath):
				with h5py.File(self.filepath + '/' + datapath, 'r') as h5f:
					X_dset = h5f['X']
					Y_dset = h5f['Y']
					print(X_dset.shape[0])
					# for j in range(100):
					image_3d = np.zeros((self.batch_size, 128, 192, 64, self.channels))
					label_3d = np.zeros((self.batch_size, 2))
					for i in range(int(((X_dset.shape[0] - X_dset.shape[0] % self.batch_size)/self.batch_size))):
						for j in range(self.batch_size):
							image_3d[j,:,:,:,0] = X_dset[i * self.batch_size + j]
							label_3d[j,0] = Y_dset[i * self.batch_size + j,:,4] 
							label_3d[j,1] = Y_dset[i * self.batch_size + j,:,5] or Y_dset[i * self.batch_size + j,:,6]
						yield (image_3d, label_3d)# np.argmax(Y_dset[i,:,4:7])
