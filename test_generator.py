import h5py
import tensorflow as tf
import numpy as np

class test_generator:
	def __init__(self, batch_size):
		self.file = "/media/tkal976/Transcend/Tharindu/new_eye_data/5_1.h5"
		self.batch_size = batch_size
		self.time_sequence = 1

	def __call__(self):
		with h5py.File(self.file, 'r') as h5f:
			X_dset = h5f['X']
			Y_dset = h5f['Y']
			print(X_dset.shape[0])
			# for j in range(100):
			image_3d = np.zeros((self.batch_size, self.time_sequence, 256, 384, 64))
			label_3d = np.zeros((self.batch_size, 4))
			count = 0
			while count < 1000:
				count = count + 1
				for i in range(int(((X_dset.shape[0] - X_dset.shape[0] % self.batch_size)/self.batch_size))):
					for j in range(self.batch_size):
						image_3d[j,0,:,:,:] = X_dset[i * self.batch_size + j]
						label_3d[j,:] = Y_dset[i * self.batch_size + j,:,0:4]
					yield (image_3d, label_3d)# np.argmax(Y_dset[i,:,4:7])
