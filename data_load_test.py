from __future__ import absolute_import, division, print_function

import tensorflow as tf 
import numpy as np 
import h5py
import train_generator as geny_tr

datapath ="/media/tkal976/Transcend/Tharindu/new_eye_data/7_1.h5"
g = geny_tr.train_generator(1)
ds = tf.data.Dataset.from_generator(g, output_types=((tf.uint8, tf.uint8)))
print("printing DS")
print(ds)

value = ds.make_one_shot_iterator().get_next()

sess = tf.Session()

# Example on how to read elements
while True:
	try:
		# print("trying")
		data = sess.run(value)
		print(data[0].shape)
		print(data[1])
	except tf.errors.OutOfRangeError:
		print('done.')
		break