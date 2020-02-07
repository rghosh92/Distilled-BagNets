import numpy as np
import keras
import sys
import h5py
from copy import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.utils import conv_utils
from keras import optimizers
import time
import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import h5py
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import pickle
from scipy.ndimage import zoom
from numpy import loadtxt
from copy import copy
sys.path.append('../Libraries')
from pyramidconv import *
from invariance_estimation import *
from lateral_weight_tying import *



def prune_data_random_class_balanced(x, y_orig, percentage):

	y = copy(np.argmax(y_orig,1))
	S = np.shape(y)
	uni_classes = np.unique(y)
	num_classes = uni_classes.size

	only_k = np.int(percentage * S[0] / 100)
	k_per_class = np.rint(only_k / num_classes)
	sel_samp_loc = np.empty(0)

	for class_no in range(0, num_classes):
		tmp_class = uni_classes[class_no]
		tmp_class_elm = np.where(y == tmp_class)
		tmp_class_elm = tmp_class_elm[0]

		shp = np.shape(tmp_class_elm)
		perm = np.random.permutation(np.arange(shp[0]))
		perm = np.take(tmp_class_elm, perm[0:np.int(k_per_class)])

		sel_samp_loc = np.append(sel_samp_loc, perm)

	#    Perm = np.random.permutation(np.arange(S[0]))
	#    Perm = Perm[0:only_k]

	# print(x.shape)

	# print(y.shape)

	sel_samp_loc = np.delete(sel_samp_loc, 0)
	x = x[sel_samp_loc.astype('int'), :, :]
	y_orig = y_orig[sel_samp_loc.astype('int'), :]

	return x, y_orig

def load_data(filepath):

	f = h5py.File(filepath)
	counter = 0
	arrays = {}

	for k, v in f.items():
		arrays[counter] = v
		counter = counter + 1

	if len(arrays[0].shape) == 3:
		input_index = 0
	else:
		input_index = 1

	x = np.array(arrays[input_index])

	x = x[:,:,:,None]

	# x = np.swapaxes(x, 1, 3)

	y = np.array(arrays[1-input_index])
	y = y - 1

	y = keras.utils.to_categorical(y, num_classes=10)

	return x,y





def generate_mean_pairs(Images):

	#print(Images.shape)

	N = Images.shape[0]
	vals = []
	vals_neigh = []
	vals_neigh2 = []

	if Images.shape[0] == 3 & Images.shape[1] ==3: #CNN Weights

		M = Images.shape[2]
		N = Images.shape[3]
		for i in range(M):
			for j in range(N):
				rand_triples = [[0,4,8],[0,4,2],[0,4,6],[3,4,5],[3,4,2],[3,4,8],\
								[6,4,8],[6,4,2],[6,4,1],[6,4,5],[1,4,7],[2,4,8]]
				index = np.random.randint(12)
				x = np.mod(rand_triples[index][0],3)
				y = int(rand_triples[index][0]/3)
				x_n = np.mod(rand_triples[index][1],3)
				y_n = int(rand_triples[index][1]/3)
				x_n2 = np.mod(rand_triples[index][2], 3)
				y_n2 = int(rand_triples[index][2] / 3)

				vals.append(Images[x,y,i,j])
				vals_neigh.append(Images[x_n,y_n,i,j])
				vals_neigh2.append(Images[x_n2,y_n2,i,j])

	else:
		for i in range(N):
			loc_x = int((Images.shape[1]/3.0)+ np.random.randint(Images.shape[1]/3.0))
			loc_y = int(Images.shape[2]/3.0+ np.random.randint(Images.shape[2]/3.0))
			disp_x = int(np.sign((np.random.rand()-0.5)))
			disp_y = int(np.sign((np.random.rand()-0.5)))
			disp_x2 = int(np.sign((np.random.rand()-0.5))*2)
			disp_y2 = int(np.sign((np.random.rand()-0.5))*2)
			#loc_x2 = np.random.randint(Images.shape[1])
			#loc_y2 = np.random.randint(Images.shape[2])
			#disp_x2 = disp_x*2
			#disp_y2 = disp_y*2
			vals.append(Images[i,loc_x,loc_y])
			vals_neigh.append(Images[i,loc_x+disp_x,loc_y+disp_y])

			vals_neigh2.append(Images[i,loc_x+disp_x2,loc_y+disp_y2])

	return np.array(vals), np.array(vals_neigh), np.array(vals_neigh2)




def generate_scaled_images(Images,group_sizes):

	Images_scaled_all = []

	for i in range(len(group_sizes)):
		k_sizes = [1,group_sizes[i],group_sizes[i],1]
		strides = [1,group_sizes[i],group_sizes[i],1]
		#strides = [1,1,1,1]
		rates = [1,1,1,1]
		sess = tf.Session()
		Images_patches = tf.extract_image_patches(Images,k_sizes,strides,rates,"VALID")
		sess.close()
		Images_patches = K.eval(Images_patches)
		Images_scaled_all.append(copy(np.mean(Images_patches,3)))

	return Images_scaled_all

def estimate_msns(Images,group_sizes):

	Snum = np.zeros((len(group_sizes)))
	Sden = np.zeros((len(group_sizes)))

	for channels in range(Images.shape[3]):

		if len(group_sizes) == 1 & group_sizes[0] == 1:
			Images_scaled_all = [np.expand_dims(Images[:,:,:,channels],3)]
		else:
			Images_scaled_all = generate_scaled_images(np.expand_dims(Images[:,:,:,channels],3),group_sizes)
		msns_arr = []

		for i in range(len(Images_scaled_all)):
			v,vn,vn2 = generate_mean_pairs(Images_scaled_all[i])
			Snum[i] = Snum[i] + np.sum(np.power(v-vn2,2.0))
			Sden[i] = Sden[i] + (2*np.sum(np.power(v-vn,2.0)))
			#msns_arr.append(np.mean((np.power(v-vn2,2.0)))/(2*np.mean(np.power(v-vn,2.0))))
			#print(msns_arr)
	#print(Snum.shape)
	#print(Sden.shape)
	msns_arr = 2*(np.maximum(0.5,np.minimum(1,np.float32(Snum)/np.float32(Sden)))-0.5)

	return msns_arr


def swap_locs(image_patches):

	i = np.random.randint(image_patches.shape[0])
	j = np.random.randint(image_patches.shape[1])

	i2 = np.random.randint(image_patches.shape[0])
	j2 = np.random.randint(image_patches.shape[1])

	temp = image_patches[i, j,:]
	image_patches[i, j,:] = image_patches[i2, j2,:]
	image_patches[i2, j2,:] = temp

	return image_patches

def patches_to_image(image_patches,level):

	image = np.zeros((image_patches.shape[0]*level,image_patches.shape[1]*level))

	for i in range(image_patches.shape[0]):
		for j in range(image_patches.shape[1]):
			image[i*level:(i+1)*level,j*level:(j+1)*level] = np.reshape(image_patches[i,j,:],(level,level))

	return np.expand_dims(image,2)



def swap_images_at_level(images,level, numswap=1, arg="VALID"):

	k_sizes = [1, level, level, 1]
	strides = [1, level, level, 1]
	# strides = [1,1,1,1]
	rates = [1, 1, 1, 1]

	images_new = np.zeros((images.shape))

	for channels in range(images.shape[3]):
		sess = tf.Session()
		Images_patches = copy(K.eval(tf.extract_image_patches(np.expand_dims(images[:,:,:,channels],3), k_sizes, strides, rates, "VALID")))
		sess.close()
		for i in range(images.shape[0]):

			for j in range(numswap):
				Images_patches[i,:,:,:] = swap_locs(Images_patches[i,:,:,:])

			images_new[i,:,:,channels] = np.squeeze(patches_to_image(Images_patches[i,:,:,:],level))

	return images_new






def block_swap_images(images,levels,numswap):

	# Swaps whole image regions at different scales, as specified by the 'levels' argument

	for i in range(len(levels)):
		images = copy(swap_images_at_level(images,levels[i],numswap))

	return images









	# Example of finding MSNS with a sample dataset
#x_train_scaled, y_train_scaled = load_data("../Datasets/mnist original/mnist_train_scaled_0.5_2.mat")
#x_train, y_train = load_data("../Datasets/mnist original/mnist_train_original.mat")
#x_train_pruned,y_train_pruned = prune_data_random_class_balanced(x_train,y_train,5)
#x_train_pruned_scaled,y_train_pruned_scaled = prune_data_random_class_balanced(x_train_scaled,y_train_scaled,5)
#print("here")
#m_orig = estimate_msns(x_train_pruned,[1,2,3,4,5])
#print('time to do stuff')
#x_dict_scale = transformed_data(x_train_pruned,'scaling',[0.1,0.5,1.5,2.0,3.0])





