import os
import pickle
import math
import numpy as np
import scipy.misc
# import cv2

import torch
import torchvision
from torch.utils import data
from torch.autograd import Variable
from PIL import Image

import matplotlib.pyplot as plt
import global_settings



def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float().cuda()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self, img_path, model, selected_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrained_model = model

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img, resize_im=False)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
        print(input.shape)

        x = input
        for index, layer in enumerate(self.pretrained_model.modules()):
            if layer._get_name() != 'ScaleConv_steering':
            # if layer._get_name() != 'ConvolutionLayer':
                continue
            print('index: ', index, ',', ' layer:', layer._get_name())
            x = layer(x)
            if index == self.selected_layer:
                return x

    def get_single_feature(self):
        features = self.get_feature()
        print(features.shape)

        feature = features[:, 0, :, :]
        print(feature.shape)

        feature = feature.view(feature.shape[1], feature.shape[2])
        print(feature.shape)

        return feature

    def get_kernel_map(self, features):
        feature_map = []
        img_num, kernel_num, kernel_rows, kerner_cols = features.shape
        map_size = int(math.sqrt(kernel_num)) + 1

        for image_feature in features:
            print(image_feature.shape)
            for idx, feature in enumerate(image_feature):
                for _ in range(map_size):
                    for _ in range(map_size):
                        ax = plt.subplot(map_size, map_size, idx+1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        pics = plt.imshow(feature.cpu().detach().numpy(), cmap='gray')
            plt.savefig('./example/kernel_map_steerable.png')
        plt.show()

    def get_and_save_all_feature(self):
        kernel_map = []
        features = self.get_feature()
        print(features.shape)
        self.get_kernel_map(features)

    def save_feature_to_img(self):
        # to numpy
        feature = self.get_single_feature()
        feature = feature.cpu().detach().numpy()
        # # use sigmod to [0,1]
        # feature = 1.0/(1+np.exp(-1*feature))
        #
        # # to [0,255]
        # feature=np.round(feature*255)
        # print(feature[0])
        plt.imsave('./img.jpg', feature)

from matplotlib import colors

from scipy.ndimage import gaussian_filter
from copy import copy

# global_SIGMA = 0.5

class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels, transform=None,distractor=False,smoothing=False):
        # 'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
        self.inputs = inputs
        self.smoothing = smoothing

        self.transform = transform
        self.distractor = distractor
        self.dataset_name = dataset_name
        self.color_names = ['red','blue','green','yellow','violet','indigo','orange','purple','cyan','black']
        self.color_class = []

        for i in range(10):
            self.color_class.append(colors.to_rgb(self.color_names[i]))



    def __len__(self):
        # 'Denotes the total number of samples'
        return self.inputs.shape[0]

    def cutout(self, img, x, y, size):
        size = int(size/2)
        lx = np.maximum(0,x-size)
        rx = np.minimum(img.shape[0],x+size)
        ly = np.maximum(0, y - size)
        ry = np.minimum(img.shape[1], y + size)
        img[lx:rx,ly:ry,:] = 0
        return img

    def add_class_distractor(self,image,size,color):

        x_tl = int(size + ((image.shape[0] - 2 * size) * np.random.rand()) - 1)
        y_tl = int(size + ((image.shape[1] - 2 * size) * np.random.rand()) - 1)

        image[x_tl:x_tl + size, y_tl:y_tl + size, 0] = color[0] * np.ones((size, size))
        image[x_tl:x_tl + size, y_tl:y_tl + size, 1] = color[1] * np.ones((size, size))
        image[x_tl:x_tl + size, y_tl:y_tl + size, 2] = color[2] * np.ones((size, size))

        return image


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        img = self.inputs[index]

        if self.dataset_name == 'STL10' or self.dataset_name == 'TINY_IMAGENET':
            img = np.transpose(img, [1, 2, 0])

        # Cutout module begins
        # xcm = int(np.random.rand()*95)
        # ycm = int(np.random.rand()*95)
        # img = self.cutout(img,xcm,ycm,24)
        #Cutout module ends

        # print(np.max(img),np.min(img))

        img = Image.fromarray(np.uint8(img*255))


        # img = np.float32(scipy.misc.imresize(img, 2.0))
        # Optional:
        # img = img / np.max(img)

        # if self.distractor is True and self.labels[index] < 3:
        #     img = self.add_class_distractor(img,1,self.color_class[int(self.labels[index])])

        # if self.smoothing:
        #     img = gaussian_filter(img,sigma=(global_settings.global_SIGMA,global_settings.global_SIGMA,0))

        if self.transform is not None:
            img = self.transform(img)


        y = int(self.labels[index])

        return img, y


def make_mean_std(x,meen,stdee):
    # for i in
    for i in range(3):
        x[:,i,:,:] = ((x[:,i,:,:] - meen[i])/stdee[i])

    return x


def load_dataset(dataset_name, val_splits, training_size):

    os.chdir('..')
    os.chdir(dataset_name)
    a = os.listdir()
    listdict = []

    for split in range(val_splits):
        listdict.append(pickle.load(open(a[split], 'rb')))

        listdict[-1]['train_data'] = np.float32(listdict[-1]['train_data'][0:training_size, :, :])
        # listdict[-1]['train_data'] = make_mean_std(listdict[-1]['train_data'], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        listdict[-1]['train_label'] = listdict[-1]['train_label'][0:training_size]

        listdict[-1]['test_data'] = np.float32(listdict[-1]['test_data'])
        # listdict[-1]['test_data'] = make_mean_std(listdict[-1]['test_data'], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


        listdict[-1]['test_label'] = np.float32(listdict[-1]['test_label'])

    os.chdir('..')
    os.chdir('semibagnets')

    return listdict