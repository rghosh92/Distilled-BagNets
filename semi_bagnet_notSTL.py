import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils import *
import torch.nn.functional
import numpy as np
from copy import copy

# Will contain the
# - Backbone network
# - Training function
# - Inference function


class View_module(nn.Module):
    def __init__(self):
        super(View_module, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)



class net_fullbag(nn.Module):

    def __init__(self):
        super(net_fullbag, self).__init__()

        self.conv1 = nn.Conv2d(1,128,3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        # self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        # self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        # self.conv7 = nn.Conv2d(128, 128, 3,padding=1)

        # self.conv8 = nn.Conv2d(64, 64, 3)
        # self.conv9 = nn.Conv2d(64, 64, 3)
        # self.conv10 = nn.Conv2d(64, 64, 3)


        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(2)
        # self.mpool5 = nn.MaxPool2d(2)
        # self.mpool6 = nn.MaxPool2d(3)

        self.bnorm1 = nn.BatchNorm2d(128)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm4 = nn.BatchNorm2d(256)
        # self.bnorm5 = nn.BatchNorm2d(128)
        # self.bnorm6 = nn.BatchNorm2d(128)
        # self.bnorm7 = nn.BatchNorm2d(128)


        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

        self.prebag_network = nn.Sequential(

            self.conv1,
            self.bnorm1,
            nn.ReLU(inplace=True),
            self.mpool1,

            self.conv2,
            self.bnorm2,
            nn.ReLU(inplace=True),
            self.mpool2,

            self.conv3,
            self.bnorm3,
            nn.ReLU(inplace=True),
            # self.mpool3,
            #
            self.conv4,
            self.bnorm4,
            nn.ReLU(inplace=True),
            # self.mpool2,

            # self.conv5,
            # self.bnorm5,

            # self.conv6,
            # self.bnorm6,
            self.mpool3,

        )

        self.conv1_bn = nn.Conv2d(256, 256, 1)
        self.bnorm1_bn = nn.BatchNorm2d(256)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(256, 256, 1)
        self.bnorm2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.Conv2d(256, 10, 1)


    def forward(self, x):
        x_prebag = self.prebag_network(x)

        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return x_bagout

class net_primal_mnist(nn.Module):

    def __init__(self):
        super(net_primal_mnist, self).__init__()

        self.conv1 = nn.Conv2d(1,64,3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)

        # self.conv8 = nn.Conv2d(64, 64, 3)
        # self.conv9 = nn.Conv2d(64, 64, 3)
        # self.conv10 = nn.Conv2d(64, 64, 3)


        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2,padding=1)
        self.mpool4 = nn.MaxPool2d(4)
        # self.mpool5 = nn.MaxPool2d(2)


        self.bnorm1 = nn.BatchNorm2d(64)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.bnorm4 = nn.BatchNorm2d(128)



        self.prebag_network = nn.Sequential(

            self.conv1,
            self.bnorm1,
            nn.ReLU(inplace=True),
            self.mpool1,

            self.conv2,
            self.bnorm2,
            nn.ReLU(inplace=True),
            self.mpool2,
            #
            self.conv3,
            self.bnorm3,
            nn.ReLU(inplace=True),
            self.mpool3,
            # #
            #

        )

        self.postbag_network = nn.Sequential(

            # self.conv3,
            # self.bnorm3,
            # nn.ReLU(inplace=True),
            # self.mpool3,

            self.conv4,
            self.bnorm4,
            nn.ReLU(inplace=True),
            self.mpool4,

            # self.conv5,
            # self.bnorm5,
            # nn.ReLU(inplace=True),
            # self.mpool5,

        )

        self.bnorm_fc = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,10,1)


    def forward(self, x):
        x_prebag = self.prebag_network(x)
        x_postbag = self.postbag_network(x_prebag)

        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag]


class semi_bagnet_mnist_end_to_end(net_primal_mnist):  # Inherits modules from parent class

    def __init__(self):
        super(semi_bagnet_mnist_end_to_end,self).__init__()
        self.conv1_bn = nn.Conv2d(128,128,1)
        self.bnorm1_bn = nn.BatchNorm2d(128)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(128, 128, 1)
        self.bnorm2_bn = nn.BatchNorm2d(128)
        self.conv3_bn = nn.Conv2d(128,10,1)
        self.prebag_mask = np.ones(128)

    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        for p in self.prebag_network.parameters():
            p.requires_grad = True

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):

        x_prebag = self.prebag_network(x)
        M = self.prebag_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x_prebag.shape[0],1, x_prebag.shape[2], x_prebag.shape[3])
        x_postbag = self.postbag_network(x_prebag*M)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag*(1-M)]

    def forward(self,x):
        [x_out,x_prebag] = self.super_forward(x)
        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return [x_out,x_bagout]

class semi_bagnet_mnist(net_primal_mnist):  # Inherits modules from parent class

    def __init__(self):
        super(semi_bagnet_mnist,self).__init__()
        self.conv1_bn = nn.Conv2d(64,64,1)
        self.bnorm1_bn = nn.BatchNorm2d(64)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.bnorm2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.Conv2d(64,10,1)
        self.prebag_mask = np.ones(64)

    def clone_prebag_network(self,net_primal_instance):
        self.prebag_network_trainable = nn.Sequential(

            nn.Conv2d(1,32,3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(64,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,padding=1),

            # self.mpool3,

            # self.conv5,
            # self.bnorm5,

            # self.conv6,
            # self.bnorm6,
            # nn.MaxPool2d(2),

        )

        for i in range(len(self.prebag_network_trainable)):
            if str(self.prebag_network_trainable[i])[0:4] == 'Conv' or str(self.prebag_network_trainable[i])[0:4] == 'Batch':
                self.prebag_network_trainable[i].weight.data = copy(net_primal_instance.prebag_network[i].weight.clone().data)



    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        self.clone_prebag_network(net_primal_instance)
        for p in self.prebag_network.parameters():
            p.requires_grad = True

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):
        x_prebag = self.prebag_network_trainable(x)
        M = self.prebag_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x_prebag.shape[0],1, x_prebag.shape[2], x_prebag.shape[3])
        x_postbag = self.postbag_network(self.prebag_network(x)*M)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag*(1-M)]

    def forward(self,x):
        [x_out,x_prebag] = self.super_forward(x)
        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return [x_out,x_bagout]

class net_primal(nn.Module):

    def __init__(self,in_channels=3):
        super(net_primal, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,128,3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3,padding=1)

        # self.conv8 = nn.Conv2d(64, 64, 3)
        # self.conv9 = nn.Conv2d(64, 64, 3)
        # self.conv10 = nn.Conv2d(64, 64, 3)


        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(2,padding=1)
        self.mpool5 = nn.MaxPool2d(2)
        self.mpool6 = nn.MaxPool2d(2)


        self.bnorm1 = nn.BatchNorm2d(128)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm4 = nn.BatchNorm2d(256)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.bnorm6 = nn.BatchNorm2d(512)
        self.bnorm7 = nn.BatchNorm2d(512)


        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

        self.prebag_network = nn.Sequential(

            self.conv1,
            self.bnorm1,
            nn.ReLU(inplace=True),
            # self.mpool1,

            self.conv2,
            self.bnorm2,
            nn.ReLU(inplace=True),
            self.mpool2,

            self.conv3,
            self.bnorm3,
            nn.ReLU(inplace=True),
            # self.mpool3,
            #
            self.conv4,
            self.bnorm4,
            nn.ReLU(inplace=True),
            # self.mpool2,

            # self.conv5,
            # self.bnorm5,

            # self.conv6,
            # self.bnorm6,
            self.mpool3,

        )

        self.postbag_network = nn.Sequential(

            self.conv5,
            self.bnorm5,
            nn.ReLU(inplace=True),
            self.mpool4,

            self.conv6,
            self.bnorm6,
            nn.ReLU(inplace=True),
            self.mpool5,

            self.conv7,
            self.bnorm7,
            nn.ReLU(inplace=True),
            self.mpool6,
            # self.conv9,
            # self.bnorm9,
            # # self.mpool5,
            #
            # self.conv10,
            # self.bnorm10,
            # self.mpool5,
        )

        self.bnorm_fc = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10,1)


    def forward(self, x):
        x_prebag = self.prebag_network(x)
        x_postbag = self.postbag_network(x_prebag)

        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag]

class semi_bagnet(net_primal):  # Inherits modules from parent class

    def __init__(self,in_channels=3):
        super(semi_bagnet,self).__init__()
        self.in_channels = in_channels
        self.conv1_bn = nn.Conv2d(256,256,1)
        self.bnorm1_bn = nn.BatchNorm2d(256)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(256, 256, 1)
        self.bnorm2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.Conv2d(256,10,1)
        self.prebag_mask = np.ones(256)

    def clone_prebag_network(self,net_primal_instance):
        self.prebag_network_trainable = nn.Sequential(

            nn.Conv2d(self.in_channels,64,3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # self.mpool3,
            #
            nn.Conv2d(128,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # self.mpool2,

            # self.conv5,
            # self.bnorm5,

            # self.conv6,
            # self.bnorm6,
            nn.MaxPool2d(2),

        )

        for i in range(len(self.prebag_network_trainable)):
            if str(self.prebag_network_trainable[i])[0:4] == 'Conv' or str(self.prebag_network_trainable[i])[0:4] == 'Batch':
                self.prebag_network_trainable[i].weight.data = copy(net_primal_instance.prebag_network[i].weight.clone().data)



    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        self.clone_prebag_network(net_primal_instance)
        for p in self.prebag_network.parameters():
            p.requires_grad = False

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):
        x_prebag = self.prebag_network_trainable(x)
        M = self.prebag_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x_prebag.shape[0],1, x_prebag.shape[2], x_prebag.shape[3])
        x_postbag = self.postbag_network(self.prebag_network(x)*M)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag*(1-M)]

    def forward(self,x):
        [x_out,x_prebag] = self.super_forward(x)
        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return [x_out,x_bagout]

class semi_bagnet_end_to_end(net_primal):  # Inherits modules from parent class

    def __init__(self,in_channels=3):
        super(semi_bagnet_end_to_end,self).__init__()
        self.in_channels = in_channels
        self.conv1_bn = nn.Conv2d(256,256,1)
        self.bnorm1_bn = nn.BatchNorm2d(256)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(256, 256, 1)
        self.bnorm2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.Conv2d(256,10,1)
        self.prebag_mask = np.ones(256)

    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        for p in self.prebag_network.parameters():
            p.requires_grad = True

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):

        x_prebag = self.prebag_network(x)
        M = self.prebag_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x_prebag.shape[0],1, x_prebag.shape[2], x_prebag.shape[3])
        x_postbag = self.postbag_network(x_prebag*M)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)

        return [x,x_prebag*(1-M)]

    def forward(self,x):

        [x_out,x_prebag] = self.super_forward(x)
        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return [x_out,x_bagout]




class net_primal_recon(nn.Module):

    def __init__(self,in_channels=3):
        super(net_primal_recon, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,128,3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_mid_loc = nn.Conv2d(256, 128, 3, padding=1)
        self.conv_mid_glob = nn.Conv2d(256,128,3,padding=1)
        self.conv5 = nn.Conv2d(128, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3,padding=1)

        # self.conv8 = nn.Conv2d(64, 64, 3)
        # self.conv9 = nn.Conv2d(64, 64, 3)
        # self.conv10 = nn.Conv2d(64, 64, 3)


        self.mpool1 = nn.MaxPool2d(2)
        self.mpool2 = nn.MaxPool2d(2)
        self.mpool3 = nn.MaxPool2d(2)
        self.mpool4 = nn.MaxPool2d(2)
        self.mpool5 = nn.MaxPool2d(2)
        self.mpool6 = nn.MaxPool2d(3)

        self.bnorm1 = nn.BatchNorm2d(128)
        self.bnorm2 = nn.BatchNorm2d(256)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.bnorm_midglob = nn.BatchNorm2d(128)
        self.bnorm_midloc = nn.BatchNorm2d(128)
        self.bnorm5 = nn.BatchNorm2d(512)
        self.bnorm6 = nn.BatchNorm2d(512)
        self.bnorm7 = nn.BatchNorm2d(512)


        self.drop1 = nn.Dropout()
        self.drop2 = nn.Dropout()

        self.prebag_network = nn.Sequential(

            self.conv1,
            self.bnorm1,
            nn.ReLU(inplace=True),
            self.mpool1,

            self.conv2,
            self.bnorm2,
            nn.ReLU(inplace=True),
            self.mpool2,

            self.conv3,
            self.bnorm3,
            nn.ReLU(inplace=True),
            # self.mpool3,
            #

        )

        self.midglob_network = nn.Sequential(

            self.conv_mid_glob,
            self.bnorm_midglob,
            nn.ReLU(inplace=True),
            self.mpool3,
        )

        self.midloc_network = nn.Sequential(

            self.conv_mid_loc,
            self.bnorm_midloc,
            nn.ReLU(inplace=True),
            self.mpool3,
        )

        self.postbag_network = nn.Sequential(

            self.conv5,
            self.bnorm5,
            nn.ReLU(inplace=True),
            self.mpool4,

            self.conv6,
            self.bnorm6,
            nn.ReLU(inplace=True),
            self.mpool5,

            self.conv7,
            self.bnorm7,
            nn.ReLU(inplace=True),
            self.mpool6,
            # self.conv9,
            # self.bnorm9,
            # # self.mpool5,
            #
            # self.conv10,
            # self.bnorm10,
            # self.mpool5,
        )

        self.bnorm_fc = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,10,1)


    def forward(self, x):
        x_prebag = self.prebag_network(x)
        x_postbag = self.postbag_network(x_prebag)

        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)
        return [x,x_prebag]


class three_way_semibagnet_end_to_end(net_primal):

    def __init__(self,in_channels=3):
        super(three_way_semibagnet_end_to_end,self).__init__()
        self.in_channels = in_channels
        self.conv1_bn = nn.Conv2d(256,256,1)
        self.bnorm1_bn = nn.BatchNorm2d(256)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(256, 256, 1)
        self.bnorm2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.Conv2d(256,10,1)
        self.prebag_mask = np.ones(256)


        self.postbag_network2 = nn.Sequential(

            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,padding=1),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            View_module(),
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512,10,1)

        )

    def set_network2_grad(self,value):

        for p in self.parameters():
            p.requires_grad = not(value)

        for p in self.postbag_network2.parameters():
            p.requires_grad = value


    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        for p in self.prebag_network.parameters():
            p.requires_grad = True

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):

        x_prebag = self.prebag_network(x)
        M = self.prebag_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x_prebag.shape[0],1, x_prebag.shape[2], x_prebag.shape[3])

        x_postbag = self.postbag_network(x_prebag*M)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)

        return [x,x_prebag*(1-M)]

    def forward(self,x):

        [x_out,x_prebag] = self.super_forward(x)
        x_bagout2 = self.postbag_network2(x_prebag)
        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_prebag)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        return [x_out,x_bagout,x_bagout2]




class semi_bagnet_reconpath_end_to_end(net_primal_recon):  # Inherits modules from parent class

    def __init__(self,in_channels=3):
        super(semi_bagnet_reconpath_end_to_end,self).__init__()
        self.in_channels = in_channels
        self.recon_conv1 = nn.Conv2d(128,256,3,padding=1)
        self.conv1_bn = nn.Conv2d(128,256,1)
        self.bnorm1_bn = nn.BatchNorm2d(256)
        # self.conv2_bn = nn.Conv2d(64, 64, 1)
        self.conv2_bn = nn.Conv2d(256, 256, 1)
        self.bnorm2_bn = nn.BatchNorm2d(256)
        self.conv3_bn = nn.Conv2d(256,10,1)


    def copy_weights_from_parent(self,net_primal_instance):

        self.prebag_network = net_primal_instance.prebag_network
        self.midloc_network = net_primal_instance.midloc_network
        self.midglob_network = net_primal_instance.midglob_network

        self.recon_conv1.weight.requires_grad = False


        for p in self.prebag_network.parameters():
            p.requires_grad = True

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def change_grad_mode(self):
        
        for p in self.parameters():
            p.requires_grad = not(p.requires_grad)

    def glob_grad_mode(self):

        for p in self.midloc_network.parameters():
            p.requires_grad = False

        for p in self.prebag_network.parameters():
            p.requires_grad = False

    def undo_glob_grad_mode(self):

        for p in self.midloc_network.parameters():
            p.requires_grad = True

        for p in self.prebag_network.parameters():
            p.requires_grad = True



    def super_forward(self, x):

        Up = nn.UpsamplingBilinear2d(scale_factor=2)

        x_prebag = self.prebag_network(x)
        x_midloc = self.midloc_network(x_prebag)
        prebag_recon = Up(F.relu(self.recon_conv1(x_midloc)))
        # glob_recon = self.recon_conv1(x_midloc)

        x_bag1 = F.relu(self.bnorm1_bn(self.conv1_bn(x_midloc)))
        x_bag1 = F.relu(self.bnorm2_bn(self.conv2_bn(x_bag1)))
        x_bagout = self.conv3_bn(x_bag1)

        x_midglob = self.midglob_network(x_prebag-prebag_recon)
        x_postbag = self.postbag_network(x_midglob)
        x = x_postbag.view(x_postbag.size(0), -1)
        x = F.relu(self.bnorm_fc(self.fc1(x)))
        x = self.fc2(x)

        # return [x,x_bagout,glob_recon,x_midglob]
        #
        return [x, x_bagout, x_prebag, prebag_recon]

    def forward(self,x):

        [x_out,x_bagout,x_prebag,prebag_recon] = self.super_forward(x)

        return [x_out,x_bagout,x_prebag,prebag_recon]







class globalpool_net_mnist(net_primal_mnist):  # Inherits modules from parent class

    def __init__(self):
        super(globalpool_net_mnist,self).__init__()
        self.linear1 = nn.Linear(64,10)
        self.bnorm_input = nn.BatchNorm1d(64)
        # self.bnorm1_bn = nn.BatchNorm2d(64)
        # self.conv2_bn = nn.Conv2d(64,10,1)


    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        for p in self.prebag_network.parameters():
            p.requires_grad = False

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):

        x_prebag = self.prebag_network(x)
        x_prebag_pooled = F.avg_pool2d(x_prebag,(x_prebag.shape[2],x_prebag.shape[3]))

        return x_prebag_pooled

    def forward(self,x):
        x_prebag_pooled = self.super_forward(x)
        x = x_prebag_pooled.view(x_prebag_pooled.size(0), -1)
        x = self.bnorm_input(x)
        x_out = self.linear1(x)

        return x_out




class globalpool_net(net_primal):  # Inherits modules from parent class

    def __init__(self):
        super(globalpool_net,self).__init__()
        self.linear1 = nn.Linear(128,10)
        self.bnorm_input = nn.BatchNorm1d(128)
        # self.bnorm1_bn = nn.BatchNorm2d(64)
        # self.conv2_bn = nn.Conv2d(64,10,1)


    def copy_weights_from_parent(self,net_primal_instance):
        self.prebag_network = net_primal_instance.prebag_network
        for p in self.prebag_network.parameters():
            p.requires_grad = False

        self.postbag_network = net_primal_instance.postbag_network
        self.fc1 = net_primal_instance.fc1
        self.fc2 = net_primal_instance.fc2
        self.bnorm_fc = net_primal_instance.bnorm_fc

    def super_forward(self, x):

        x_prebag = self.prebag_network(x)
        x_prebag_pooled = F.avg_pool2d(x_prebag,(x_prebag.shape[2],x_prebag.shape[3]))

        return x_prebag_pooled

    def forward(self,x):
        x_prebag_pooled = self.super_forward(x)
        x = x_prebag_pooled.view(x_prebag_pooled.size(0), -1)
        x = self.bnorm_input(x)
        x_out = self.linear1(x)

        return x_out


