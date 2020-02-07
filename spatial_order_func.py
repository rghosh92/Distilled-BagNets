import numpy as np
import scipy.misc
import torch.nn.functional as F
from PIL import Image as Img

def spatial_order_single(Image):

    epsilon = 0.00000001

    Image = np.double(Image)
    M = Image.shape[0]
    N = Image.shape[1]
    I1_00 = Image[0:M-1,0:N-1]
    I1_11 = Image[1:M,1:N]
    I1_01 = Image[0:M-1,1:N]
    I1_10 = Image[1:M,0:N-1]
    S1 = ((I1_00-I1_11)**2+(I1_00-I1_01)**2+(I1_00-I1_10)**2)/3
    S1 = np.mean(S1) + epsilon

    I2_00 = Image[0:M - 2, 0:N - 2]
    I2_22 = Image[2:M, 2:N]
    I2_02 = Image[0:M - 2, 2:N]
    I2_20 = Image[2:M, 0:N - 2]
    S2 = ((I2_00 - I2_22) ** 2 + (I2_00 - I2_02) ** 2 + (I2_00 - I2_02) ** 2) / 3
    S2 = np.mean(S2) + epsilon

    sp_order = (S2/S1)-1

    return sp_order

def spatial_order_single_multiscale(Image,scales):

    sp_orders = []
    M = Image.shape[0]
    N = Image.shape[1]

    for i in range(len(scales)):
        Image_temp = np.array(Img.fromarray(Image).resize((int(M/scales[i]),int(N/scales[i]))))

        # Image_temp = scipy.misc.imresize(Image,(int(M/scales[i]),int(N/scales[i])))
        sp_orders.append(spatial_order_single(Image_temp))

    return sp_orders

def spatial_order_cnn_maps_multiscale(fmaps,scales):

    sp_orders = [[] for x in range(fmaps.shape[1])]

    for map in range(fmaps.shape[1]):
        for instances in range(fmaps.shape[0]):
            sp_orders[map].append(spatial_order_single_multiscale(fmaps[instances,map,:,:],scales))


    sp_orders = np.squeeze(np.array(sp_orders))

    return np.mean(sp_orders,1)









