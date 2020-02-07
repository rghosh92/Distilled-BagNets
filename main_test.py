import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from copy import copy

from torch.optim.lr_scheduler import StepLR

# from ScaleSteerableInvariant_Network import *
from semi_bagnet_STL import *
# from Network import *
from spatial_order_func import *

import numpy as np
import sys, os
from utils import Dataset, load_dataset
from matplotlib.pyplot import figure


import global_settings
global_settings.init()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# This is the testbench for the
# MNIST-Scale, FMNIST-Scale and CIFAR-10-Scale datasets.
# The networks and network architecture are defiend
# within their respective libraries


def train_fullbag_network(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            bagouts = net(inputs)
            bagloss_items = []
            for bagx in range(bagouts.shape[2]):
                for bagy in range(bagouts.shape[3]):
                    bagloss_items.append(criterion(bagouts[:, :, bagx, bagy], labels))

            loss = sum(bagloss_items) / (bagouts.shape[2] * bagouts.shape[3])

            loss.backward()
            optimizer.step()
        # print('break')
    net = net.eval()
    return net



def train_network(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            [outputs, temp] = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # print('break')
    net = net.eval()
    return net

def train_semibag(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay,mask):

    net = net
    net = net.cuda()
    net = net.train()
    slow_parameters = []
    fast_parameters = []
    for name, parameter in net.named_parameters():
        if 'prebag_network' in name and'prebag_network_trainable' not in name:
            slow_parameters.append(parameter)
        else:
            fast_parameters.append(parameter)

    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD([
    {'params': slow_parameters, 'lr': init_rate},
    {'params': fast_parameters, 'lr': init_rate}], momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))


        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            [outs, bagouts] = net(inputs)

            bagloss_items = []
            for bagx in range(bagouts.shape[2]):
                for bagy in range(bagouts.shape[3]):
                    bagloss_items.append(criterion(bagouts[:,:,bagx,bagy],labels))

            total_loss = [4*sum(bagloss_items)/(bagouts.shape[2]*bagouts.shape[3]),criterion(outs, labels)]
            total_loss = sum(total_loss)
            total_loss.backward()
            optimizer.step()
        # print('break')
    net = net.eval()

    return net


def train_semibag_threeway(net, trainloader, trainloader2, testloader, listdict, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()
    slow_parameters = []
    fast_parameters = []
    for name, parameter in net.named_parameters():
        if 'prebag_network' in name and'prebag_network_trainable' not in name:
            slow_parameters.append(parameter)
        else:
            fast_parameters.append(parameter)

    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD([
    {'params': slow_parameters, 'lr': init_rate},
    {'params': fast_parameters, 'lr': init_rate}], momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))


        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            [outs, bagouts,outs2] = net(inputs)

            bagloss_items = []
            for bagx in range(bagouts.shape[2]):
                for bagy in range(bagouts.shape[3]):
                    bagloss_items.append(criterion(bagouts[:,:,bagx,bagy],labels))

            total_loss = [4*sum(bagloss_items)/(bagouts.shape[2]*bagouts.shape[3]),criterion(outs, labels)]
            total_loss = sum(total_loss)
            total_loss.backward()

            # net.set_network2_grad(False)
            optimizer.step()



    net.set_network2_grad(True)
    optimizer = optim.SGD([
    {'params': slow_parameters, 'lr': init_rate},
    {'params': fast_parameters, 'lr': init_rate}], momentum=0.9, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(int(total_epochs)):

        # torch.cuda.empty_cache()
        scheduler.step()
        # print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader2, 0):

            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            [outs, bagouts, outs2] = net(inputs)
            loss2 = criterion(outs2, labels)
            loss2.backward()
            optimizer.step()


    net = net.eval()


    return net



def train_semibag_recon(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()
    slow_parameters = []
    fast_parameters = []

    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(),lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        # torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            [outs, bagouts] = net(inputs)
            [outs, bagouts, prebag, prebagrecon] = net(inputs)
            # [outs,bagouts,glob_recon,x_midglob] = net(inputs)

            bagloss_items = []

            net.change_grad_mode()
            # recon_loss = 10*criterion_recon(glob_recon,x_midglob)
            recon_loss = 10 * criterion_recon(prebagrecon, prebag)
            recon_loss.backward(retain_graph=True)

            # optimizer.step()
            # optimizer.zero_grad()

            net.change_grad_mode()
            # net.glob_grad_mode()

            # redun_loss = -0.1*criterion_recon(glob_recon,x_midglob)
            # redun_loss.backward(retain_graph=True)
            # net.undo_glob_grad_mode()

            for bagx in range(bagouts.shape[2]):
                for bagy in range(bagouts.shape[3]):
                    bagloss_items.append(criterion(bagouts[:,:,bagx,bagy],labels))

            # cum_logits_bag = torch.sum(torch.sum(bagouts,3),2)/(bagouts.shape[2]*bagouts.shape[3])
            # final_logits = 2*cum_logits_bag + outs
            total_loss = [4*sum(bagloss_items)/(bagouts.shape[2]*bagouts.shape[3]),criterion(outs, labels)]
            total_loss = sum(total_loss)
            # total_loss = criterion(final_logits,labels)
            total_loss.backward()

            optimizer.step()


        # print('break')
    net = net.eval()

    return net





import matplotlib.pyplot as plt


def compute_order_with_mask(net,trainloader,mask):

    net = net.eval()
    count = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        # [outs,outs_prebag] = net(inputs)

        outs_prebag = net.prebag_network(inputs)

        outs_prebag = outs_prebag.detach().cpu().numpy()

        if i == 0:
            map_orders = np.zeros(outs_prebag.shape[1])
        else:
            map_orders = map_orders + spatial_order_cnn_maps_multiscale(outs_prebag, [1])
            count = count+1

        if i == 1:
            break

    fmap_orders = map_orders/count

    # plt.hist(fmap_orders,bins='auto')
    # plt.show()
    # ptile = np.percentile(fmap_orders, ptile)
    # print(ptile)

    order_high = np.sum(fmap_orders*mask.cpu().numpy())/np.sum(mask.cpu().numpy())
    mask = 1 - mask
    order_low = np.sum(fmap_orders*mask.cpu().numpy())/np.sum(mask.cpu().numpy())

    return order_high, order_low


def sort_maps_by_order(net,trainloader,ptile):

    net = net.eval()
    count = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        [outs,outs_prebag] = net(inputs)

        outs_prebag = outs_prebag.detach().cpu().numpy()

        if i == 0:
            map_orders = np.zeros(outs_prebag.shape[1])
        else:
            map_orders = map_orders + spatial_order_cnn_maps_multiscale(outs_prebag, [1])
            count = count+1
        if i == 5:
            break


    fmap_orders = map_orders/count

    # plt.hist(fmap_orders,bins='auto')
    # plt.show()
    ptile = np.percentile(fmap_orders, ptile)
    print(ptile)
    mask = fmap_orders > ptile

    mask = torch.from_numpy(np.float32(mask)).cuda()

    return mask

    #
    # ptile = np.percentile(torch.var(net.linear1.weight,0).detach().cpu().numpy(),50)
    # mask = torch.var(net.linear1.weight, 0) < ptile


def test_network(net, testloader, test_labels,out_size=1):

    net = net.eval()
    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        # if out_size == 1:
        #     outputs = net(images)
        # else:
        [outputs, temp] = net(images)

        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        # torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy

def test_semibag_network(net, testloader, test_labels):

    net = net.eval()
    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        # [outs,bagouts,noneed,noneedatall] = net(images)
        [outs,bagouts,outs2] = net(images)

        bags = bagouts.view(bagouts.shape[0],bagouts.shape[1],bagouts.shape[2]*bagouts.shape[3])
        bags = torch.mean(bags,2)

        _, predicted = torch.max(outs+bags+2*outs2, 1)

        correct = correct + torch.sum(predicted == labels)
        # torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy


def test_semibag_network_threeway(net, vanilla_net, testloader, test_labels):

    net = net.eval()
    vanilla_net = vanilla_net.eval()
    correct_bags = torch.tensor(0)
    correct_outs = torch.tensor(0)
    correct_outs2 = torch.tensor(0)
    correct_vanilla = torch.tensor(0)

    list_bags = []
    list_outs = []
    list_outs2 = []
    list_vanilla = []

    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        # [outs,bagouts,noneed,noneedatall] = net(images)
        [outs,bagouts,outs2] = net(images)

        bags = bagouts.view(bagouts.shape[0],bagouts.shape[1],bagouts.shape[2]*bagouts.shape[3])
        bags = torch.mean(bags,2)

        _, predicted = torch.max(bags, 1)
        correct_bags = correct_bags + torch.sum(predicted == labels)
        list_bags.append(bags.cpu().detach().numpy())

        _, predicted = torch.max(outs, 1)
        correct_outs = correct_outs + torch.sum(predicted == labels)
        list_outs.append(outs.cpu().detach().numpy())


        _, predicted = torch.max(outs2, 1)
        correct_outs2 = correct_outs2 + torch.sum(predicted == labels)
        list_outs2.append(outs2.cpu().detach().numpy())

        [outs,temp,tempo] = vanilla_net(images)
        correct_vanilla = correct_vanilla + torch.sum(predicted == labels)

        list_vanilla.append(outs.cpu().detach().numpy())
        #
        # if i == 20:
        #     break

        # torch.cuda.empty_cache()

    accuracy_bags = float(correct_bags)/float(total)
    accuracy_outs = float(correct_outs)/float(total)
    accuracy_outs2 = float(correct_outs2)/float(total)
    accuracy_vanilla = float(correct_vanilla)/float(total)

    # print(accuracy_outs,accuracy_bags,accuracy_outs2)
    return accuracy_outs,accuracy_bags,accuracy_outs2
    # return np.concatenate(list_bags,axis=0), np.concatenate(list_outs,axis=0), np.concatenate(list_outs2,axis=0),np.concatenate(list_vanilla,axis=0)


import math



def test_semibag_network_interaction(net,vanilla_net, testloader, test_labels,siz):

    net = net.eval()

    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    corr_outs = 0
    corr_bags = 0
    corr_outs2 = 0
    corr_vanilla = 0

    count = 0

    for i in range(int(len(test_labels) / testloader.batch_size)):

        total = 0
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        # [outs,bagouts,noneed,noneedatall] = net(images)


        total_mask = torch.ones_like(images)
        diffsum_outs = 0
        diffsum_bags = 0
        diffsum_outs2 = 0
        diffsum_vanilla = 0


        corrsum_outs = 0
        corrsum_bags = 0
        corrsum_outs2 = 0
        corrsum_vanilla = 0

        for ii in range(images.shape[0]):
            [outs, bagouts, outs2] = net(images[ii,:,:,:].unsqueeze(0))
            [outs_vanilla, other] = vanilla_net(images[ii,:,:,:].unsqueeze(0))


            bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2] * bagouts.shape[3])
            bags_orig = torch.mean(bags, 2)

            x_curr = 0
            y_curr = 0
            all_dummy = images[ii,:,:,:].detach().clone().cpu().numpy()
            while 1:
                if x_curr+siz>images.shape[2] or y_curr+siz>=images.shape[3]:
                    break

                while 1:

                    dummy = images[ii,:,:,:].detach().clone().cpu().numpy()
                    dummy[0,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(dummy[0,x_curr:x_curr+siz,y_curr:y_curr+siz])
                    dummy[1,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(dummy[1,x_curr:x_curr+siz,y_curr:y_curr+siz])
                    dummy[2,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(dummy[2,x_curr:x_curr+siz,y_curr:y_curr+siz])

                    all_dummy[0,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(all_dummy[0,x_curr:x_curr+siz,y_curr:y_curr+siz])
                    all_dummy[1,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(all_dummy[1,x_curr:x_curr+siz,y_curr:y_curr+siz])
                    all_dummy[2,x_curr:x_curr+siz,y_curr:y_curr+siz] = np.mean(all_dummy[2,x_curr:x_curr+siz,y_curr:y_curr+siz])


                    [outs_dum, bagouts, outs2_dum] = net(torch.from_numpy(dummy).cuda().unsqueeze(0))

                    bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2] * bagouts.shape[3])
                    bags_dum = torch.mean(bags, 2)

                    diffsum_bags = diffsum_bags + bags_orig-bags_dum
                    diffsum_outs = diffsum_outs + outs-outs_dum
                    diffsum_outs2 = diffsum_outs2 + outs2-outs2_dum

                    [outs_dum, other] = vanilla_net(torch.from_numpy(dummy).cuda().unsqueeze(0))
                    diffsum_vanilla = diffsum_vanilla + outs_vanilla-outs_dum


                    y_curr = y_curr + (2*siz)
                    if y_curr + siz >= images.shape[3]:
                        break

                x_curr = x_curr + (2*siz)
                if x_curr +siz >= images.shape[2]:
                    break

            [outs_dum, bagouts, outs2_dum] = net(torch.from_numpy(all_dummy).cuda().unsqueeze(0))
            bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2] * bagouts.shape[3])
            bags_dum = torch.mean(bags, 2)

            diffall_bags = bags_orig - bags_dum
            diffall_outs = outs - outs_dum
            diffall_outs2 = outs2 - outs2_dum

            [vanilla_dum, other] = vanilla_net(torch.from_numpy(all_dummy).cuda().unsqueeze(0))
            diffall_vanilla =  outs_vanilla - vanilla_dum


            [one, temp] = scipy.stats.pearsonr(diffsum_outs.cpu().numpy().squeeze(),diffall_outs.cpu().numpy().squeeze())
            [two, temp] = scipy.stats.pearsonr(diffsum_bags.cpu().squeeze(), diffall_bags.cpu().squeeze())
            [three, temp] = scipy.stats.pearsonr(diffsum_outs2.detach().cpu().squeeze(),diffall_outs2.detach().cpu().squeeze())
            [four, temp] = scipy.stats.pearsonr(diffsum_vanilla.detach().cpu().squeeze(),diffall_vanilla.detach().cpu().squeeze())


            if math.isnan(one) or math.isnan(two) or math.isnan(three):
                continue
            else:
                corrsum_outs = corrsum_outs + one
                corrsum_bags = corrsum_bags + two
                corrsum_outs2 = corrsum_outs2 + three
                corrsum_vanilla = corrsum_vanilla + four
                total = total+1

        corr_outs = corr_outs + (corrsum_outs/(total))
        corr_bags = corr_bags + (corrsum_bags/total)
        corr_outs2 = corr_outs2 + (corrsum_outs2/(total))
        corr_vanilla = corr_vanilla + (corrsum_vanilla/(total))


        count = count+1

        if i == 20:
            break

        # torch.cuda.empty_cache()

    corr_outs = corr_outs/count
    corr_bags = corr_bags / count
    corr_outs2 = corr_outs2 / count
    corr_vanilla = corr_vanilla / count


    return corr_outs,corr_bags,corr_outs2,corr_vanilla


def get_premodel_semibag_threeway(net):

    part_model = nn.Sequential(

                net.conv1,
                net.bnorm1,
                nn.ReLU(inplace=True),
                net.mpool1,

                net.conv2,
                net.bnorm2,
                nn.ReLU(inplace=True),
                net.mpool2,

                net.conv3,
                net.bnorm3,
                nn.ReLU(inplace=True),
                # self.mpool3,
                #
                net.conv4,
                net.bnorm4,
                nn.ReLU(inplace=True),
                # self.mpool2,

                # self.conv5,
                # self.bnorm5,

                # self.conv6,
                # self.bnorm6,
                net.mpool3,

            )
    part_model = part_model.eval()
    return part_model



def visualize_semibag_network_threeway(net, testloader, test_labels):

    net = net.eval()
    part_net = get_premodel_semibag_threeway(net)

    correct_bags = torch.tensor(0)
    correct_outs = torch.tensor(0)
    correct_outs2 = torch.tensor(0)

    images_bags = []
    images_outs = []


    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for j in range(int(len(test_labels) / testloader.batch_size)):
        dataiter.next()
        dataiter.next()
        dataiter.next()

        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        # [outs,bagouts,noneed,noneedatall] = net(images)
        [outs,bagouts,outs2] = net(images)
        # mid_outs = part_net(images)


        [val,inds] = torch.max(mid_outs,0)

        for i in range(30):

            [temp,index] = torch.max(val[i,:,:].view(-1),0)
            flat_inds = inds[i,:,:].view(-1)
            img = images[flat_inds[index].cpu().numpy(), :, :, :].cpu().numpy().squeeze().transpose(1, 2, 0)

            index = index.cpu().numpy()
            x_loc = int(np.floor(index/12))
            y_loc = int(np.mod(index,12))

            if net.prebag_mask[i] == 0:
                # images_outs
                images_outs.append(img[np.maximum(0,(8*x_loc)-12):np.minimum((8*x_loc)+12,95),np.maximum(0,(8*y_loc)-12):np.minimum((8*y_loc)+12,95)])
            else:
                # images_bags
                images_bags.append(img[np.maximum(0,(8*x_loc)-12):np.minimum((8*x_loc)+12,95),np.maximum(0,(8*y_loc)-12):np.minimum((8*y_loc)+12,95)])

        if j == 0:
            break

    for i in range(len(images_outs)):
        print("Visualizations for Path A")
        plt.subplot(2,15,i+1)
        plt.axis('off')
        plt.imshow(images_outs[i])

    for i in range(len(images_bags)):
        plt.subplot(2,15,i+16)
        print("Visualizations for Path B")
        plt.axis('off')
        plt.imshow(images_bags[i])

    plt.show(block=True)

    return images_bags,images_outs



import scipy.stats

def logit_correlation(list_bags,list_outs,list_outs2,list_vanilla):
    # Will return a matrix with Cij showing the correlation between the logits of network types i and j.
    # list_ba

    list_all = [list_outs,list_bags,list_outs2,list_vanilla]
    corr_mat = np.zeros((len(list_all),len(list_all)))

    for i in range(len(list_all)):
        for j in range(len(list_all)):
            corr_sum = 0
            for k in range(list_all[i].shape[0]):
                [value,pvalue] = scipy.stats.pearsonr(list_all[i][k,:],list_all[j][k,:])
                corr_sum = corr_sum + value
            corr_mat[i,j] = corr_sum/list_all[i].shape[0]

    return corr_mat





def test_fullbag_network(net, testloader, test_labels):

    net = net.eval()
    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        bagouts = net(images)
        bags = bagouts.view(bagouts.shape[0],bagouts.shape[1],bagouts.shape[2]*bagouts.shape[3])
        bags = torch.mean(bags,2)

        _, predicted = torch.max(bags, 1)
        correct = correct + torch.sum(predicted == labels)
        # torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy

def create_traintestloaders(listdict):

    train_data = listdict[i]['train_data']
    train_labels = listdict[i]['train_label']
    test_data = listdict[i]['test_data']
    test_labels = listdict[i]['test_label']

    # Data_train = Dataset(dataset_name, train_data, train_labels, transform_train,True)
    Data_train = Dataset(dataset_name, train_data, train_labels, transform_train)
    Data_train_phase2 = Dataset(dataset_name, train_data, train_labels, transform_train_phase2)
    Data_test = Dataset(dataset_name, test_data, test_labels, transform_test)
    # Data_test_smoothed =  Dataset(dataset_name, test_data, test_labels, transform_test,smoothing=True)
    # Data_test_distract = Dataset(dataset_name, test_data, test_labels, transform_test,True)


    trainloader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=False, num_workers=0)
    trainloader_small = torch.utils.data.DataLoader(Data_train, batch_size=20, shuffle=False, num_workers=0)
    trainloader_phase2 = torch.utils.data.DataLoader(Data_train_phase2, batch_size=batch_size, shuffle=False, num_workers=0)


    testloader = torch.utils.data.DataLoader(Data_test, batch_size=50, shuffle=False, num_workers=0)
    # testloader_smooth = torch.utils.data.DataLoader(Data_test_smoothed, batch_size=20, shuffle=False, num_workers=0)

    # testloader_with_distractor = torch.utils.data.DataLoader(Data_test_distract, batch_size=20, shuffle=False, num_workers=0)
    return trainloader,trainloader_small,testloader,trainloader_phase2



def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        # if np.random.random() > p:
        #     return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout


if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_name = 'STL10'
    val_splits = 1
    # Good result on MNIST-Scale 1000 Training
    # training_size = 1000
    # batch_size = 100
    # init_rate = 0.05
    # weight_decay = 0.06

    training_size = 5000
    batch_size = 200
    init_rate = 0.02
    init_rate2 = 0.02
    decay_normal = 0.00001
    step_size = 10
    gamma = 0.7
    full_network_epochs = 0
    semibag_epochs = 200
    train_mode = True
    # writepath = './result/stl10_dct.txt'
    # mode = 'a+' if os.path.exists(writepath) else 'w+'
    # f = open(writepath, mode)
    # f.write('Number of epoch is: ' + str(total_epochs) + '\n')

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    augmentation_phase2 = transforms.RandomApply([
        transforms.RandomCrop(96, (12, 12), fill=(int(255 * 0.41), int(255 * 0.40), int(255 * 0.37))),
        # transforms.RandomCrop(96,(12,12),fill=0),
        # transforms.RandomResizedCrop(96,[0.8,1],ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        cutout(32,
               1,
               True,mask_color=(int(255 * 0.41), int(255 * 0.40), int(255 * 0.37))),
    ],p=1)

    augmentation_phase1 = transforms.RandomApply([
        # transforms.RandomCrop(96, (12, 12),fill=0.5),
        # transforms.RandomCrop(96,(12,12),fill =(int(255*0.41),int(255*0.40),int(255*0.37))),
        # transforms.RandomResizedCrop(96, [0.8, 1],ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        # cutout(32,
        #        1,
        #        True),
    ],p=1)

    transform_train = transforms.Compose([
        # transforms.Lambda(lambda x: x.convert("RGB")),
        augmentation_phase1,
        transforms.ToTensor()])
        # normalize])

    transform_train_phase2 = transforms.Compose([
        # transforms.Lambda(lambda x: x.convert("RGB")),
        augmentation_phase2,
        transforms.ToTensor()])
        # normalize])


    transform_test = transforms.Compose([
        transforms.ToTensor()])


    # transform_train = transforms.Compose([transforms.ToTensor()])
    # transform_test = transforms.Compose([transforms.ToTensor()])

    networks = [net_primal(), semi_bagnet()]
    # networks = [net_primal_recon(), semi_bagnet_reconpath_end_to_end()]
    # networks = [net_fullbag()]
    listdict = load_dataset(dataset_name, val_splits, training_size)

    accuracy_all = np.zeros((val_splits, len(networks)))

    for idx in range(1):

        # f.write('%d test cycle: \n' % (idx + 1))
        for i in range(val_splits):

            # networks = [net_primal(),semi_bagnet_end_to_end()]

            networks = [net_primal(),three_way_semibagnet_end_to_end()]
            # networks = [net_fullbag()]

            trainloader,trainloader_small,testloader,trainloader2 = create_traintestloaders(listdict)

            # net = train_fullbag_network(networks[0], trainloader, init_rate, step_size, gamma, full_network_epochs, decay_normal)
            # net = torch.load('./Networks/vgg7_normal_stl10_'+str(i)+'.pt')
            # accuracy = test_network(net, testloader, listdict[i]['test_label'],2)
            # torch.save(net,'./vgg7_normal_stl10_'+str(i)+'.pt')
            # print("Test:", accuracy)
            # print('Stage 1 over')

            # net,mask = train_and_identify_bow_maps(networks[1], trainloader, init_rate2, step_size, gamma, gpool_epochs, decay_normal)
            # accuracy = test_network(net, testloader, listdict[i]['test_label'])
            # print("Test:", accuracy)

            # networks[1].copy_weights_from_parent(copy(networks[0]))
            # # networks[2].prebag_mask = torch.from_numpy(np.float32(mask)).cuda()

            if train_mode:

                net = train_network(networks[0], trainloader, init_rate, step_size, gamma, full_network_epochs,
                                    decay_normal)
                networks[0] = net
                # networks[1].copy_weights_from_parent(net)
                networks[1].generate_from_scratch()

                mask = sort_maps_by_order(net, trainloader, 50)
                networks[1].prebag_mask = mask.float()
                net_semibag = train_semibag_threeway(networks[1], trainloader,trainloader2,testloader,listdict, init_rate, step_size, gamma, semibag_epochs, decay_normal)
                torch.save(net_semibag, './vgg7_threeway_stl10_training5000_withaugmentation'+str(i)+'.pt')

            else:
                net_semibag = torch.load( './vgg7_threeway_stl10_training5000_withaugmentation'+str(i)+'.pt')


            accuracy_train = test_semibag_network_threeway(net_semibag, net_semibag,trainloader_small,
                                                           listdict[i]['train_label'])
            accuracy = test_semibag_network_threeway(net_semibag,net_semibag, testloader, listdict[i]['test_label'])
            print("Train:", accuracy_train,"Test:", accuracy)



            # net_semibag =  train_semibag(networks[1], trainloader, init_rate, step_size, gamma, semibag_epochs, decay_normal,mask)


            # torch.save(net, './model/stl10_cnn.pickle')
            # torch.save(net_semibag,'./vgg7_latest_semibag_end_to_end_stl10_ordersort_50_training500'+str(i)+'.pt')
            # torch.save(net_semibag,'./vgg7_latest_newopt_semibag_end_to_end_stl10_training500'+str(i)+'.pt')
            # net_semibag = torch.load('./temp_stl10_semibag.pt')

            # torch.save(net,'./vgg7_pathA_standalone'+str(i)+'.pt')


            # net_semibag = torch.load('./vgg7_latest_newopt_semibag_end_to_end_stl10_training500'+str(i)+'.pt')
            # net_semibag = torch.load('./vgg7_latest_semibag_end_to_end_stl10_ordersort_50_training500'+str(i)+'.pt')
            # net_semibag = torch.load('./vgg7_latest_semibag_threeway_end_to_end_stl10_training5000'+str(i)+'.pt')
            # vanilla_net = torch.load('./stl10_vanilla_cnn_vgg7_training5000.pt')
            #
            # for siz in range(10,60,2):
            #     a = test_semibag_network_interaction(net_semibag, vanilla_net, testloader, listdict[i]['test_label'], siz)
            #     print(a)
            # break

            # net_semibag = torch.load('./4conv_latest_semibag_threeway_end_to_end_fmnist_training60k0.pt')

            # net_semibag = torch.load('./4conv_latest_semibag_threeway_end_to_end_MNIST_training10k0.pt')

            # net = torch.load('./stl10_vanilla_cnn_vgg7_training5000.pt')
            # vanilla_net = torch.load('./4conv_normal_Fmnist_60k_' + str(i) + '.pt')
            # vanilla_net = torch.load('./4conv_normal_mnist_10k_' + str(i) + '.pt')

            # list_bags, list_outs, list_outs2, list_vanilla = test_semibag_network_threeway(net_semibag,vanilla_net, testloader, listdict[i]['test_label'])
            # images_bags, images_outs = visualize_semibag_network_threeway(net_semibag,testloader,listdict[i]['test_label'])


            # corr_mat = logit_correlation(list_bags, list_outs, list_outs2, list_vanilla)
            # print(corr_mat)

            print('here')

            # net_semibag = torch.load('./vgg7_semibag_end_to_end_stl10_ordersort_50_'+str(i)+'.pt')
            # mask_upper,mask_lower = compute_order_with_mask(net_semibag,trainloader,net_semibag.prebag_mask)

            # break

            # accuracy_train = test_network(net_semibag, trainloader_small, listdict[i]['train_label'])
            # accuracy = test_network(net_semibag, testloader, listdict[i]['test_label'])
            # accuracy_smooth = test_network(net_semibag, testloader_smooth, listdict[i]['test_label'])

            # accuracy = test_semibag_network_threeway(net_semibag, testloader, listdict[i]['test_label'])

            # for temp in range(10):
            #     global_settings.global_SIGMA = 0.3 + (0.025*temp)
            #     accuracy_smooth = test_semibag_network_threeway(net_semibag, testloader_smooth, listdict[i]['test_label'])
                # accuracy_smooth = test_network(net, testloader_smooth, listdict[i]['test_label'])
                # print(accuracy_smooth)


             # accuracy_distractor = test_semibag_network_threeway(net_semibag, testloader_with_distractor, listdict[i]['test_label'])


            # accuracy_train = test_fullbag_network(net, trainloader_small, listdict[i]['train_label'])
            # accuracy = test_fullbag_network(net, testloader, listdict[i]['test_label'])

            # print("Train with Distractors:", accuracy_train,"Test:", accuracy, "Test with Distractors:", accuracy_distractor)

            # print("Original:", accuracy,"With test-time smoothness:", accuracy_smooth)

            # f.write("Train:" + str(accuracy_train) + '\t' + "Test:" + str(accuracy) + '\n')
            accuracy_all[i,idx] = accuracy

        print("Mean Accuracies of Networks:", np.mean(accuracy_all, 1))
        # f.write("Mean Accuracies of Networks:\t" + str(np.mean(accuracy_all, 1)) + '\n')
        # print("Standard Deviations of Networks:", np.std(accuracy_all, 0))
    # f.close()
