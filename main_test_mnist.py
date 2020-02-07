import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from copy import copy

from torch.optim.lr_scheduler import StepLR

# from ScaleSteerableInvariant_Network import *
from semi_bagnet import *
# from Network import *
from spatial_order_func import *

import numpy as np
import sys, os
from utils import Dataset, load_dataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

        torch.cuda.empty_cache()
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

        torch.cuda.empty_cache()
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

        torch.cuda.empty_cache()
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

import matplotlib.pyplot as plt

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


def train_and_identify_bow_maps(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

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

        torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outs = net(inputs)
            loss = criterion(outs,labels)
            loss.backward()
            optimizer.step()

    ptile = np.percentile(torch.var(net.linear1.weight,0).detach().cpu().numpy(),20)
    mask = torch.var(net.linear1.weight, 0) < ptile
    # mask = 1-mask

        # print('break')
    net = net.eval()

    return net,mask

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

        if out_size == 1:
            outputs = net(images)
        else:
            [outputs, temp] = net(images)

        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        torch.cuda.empty_cache()

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

        [outs,bagouts] = net(images)

        bags = bagouts.view(bagouts.shape[0],bagouts.shape[1],bagouts.shape[2]*bagouts.shape[3])
        bags = torch.mean(bags,2)

        _, predicted = torch.max(outs+2*bags, 1)
        correct = correct + torch.sum(predicted == labels)
        torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy

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
        torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy


def create_traintestloaders(listdict,i):
    train_data = listdict[0]['train_data']
    train_labels = listdict[0]['train_label']
    test_data = listdict[0]['test_data']
    test_labels = listdict[0]['test_label']

    Data_train = Dataset(dataset_name, train_data, train_labels, transform_train)
    Data_test = Dataset(dataset_name, test_data, test_labels, transform_test)

    trainloader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=False, num_workers=4)
    trainloader_small = torch.utils.data.DataLoader(Data_train, batch_size=20, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(Data_test, batch_size=20, shuffle=False, num_workers=2)

    return trainloader,trainloader_small,testloader

if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_name = 'FMNIST'
    val_splits = 9
    layer_splits = [10,20,30,40,50,60,70,80,90]
    # Good result on MNIST-Scale 1000 Training
    # training_size = 1000
    # batch_size = 100
    # init_rate = 0.05
    # weight_decay = 0.06

    training_size = 10000
    batch_size = 400
    init_rate = 0.04
    init_rate2 = 0.04
    decay_normal = 0.0001
    step_size = 10
    gamma = 0.7
    full_network_epochs = 0
    gpool_epochs = 20
    semibag_epochs = 200

    # writepath = './result/stl10_dct.txt'
    # mode = 'a+' if os.path.exists(writepath) else 'w+'
    # f = open(writepath, mode)
    # f.write('Number of epoch is: ' + str(total_epochs) + '\n')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])

    networks = [net_primal_mnist(), semi_bagnet_mnist_end_to_end()]
    # networks = [net_fullbag()]
    listdict = load_dataset(dataset_name, 1, training_size)

    accuracy_all = np.zeros((val_splits, len(networks)))

    for idx in range(1):

        # f.write('%d test cycle: \n' % (idx + 1))
        for i in range(val_splits):

            networks = [net_primal_mnist(), semi_bagnet_mnist_end_to_end()]
            # networks = [net_fullbag()]
            trainloader,trainloader_small,testloader = create_traintestloaders(listdict,i)

            # net = train_fullbag_network(networks[0], trainloader, init_rate, step_size, gamma, full_network_epochs, decay_normal)


            # net = train_network(networks[0], trainloader, init_rate, step_size, gamma, full_network_epochs, decay_normal)
            # accuracy = test_network(net, testloader, listdict[i]['test_label'],2)
            # torch.save(net,'./4conv_normal_Fmnist_60k_' + str(i) + '.pt')
            #
            # print("Test:", accuracy)
            # print('Stage 1 over')
            # net = torch.load('./4conv_normal_Fmnist_60k_' + str(i) + '.pt')
            # networks[1].copy_weights_from_parent(net)
            #
            # mask = sort_maps_by_order(net, trainloader, layer_splits[i])
            #
            # mask = 1-mask

            # net,mask = train_and_identify_bow_maps(networks[1], trainloader, init_rate2, step_size, gamma, gpool_epochs, decay_normal)
            # accuracy = test_network(net, testloader, listdict[i]['test_label'])
            # print("Test:", accuracy)

            # networks[1].copy_weights_from_parent(copy(networks[0]))
            # networks[2].prebag_mask = torch.from_numpy(np.float32(mask)).cuda()
            # networks[1].prebag_mask = mask.float()
            # net_semibag =  train_semibag(networks[1], trainloader, init_rate, step_size, gamma, semibag_epochs, decay_normal,mask)
            # torch.save(net, './model/stl10_cnn.pickle')
            # torch.save(net_semibag,'./vgg7_latest_Mnist_10k_semibag_end_to_end_'+str(layer_splits[i])+'.pt')

            # net_normal = torch.load('./4conv_normal_mnist_10k_' + str(i) + '.pt')
            net_semibag = torch.load('./vgg7_latest_Mnist_60k_semibag_end_to_end_10.pt')

            accuracy_train = test_semibag_network(net_semibag, trainloader_small, listdict[0]['train_label'])
            accuracy = test_semibag_network(net_semibag, testloader, listdict[0]['test_label'])

            # accuracy_train = test_fullbag_network(net, trainloader_small, listdict[i]['train_label'])
            # accuracy = test_fullbag_network(net, testloader, listdict[i]['test_label'])

            print("Train:", accuracy_train, "Test:", accuracy)
            # f.write("Train:" + str(accuracy_train) + '\t' + "Test:" + str(accuracy) + '\n')
            accuracy_all[i,idx] = accuracy

        print("Mean Accuracies of Networks:", np.mean(accuracy_all, 1))
        # f.write("Mean Accuracies of Networks:\t" + str(np.mean(accuracy_all, 1)) + '\n')
        # print("Standard Deviations of Networks:", np.std(accuracy_all, 0))
    # f.close()