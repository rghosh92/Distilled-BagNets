import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torchvision import datasets, models, transforms
import sys
import time
import json
import copy
import seaborn as sns
import numpy as np
import pickle
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

from ScaleSteerableInvariant_Network_groupeq import *

def train(net, trainloader, trainlabel, total_epochs):

    print('train start...')

    for epoch in range(total_epochs):

        torch.cuda.empty_cache()

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        print('Epoch {}/{}'.format(epoch+1, total_epochs))
        print('-' * 10)

        # torch.cuda.empty_cache()
        # scheduler.step()

        running_loss = 0.0
        running_corrects = 0

        scheduler.step()

        for i, data in enumerate(trainloader, 0):

            dataset_sizes = len(trainlabel)

            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(trainloader), loss.item() * inputs.size(0)), end="")

        sys.stdout.flush()

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes

        avg_loss = epoch_loss
        t_acc = epoch_acc
        print()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print()

    return net


def test(net, testloader, testlabel):

    net = net.eval()

    dataset_sizes = len(testlabel)

    running_loss = 0.0
    running_corrects = 0
    total_correct_number = []

    print('Top1 Acc, Test Start...')
    print()

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(testloader), loss.item() * inputs.size(0)), end="")
        sys.stdout.flush()

        loss = running_loss / len(testloader)
        acc = running_corrects.double() / dataset_sizes

    #     # topk error accuracy
    #     topk_correct_number = correct_prediction(outputs, labels, topk=(5,))
    #     # print('Number of correct prediction: {:.4f}'.format(topk_correct_number))
    #     total_correct_number.append(topk_correct_number)
    #
    # acc = sum(total_correct_number) / dataset_sizes

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))
    print('Test Acc: {:.4f}'.format(acc))
    print()

    return acc

def test_topk(net, testloader, testlabel):

    net = net.eval()

    dataset_sizes = len(testlabel)

    running_loss = 0.0
    running_corrects = 0
    total_correct_number = []

    print('Topk Acc, Test Start...')
    print()

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)

        # _, preds = torch.max(outputs, 1)
        # loss = criterion(outputs, labels)
        #
        # # statistics
        # running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(preds == labels.data)
        # print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(testloader), loss.item() * inputs.size(0)), end="")
        # sys.stdout.flush()
        #
        # loss = running_loss / len(testloader)
        # acc = running_corrects.double() / dataset_sizes

        # topk error accuracy
        topk_correct_number = correct_prediction(outputs, labels, topk=(5,))
        # print('Number of correct prediction: {:.4f}'.format(topk_correct_number))
        total_correct_number.append(topk_correct_number)

    acc = sum(total_correct_number) / dataset_sizes

    print()
    # print('Test Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))
    print('Test Acc: {:.4f}'.format(acc))
    print()

    return acc


def correct_prediction(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []
    for k in topk:
        # correct_k is the number of correct prediction number
        correct_k = correct[:k].view(-1).float().sum(0)
        correct = correct_k.view(1).cpu().numpy()[0]
        # res.append(correct_k.mul_(100.0 / batch_size))

    return correct


if __name__ == '__main__':

    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    norm_mean=[0.485, 0.456, 0.406]
    norm_stdv=[0.229, 0.224, 0.225]

    # TODO: Define your transforms for the training, validation, and testing sets
    # defining data transforms for training, validation and test data and also normalizing whole data
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(norm_mean, norm_stdv)])

    # data_transforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    #     transforms.RandomRotation(30),
    #     transforms.RandomHorizontalFlip(p=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(norm_mean, norm_stdv)])

    # TODO: Load the datasets with ImageFolder
    trainset = datasets.ImageFolder(train_dir, transform=data_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=data_transforms)
    testset = datasets.ImageFolder(test_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    # Save class ids
    class_idx = trainset.class_to_idx

    # TODO: Save the data into list for pickle file

    # train_data = np.zeros([6552,3,192,192])
    # train_label = np.zeros([6552])
    # test_data = np.zeros([819,3,192,192])
    # test_label = np.zeros([819])

    # print('train data')

    # for idx in range(6552):
    #     print(idx)
    #     train_data[idx,:,:,:] = train[idx][0]
    #     train_label[idx] = train[idx][1]

    # # save into splits
    # i = 0
    # for split in range(9):
    #     train_data = np.zeros([728,3,192,192])
    #     train_label = np.zeros([728])
    #     for idx in range(728):
    #         print(i)
    #         train_data[idx, :, :, :] = train[i][0]
    #         train_label[idx] = train[i][1]
    #         i += 1
    #     dict = {}
    #     dict['train_data'] = train_data
    #     dict['train_label'] = train_label
    #     pickle.dump(dict, open('flowers_102_train' +str(split)+ '.pickle', 'wb'))

    # print('test data')

    # for idx in range(819):
    #     print(idx)
    #     test_data[idx,:,:,:] = test[idx][0]
    #     test_label[idx] = test[idx][1]

    # dict = {}
    # dict['train_data'] = train_data
    # dict['train_label'] = train_label
    # dict['test_data'] = test_data
    # dict['test_label'] = test_label

    # pickle.dump(dict, open('flowers_102_test.pickle', 'wb'))

    # TODO: Load pre-trained VGG model
    # define a network by loading pretrained imagenet vgg models
    networks = models.vgg11(pretrained=True)
    # print(networks)

    # TODO: Freeze the sequential layer and only train on new classifier layer
    # apply the new classifier layer and only trained on them
    for parma in networks.parameters():
        parma.requires_grad = False

    networks.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(p=0.5),
                                              torch.nn.Linear(4096, 4096),
                                              torch.nn.ReLU(),
                                              torch.nn.Dropout(p=0.5),
                                              torch.nn.Linear(4096, 102))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = networks.to(device)
    print(network)

    # TODO: Train on new classifier - fully connected layer
    # define parameters
    init_rate = 0.01
    decay_normal = 0.0001
    batch_size = 32
    step_size = 10
    gamma = 0.7
    epochs = 200

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(network.classifier.parameters(), lr=init_rate, momentum=0.9, weight_decay=decay_normal)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    trainlable = trainset.targets
    model = train(network, trainloader, trainlable, total_epochs=epochs)
    torch.save(model, './flower102_train_only_fc.pt')
    acc = test(model, testloader, testset.targets)
    acc_topk = test_topk(model, testloader, testset.targets)
