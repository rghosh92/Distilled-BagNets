import os, sys
import glob
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle

# from AlexNet import *
# from Network import *
from BagNet import *
# from ScaleSteerableInvariant_Network_groupeq import *
from semi_bagnet_STL import *
from spatial_order_func import *

from torch.optim.lr_scheduler import StepLR

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS_START = 451
NUM_IMAGES_PER_CLASS = 500
NUM_CLASS_USED = 200
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root, split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        # self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.image_paths = []
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        # fp = open('./result/tinyimagenet_name_label.txt', 'a+')
        # for i, text in enumerate(self.label_texts):
        #     fp.write("%s : %s" % (text, i) + '\n')

        if self.split == 'train':
            # convert class name to label
            for label_text, i in self.label_text_to_number.items():
                # for cnt in range(NUM_IMAGES_PER_CLASS_START-1, (NUM_IMAGES_PER_CLASS_START-1)+50):
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    # print(cnt)
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
                    # fp.write('%s_%d.%s : %d' % (label_text, cnt, EXTENSION, i) + '\n')
                    # create image paths based on NUM_IMAGES_PER_CLASS
                    temp_file_path = '%s/images/%s_%d.%s' % (label_text, label_text, cnt, EXTENSION)
                    self.image_paths.append(os.path.join(self.split_dir, temp_file_path))

        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]
                    temp_file_path = 'images/%s' % (file_name)
                    self.image_paths.append(os.path.join(self.split_dir, temp_file_path))

        # # read all images into torch tensor in memory to minimize disk IO overhead
        # if self.in_memory:
        #     self.images = [self.read_image(path) for path in self.image_paths]

        self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]

            # # after transform, change the image size on gpu cause there is not enough space for image to be 192 on cpu
            # img = np.transpose(img, [1, 2, 0])
            # img = Image.fromarray(np.uint8(255.0*img))
            # img = img.resize((192, 192), Image.ANTIALIAS)
            # # transform PIL image back on tensor
            # img = transforms(img)

            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        # img = np.array(Image.open(path))
        img = Image.open(path)
        return self.transform(img) if self.transform else img

    def save_data(self):
        lengh = len(self.image_paths)
        data = np.zeros([lengh, 3, 64, 64])
        label = np.zeros([lengh])
        temp = []
        for idx, path in enumerate(self.image_paths):
            image_name = path.split('/')[-1]
            image = self.read_image(path)
            if len(image.shape) == 2:
                data[idx, 0, :, :] = image[:, :]
                temp.append(image)
            if len(image.shape) == 3:
                data[idx, :, :, :] = image
            label[idx] = self.labels[image_name]
        # print(data.shape, label.shape, len(temp))
        return data, label


def train(net, trainloader, trainlabel, init_rate, total_epochs, weight_decay, step_size, gamma):

    # net = net
    # net = net.cuda()
    net = net.train()

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # best_acc = 0.0
    # best = 0

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
            inputs = inputs.cuda()
            labels = labels.cuda()

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

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0
    total_correct_number = []

    print('Test Start...')
    print()

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # TODO: for vanilla cnn(net_primal)
        # [outputs, prebags] = net(inputs)

        # TODO: for only bagnets(net_fullbag)
        bags = net(inputs)

        # TODO: for net_primal
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

        # TODO: topk error accuracy
        # topk_correct_number = correct_prediction(outputs, labels, topk=(5,))
        # # print('Number of correct prediction: {:.4f}'.format(topk_correct_number))
        # total_correct_number.append(topk_correct_number)

        # prebags_total = []
        # pre = prebags.view(prebags.shape[0], prebags.shape[1], prebags.shape[2]*prebags.shape[3])
        # pre = torch.mean(pre, 2)
        # prebags_correct_number = correct_prediction(pre, labels, topk=(5,))
        # prebags_total.append(prebags_correct_number)

        # TODO: for only bagnet(net_fullbag)
        bags = bags.view(bags.shape[0], bags.shape[1], bags.shape[2]*bags.shape[3])
        bagouts = torch.mean(bags, 2)
        bagouts_correct_number = correct_prediction(bagouts, labels, topk=(5,))
        total_correct_number.append(bagouts_correct_number)

    acc = sum(total_correct_number) / dataset_sizes
    # acc_pre = sum(prebags_total) / dataset_sizes

    return acc
    # return acc, acc_pre


# TOP5 acc
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


def train_normal_network(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()

    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epochs):

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

            # TODO: for vanilla cnn(net_primal)
            # [outputs, temp] = net(inputs)

            # TODO: for bagnets(net_fullbag)
            bags = net(inputs)
            bagouts = bags.view(bags.shape[0], bags.shape[1], bags.shape[2] * bags.shape[3])
            outputs = torch.mean(bagouts, 2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    net = net.eval()

    return net


def train_semibag_threeway(net, trainloader, trainloader2, trainlable, init_rate, step_size, gamma, total_epochs, weight_decay):

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

        # torch.cuda.empty_cache()
        scheduler.step()
        print()
        print('Epoch {}/{}'.format(epoch+1, total_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects_bag = 0
        running_corrects_out = 0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            [outs, bagouts, outs2] = net(inputs)

            bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2] * bagouts.shape[3])
            bags = torch.mean(bags, 2)

            _, preds_bag = torch.max(bags, 1)
            _, preds_out = torch.max(outs, 1)

            bagloss_items = []
            for bagx in range(bagouts.shape[2]):
                for bagy in range(bagouts.shape[3]):
                    bagloss_items.append(criterion(bagouts[:,:,bagx,bagy], labels))

            total_loss = [4*sum(bagloss_items)/(bagouts.shape[2]*bagouts.shape[3]), criterion(outs, labels)]
            total_loss = sum(total_loss)
            total_loss.backward()

            optimizer.step()

            # statistics
            running_loss += total_loss.item() * inputs.size(0)
            running_corrects_bag += torch.sum(preds_bag == labels.data)
            running_corrects_out += torch.sum(preds_out == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(trainloader), total_loss.item() * inputs.size(0)), end="")

        sys.stdout.flush()

        dataset_sizes = len(trainlable)
        epoch_loss = running_loss / dataset_sizes
        epoch_acc_a = running_corrects_bag.double() / dataset_sizes
        epoch_acc_b = running_corrects_out.double() / dataset_sizes
        print()
        print('Train Loss: {:.4f} Acc_bag: {:.4f} Acc_out: {:.4f}'.format(epoch_loss, epoch_acc_a, epoch_acc_b))
        print()

    # TODO: TRAINING PHASE2

    running_loss2 = 0.0
    running_corrects_out2 = 0

    net.set_network2_grad(True)
    optimizer = optim.SGD([
        {'params': slow_parameters, 'lr': init_rate},
        {'params': fast_parameters, 'lr': init_rate}], momentum=0.9, weight_decay=weight_decay)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(total_epochs):

        # torch.cuda.empty_cache()
        scheduler.step()

        print()
        print('Epoch {}/{}'.format(epoch+1, total_epochs))
        print('-' * 10)

        for i, data in enumerate(trainloader2, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            [outs, bagouts, outs2] = net(inputs)

            _, preds_out2 = torch.max(outs2, 1)

            loss2 = criterion(outs2, labels)
            loss2.backward()

            optimizer.step()

            # statistics
            running_loss2 += loss2.item() * inputs.size(0)
            running_corrects_out2 += torch.sum(preds_out2 == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i + 1, len(trainloader), loss2.item() * inputs.size(0)), end="")

        sys.stdout.flush()

        epoch_loss = running_loss / dataset_sizes
        epoch_acc_c = running_corrects_out2.double() / dataset_sizes
        print()
        print('Train Loss: {:.4f} Acc_out2: {:.4f}'.format(epoch_loss, epoch_acc_c))
        print()

    return net


def sort_maps_by_order(net, trainloader, ptile):

    net = net.eval()
    count = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        [outs, outs_prebag] = net(inputs)

        outs_prebag = outs_prebag.detach().cpu().numpy()

        if i == 0:
            map_orders = np.zeros(outs_prebag.shape[1])
        else:
            map_orders = map_orders + spatial_order_cnn_maps_multiscale(outs_prebag, [1])
            count = count+1
        if i == 5:
            break

    fmap_orders = map_orders/count

    ptile = np.percentile(fmap_orders, ptile)
    print(ptile)
    mask = fmap_orders > ptile
    mask = torch.from_numpy(np.float32(mask)).cuda()

    return mask


def test_semibag_network_threeway(net, testloader, test_labels):

    net = net.eval()

    # correct_bags = torch.tensor(0)
    # correct_outs = torch.tensor(0)
    # correct_outs2 = torch.tensor(0)

    list_bags = []
    list_outs = []
    list_outs2 = []
    list_all = []
    list_ab = []
    list_ac = []

    dataset_sizes = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):

        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()

        [outs, bagouts, outs2] = net(images)

        # TODO: TOP1 ACC

        # bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2]*bagouts.shape[3])
        # bags = torch.mean(bags, 2)
        #
        # _, predicted = torch.max(bags, 1)
        # correct_bags = correct_bags + torch.sum(predicted == labels)
        # list_bags.append(bags.cpu().detach().numpy())
        #
        # _, predicted = torch.max(outs, 1)
        # correct_outs = correct_outs + torch.sum(predicted == labels)
        # list_outs.append(outs.cpu().detach().numpy())
        #
        # _, predicted = torch.max(outs2, 1)
        # correct_outs2 = correct_outs2 + torch.sum(predicted == labels)
        # list_outs2.append(outs2.cpu().detach().numpy())

        # TODO: TOP5 ACC

        bags = bagouts.view(bagouts.shape[0], bagouts.shape[1], bagouts.shape[2]*bagouts.shape[3])
        bags = torch.mean(bags, 2)
        correct_bags = correct_prediction(bags, labels, topk=(5,))
        list_bags.append(correct_bags)

        correct_outs = correct_prediction(outs, labels, topk=(5,))
        list_outs.append(correct_outs)

        correct_outs2 = correct_prediction(outs2, labels, topk=(5,))
        list_outs2.append(correct_outs2)

        correct_all = correct_prediction(outs+bags+outs2, labels, topk=(5,))
        list_all.append(correct_all)

        correct_ab = correct_prediction(outs + bags, labels, topk=(5,))
        list_ab.append(correct_ab)

        correct_ac = correct_prediction(outs + outs2, labels, topk=(5,))
        list_ac.append(correct_ac)

    # accuracy_bags = float(correct_bags)/float(total)
    # accuracy_outs = float(correct_outs)/float(total)
    # accuracy_outs2 = float(correct_outs2)/float(total)

    accuracy_bags = sum(list_bags) / dataset_sizes
    accuracy_outs = sum(list_outs) / dataset_sizes
    accuracy_outs2 = sum(list_outs2) / dataset_sizes
    accuracy_all = sum(list_all) / dataset_sizes
    accuracy_ab = sum(list_ab) / dataset_sizes
    accuracy_ac = sum(list_ac) / dataset_sizes

    return accuracy_outs, accuracy_bags, accuracy_outs2, accuracy_all, accuracy_ab, accuracy_ac


if __name__ == '__main__':

    # TODO: Build the dataset for pytporch dataloader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation = transforms.RandomApply([
        transforms.RandomResizedCrop(64, scale=(0.3, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)])

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        augmentation,
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # normalize,
    ])

    # transforms = transforms.Compose([transforms.ToTensor(), ])

    tiny_train = TinyImageNet('./tiny-imagenet-200/', split='train', transform=training_transform)
    # train_data, train_label = tiny_train.save_data()
    # print(len(train_data), len(train_label))

    tiny_val = TinyImageNet('./tiny-imagenet-200/', split='val', transform=valid_transform)
    # test_data, test_label = tiny_val.save_data()
    # print(len(test_data), len(test_label))

    # data_dict = {}
    # # data_dict['train_data'] = train_data
    # # data_dict['train_label'] = train_label
    # data_dict['test_data'] = test_data
    # data_dict['test_label'] = test_label

    # try:
    #     os.mkdir('TINY_IMAGENET/without_normalization/')
    # except:
    #     None
    #
    # os.chdir('TINY_IMAGENET/without_normalization/')
    #
    # pickle.dump(data_dict, open('tiny_imagenet_test.pickle', 'wb'))
    #
    # os.chdir("..")

    init_rate = 0.01
    decay_normal = 0.0001
    batch_size = 200
    step_size = 10
    gamma = 0.7
    epochs = 200
    epochs_vanilla = 200

    train_mode = True

    # Load Data
    train_loader = torch.utils.data.DataLoader(tiny_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(tiny_val, batch_size=batch_size, shuffle=False, num_workers=0)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # TODO: Define the Network
    # networks = [bagnet9()]
    # networks = [net_primal(), three_way_semibagnet_end_to_end()]
    networks = [net_fullbag()]

    net = train_normal_network(networks[0], train_loader, init_rate, step_size, gamma, epochs_vanilla, decay_normal)
    torch.save(net, './semibagnets/tinyimagenet_aug_fullbagnet.pt')
    acc_tr = test(net, train_loader, tiny_train.labels)
    acc_te = test(net, test_loader, tiny_val.labels)
    print("Train:", acc_tr, "Test:", acc_te)

    # networks[1].generate_from_scratch()

    # mask = sort_maps_by_order(net, train_loader, 50)
    # networks[1].prebag_mask = mask.float()

    # TODO: TO TRAIN BAGNETS OR OTHER NETWORKS
    # for network in networks:
    #
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     network = network.to(device)
    #
    #     trainlable = tiny_train.labels
    #     model = train(network, train_loader, trainlable, init_rate, total_epochs=epochs, weight_decay=decay_normal,
    #                   step_size=step_size, gamma=gamma)
    #     torch.save(model, './dct_normalcnn_tinyimagenet_training_aug.pt')
    #     acc = test(model, test_loader, tiny_val.labels)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network = network.to(device)

    # if train_mode:
    #     # TODO： TRAIN SEMI-BAGENET
    #     net_semibag = train_semibag_threeway(networks[1], train_loader, train_loader, tiny_train.labels, init_rate, step_size, gamma,
    #                                          epochs, decay_normal)
    #     torch.save(net_semibag, './semibagnets/tinyimagenet_aug_gullbag.pt')
    # else:
    #     net_semibag = torch.load('./semibagnets/tinyimagenet_nonorm_64.pt')
    #
    # # TODO: TEST THE MODEL
    # acc_train = test_semibag_network_threeway(net_semibag, train_loader, tiny_train.labels)
    # acc_test = test_semibag_network_threeway(net_semibag, test_loader, tiny_val.labels)
    # print("Train:", acc_train, "Test:", acc_test)



