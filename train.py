import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

from tqdm import tqdm

from unet.unet_model import UNet

batch_size = 128
num_epochs = 200

# load data
data_amp = sio.loadmat('data/train_data.mat')
train_data_amp = data_amp['train_data_amp']
train_data = train_data_amp

train_label_mask = data_amp['train_label_instance']
num_train_instances = len(train_data)

train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
train_label = torch.from_numpy(train_label_mask).type(torch.LongTensor)

train_dataset = TensorDataset(train_data, train_label)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



data_amp = sio.loadmat('data/test_data.mat')
test_data_amp = data_amp['test_data_amp']
test_data = test_data_amp

test_label_mask = data_amp['test_label_instance']
num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label_mask).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

unet = UNet(n_classes=7)
unet = unet.cuda()

criterion = nn.CrossEntropyLoss(size_average=False).cuda()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[10, 20, 30, 40, 60, 70, 80, 90, 100, 110, 120, 130,
                                                             140, 150, 160, 170, 180, 190, 200, 250, 300],
                                                 gamma=0.5)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    unet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    loss_x = 0
    loss_y = 0
    for (samples, labels) in tqdm(train_data_loader):
        samplesV = Variable(samples.cuda())
        labelsV = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_label = unet(samplesV)

        loss = criterion(predict_label, labelsV)
        print(loss.item())

        loss.backward()
        optimizer.step()

    unet.eval()
    loss_x = 0
    correct_train = 0
    for i, (samples, labels) in enumerate(train_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())

            predict_label = unet(samplesV)

            prediction = predict_label.data.max(1)[1]
            correct_train += prediction.eq(labelsV.data.long()).sum()

            loss = criterion(predict_label, labelsV)
            loss_x += loss.item()

    print("Training accuracy:", (100 * float(correct_train) / (num_train_instances*192)))

    train_loss[epoch] = loss_x / num_train_instances
    train_acc[epoch] = 100 * float(correct_train) / (num_train_instances*192)
    trainacc = str(100 * float(correct_train) / (num_train_instances*192))[0:6]


    loss_x = 0
    correct_test = 0

    for i, (samples, labels) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            labelsV = Variable(labels.cuda())

            predict_label = unet(samplesV)

            prediction = predict_label.data.max(1)[1]
            correct_test += prediction.eq(labelsV.data.long()).sum()

            loss = criterion(predict_label, labelsV)
            loss_x += loss.item()

    print("Test accuracy:", (100 * float(correct_test) / (num_test_instances * 192)))

    test_loss[epoch] = loss_x / num_test_instances
    test_acc[epoch] = 100 * float(correct_test) / (num_test_instances * 192)
    testacc = str(100 * float(correct_test)/(num_test_instances * 192))[0:6]

    if epoch == 0:
        temp_test = correct_test
        temp_train = correct_train
    elif correct_test > temp_test:
        torch.save(unet, 'weights/' + trainacc + 'Test' + testacc + '.pkl')
        temp_test = correct_test
        temp_train = correct_train


sio.savemat(
    'results/TrainLoss_' + 'Train' + str(100 * float(temp_train) / (num_test_instances * 192))[
                                                                 0:6] + 'Test' + str(
        100 * float(temp_test) / (num_test_instances * 192))[0:6] + '.mat', {'train_loss': train_loss})
sio.savemat(
    'results/TestLoss_' + 'Train' + str(100 * float(temp_train) / (num_test_instances * 192))[
                                                                0:6] + 'Test' + str(
        100 * float(temp_test) / (num_test_instances * 192))[0:6] + '.mat', {'test_loss': test_loss})
sio.savemat('results/TrainAccuracy_' + 'Train' + str(
    100 * float(temp_train) / (num_test_instances * 192))[0:6] + 'Test' + str(100 * float(temp_test) / (num_test_instances * 192))[
                                                                   0:6] + '.mat', {'train_acc': train_acc})
sio.savemat('results/TestAccuracy_' + 'Train' + str(
    100 * float(temp_train) / (num_test_instances * 192))[0:6] + 'Test' + str(100 * float(temp_test) / (num_test_instances * 192))[
                                                                   0:6] + '.mat', {'test_acc': test_acc})

