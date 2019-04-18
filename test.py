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

batch_size = 278
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

test_label_mask = data_amp['test_label_mask']
# test_label_mask = data_amp['test_label_instance']
num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label_mask).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# unet = torch.load('weights/95.997Test88.603.pkl')

unet = torch.load('weights/Two96.253Test95.093.pkl')


unet.eval()
loss_x = 0
correct_test = 0
for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labelsV = Variable(labels.cuda())

        predict_label = unet(samplesV)

        prediction = predict_label.data.max(1)[1]
        correct_test += prediction.eq(labelsV.data.long()).sum()

print("Test accuracy:", (100 * float(correct_test) / (num_test_instances * 192)))


sio.savemat('out/Two96.253Test95.093.mat', {'out': predict_label.cpu().numpy()} )



