
  	Oturumu Kapat
Mesaj Yaz   Adresler   Klasörler   Seçenekler   Ara   Yardım   	METU


Bir yazı eklentisi görüntüleniyor - Mesajı göster
Bunu dosya olarak indir

# Feel free to change / extend / adapt this source code as needed to complete the
homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 16
max_num_epoch = 100
hps = {'lr':0.001}

# ---- options ----
DEVICE_ID = 'cuda:0' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False # set True to visualize input, prediction and the output from the
last batch
LOAD_CHKPT = False

# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils
torch.multiprocessing.set_start_method('spawn', force=True)
# ---- utility functions -----
def get_loaders(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset' 
    train_set =
hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

# ---- ConvNet1 -----
class Net1(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=(kernel_size-1)//2) 
#(in_channels, out_channels, kernel_size) padding=1

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = self.conv1(grayscale_image)
        return x

        

# ---- ConvNet2 -----
class Net2(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(1, 3, kernel_size, padding=(kernel_size-1)//2)      

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        c1 = self.conv1(grayscale_image)
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)
        return c2

# ---- ConvNet4 -----
class Net4(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.relu2 = nn.ReLU(inplace = True)
        self.conv3 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.relu3 = nn.ReLU(inplace = True)
        self.conv4 = nn.Conv2d(1, 3, kernel_size, padding=((kernel_size-1)/2))         

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        c1 = self.conv1(grayscale_image)
        r1 = self.relu1(c1)
        c2 = self.conv2(r1)
        r2 = self.relu2(c2)
        c3 = self.conv3(r2)
        r3 = self.relu3(c3)
        c4 = self.conv4(r3)
        return c4

num_layers = [1,2,4]
kernel_sizes = [3,5]
num_kernels = [2,4,8]
learning_rates = [0.1, 0.01, 0.001, 0.0001]

# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))

for num_layer in num_layers[:1]:
    for learning_rate in learning_rates[3:]:    
        for kernel_size in kernel_sizes[:1]:
            for num_kernel in num_kernels[:1]:
                if(num_layer == 1):
                    net = Net1(kernel_size, num_kernel).to(device=device)
                elif(num_layer == 2):
                    net = Net2(kernel_size, num_kernel).to(device=device)
                elif(num_layer == 4):
                    net = Net4(kernel_size, num_kernel).to(device=device)
                criterion = nn.MSELoss()
                optimizer = optim.SGD(net.parameters(), lr=learning_rate)
                train_loader, val_loader = get_loaders(batch_size,device)


                if LOAD_CHKPT:
                    print('loading the model from the checkpoint')
                    model.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

                print('training begins')
                for epoch in range(max_num_epoch):  
                    running_loss = 0.0 # training loss of the network
                    for iteri, data in enumerate(train_loader, 0):
                        inputs, targets = data # inputs: low-resolution images,
targets: high-resolution images.

                        optimizer.zero_grad() # zero the parameter gradients

                        # do forward, backward, SGD step
                        preds = net(inputs)
                        loss = criterion(preds, targets)
                        loss.backward()
                        optimizer.step()

                        # print loss
                        running_loss += loss.item()
                        print_n = 100 # feel free to change this constant
                        if iteri % print_n == (print_n-1):    # print every print_n
mini-batches
                            print('[%d, %5d] network-loss: %.3f' %
                                (epoch + 1, iteri + 1, running_loss / 100))
                            running_loss = 0.0
                            # note: you most probably want to track the progress on
the validation set as well (needs to be implemented)

                        if (iteri==0) and VISUALIZE: 
                            hw3utils.visualize_batch(inputs,preds,targets)

                    print('Saving the model, end of epoch %d' % (epoch+1))
                    if not os.path.exists(LOG_DIR):
                        os.makedirs(LOG_DIR)
                    torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint.pt'))
                    hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,'example.png'))
