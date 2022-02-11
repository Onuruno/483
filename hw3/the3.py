# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations, 
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 16
max_num_epoch = 100
hps = {'lr':0.1}

# ---- options ----
DEVICE_ID = 'cuda:0' # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints2'
VISUALIZE = False # set True to visualize input, prediction and the output from the last batch
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
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'train'),device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'val'),device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader

def get_test_loader(batch_size,device):
    data_root = 'ceng483-s19-hw3-dataset'
    test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'test_inputs'),device=device)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader

# ---- ConvNet1 -----
class Net1(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size, padding=(kernel_size-1)//2)  #(in_channels, out_channels, kernel_size) padding=1

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        x = self.conv1(grayscale_image)
        return x

        

# ---- ConvNet2 -----
class Net2(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(num_kernels, 3, kernel_size, padding=(kernel_size-1)//2)
        #self.m1 = nn.BatchNorm2d(1)
        #self.m2 = nn.BatchNorm2d(num_kernels)
        
    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        c1 = self.conv1(grayscale_image)
        r1 = F.relu(c1)
        c2 = self.conv2(r1)
        #return torch.tanh(c2)
        return c2

# ---- ConvNet4 -----
class Net4(nn.Module):
    def __init__(self, kernel_size, num_kernels):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.conv3 = nn.Conv2d(num_kernels, num_kernels, kernel_size, padding=(kernel_size-1)//2)
        self.conv4 = nn.Conv2d(num_kernels, 3, kernel_size, padding=(kernel_size-1)//2)        

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:      
        c1 = self.conv1(grayscale_image)
        r1 = F.relu(c1)
        c2 = self.conv2(r1)
        r2 = F.relu(c2)
        c3 = self.conv3(r2)
        r3 = F.relu(c3)
        c4 = self.conv4(r3)
        return c4

num_layers = [1,2,4]
kernel_sizes = [3,5]
num_kernels = [2,4,8]

# ---- training code -----
device = torch.device(DEVICE_ID)
print('device: ' + str(device))

i=0
for num_layer in num_layers:   
    for kernel_size in kernel_sizes:
        for num_kernel in num_kernels:
            if(num_layer == 1):
                net = Net1(kernel_size, num_kernel).to(device=device)
            elif(num_layer == 2):
                net = Net2(kernel_size, num_kernel).to(device=device)
            elif(num_layer == 4):
                net = Net4(kernel_size, num_kernel).to(device=device)
            criterion = nn.MSELoss()
            learning_rate = 0.1
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
            train_loader, val_loader = get_loaders(batch_size,device)


            if LOAD_CHKPT:
                print('loading the model from the checkpoint')
                model.load_state_dict(os.path.join(LOG_DIR,'checkpoint' + str(i-1) + '.pt'))
            
            validation_losses = []
                
            print('number of layers: ' + str(num_layer))
            print('kernel size: ' + str(kernel_size))
            print('number of kernels: ' + str(num_kernel))
            print('training begins')
            for epoch in range(max_num_epoch):  
                running_loss = 0.0 # training loss of the network
                for iteri, data in enumerate(train_loader, 0):
                    inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.

                    optimizer.zero_grad() # zero the parameter gradients

                    # do forward, backward, SGD step
                    preds = net(inputs)
                    loss = criterion(preds, targets)
                    loss.backward()
                    optimizer.step()

                    # print loss
                    running_loss += loss.item()
                    print_n = 100 # feel free to change this constant
                    if iteri % print_n == (print_n-1):    # print every print_n mini-batches
                        print('[%d, %5d] network-loss: %.3f' %
                            (epoch + 1, iteri + 1, running_loss / 100))
                        running_loss = 0.0
                        # note: you most probably want to track the progress on the validation set as well (needs to be implemented)

                    if (iteri==0) and VISUALIZE: 
                        hw3utils.visualize_batch(inputs,preds,targets)
                if(epoch % 5 == 4):
                    val_loss_total=0.0
                    for val_inputs, val_targets in val_loader:
                        val_preds = net(val_inputs)
                        val_loss = criterion(val_preds, val_targets)
                        val_loss_total += val_loss.item()
                    print('Validation loss: ' + str(val_loss_total/100))
                    print('learning rate: ' + str(learning_rate))
                    if(len(validation_losses) > 0 and val_loss_total > validation_losses[-1]):
                        if(learning_rate > 0.001):
                            validation_losses.append(val_loss_total)
                            learning_rate /= 10
                            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
                        else:
                            break
                    else:
                        validation_losses.append(val_loss_total) 
                print('Saving the model, end of epoch %d' % (epoch+1))
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR)
                torch.save(net.state_dict(), os.path.join(LOG_DIR,'checkpoint' + str(i) + '.pt'))
                hw3utils.visualize_batch(inputs,preds,targets,os.path.join(LOG_DIR,'example' + str(i) + '.png'))
            i += 1
            if(num_layer == 1):
                break
            
            
net = Net2(3, 8).to(device=device)
criterion = nn.MSELoss()
learning_rate = 0.1
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
train_loader, val_loader = get_loaders(batch_size,device)
validation_losses = []

for epoch in range(max_num_epoch):  
    running_loss = 0.0 # training loss of the network
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.
        optimizer.zero_grad() # zero the parameter gradients
        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        # print loss
        running_loss += loss.item()
        print_n = 100 # feel free to change this constant
        if iteri % print_n == (print_n-1):    # print every print_n mini-batches
            print('[%d, %5d] network-loss: %.3f' %
                (epoch + 1, iteri + 1, running_loss / 100))
            running_loss = 0.0
            # note: you most probably want to track the progress on the validation set as well (needs to be implemented)
        if (iteri==0) and VISUALIZE: 
            hw3utils.visualize_batch(inputs,preds,targets)
    if(epoch % 5 == 4):
        val_loss_total=0.0
        for val_inputs, val_targets in val_loader:
            val_preds = net(val_inputs)
            val_loss = criterion(val_preds, val_targets)
            val_loss_total += val_loss.item()
        print('Validation loss: ' + str(val_loss_total/100))
        print('learning rate: ' + str(learning_rate))
        if(len(validation_losses) > 0 and val_loss_total > validation_losses[-1]):
            if(learning_rate > 0.001):
                validation_losses.append(val_loss_total)
                learning_rate /= 10
                optimizer = optim.SGD(net.parameters(), lr=learning_rate)
            else:
                break
        else:
            validation_losses.append(val_loss_total)


model = Net2(3, 8).to(device=device)
model.load_state_dict(torch.load(os.path.join(LOG_DIR,'checkpoint4.pt')))
model.eval()

test_loader = get_test_loader(100, device)

for index in range(100):
    myfile = open('test_images.txt', 'a')
    path = 'ceng483-s19-hw3-dataset/test_inputs/images/' + str(index) + '.jpg'
    myfile.write(path)
    myfile.write('\n')
    myfile.close()

for test_inputs, test_targets in test_loader:
    test_preds = model(test_inputs)
    test_preds = test_preds.cpu()
    testnp = test_preds.detach().numpy()
    testnp = (testnp/2+0.5)*255
    testnp = testnp.astype('int')
    testnp = testnp.reshape(100, 80, 80, 3)
    np.save('estimations_test.npy', testnp)  
    break
