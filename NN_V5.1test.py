# Heavily based on https://github.com/Prodicode/ann-visualizer
import torch

import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from math import cos, sin, atan
from torch import nn, optim
import copy
import torch.nn.functional as F
import torchvision.transforms as transform


"""
Param for the graphics
"""
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
title_font = {'size':'19', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
axis_font = {'size':'19'}
legend_font={'size':'15'}
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8,6.5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
"""
function to write the NN parameter to FPGA readable format
+converterd back to do
"""

def twosCom_binDec(bin1, digit):
        while len(bin1)<digit :
                bin1 = '0'+bin1
        if bin1[0] == '0':
                return int(bin1, 2)
        else:
                return -1 * (int(''.join('1' if x == '0' else '0' for x in bin1), 2) + 1)

def twosCom_decBin(dec, digit):
        dec = int(dec*(2**(digit-1)))
        if abs(dec)>2**(digit-1):
            dec = int(2**(digit-1)*dec/abs(dec))
        if dec>=0:
                bin1 = bin(dec).split("0b")[1]
                while len(bin1)<digit:
                        bin1 = '0'+bin1
        else:
                bin1 = -1*dec
                bin1 = bin(bin1-pow(2,digit)).split("0b")[1]
        return bin1,dec
def mult_model(dec, digit):
        dec = int(dec*(2**(digit-1)))
        if abs(dec)>2**(digit-1):
            dec = int(2**(digit-1)*dec/abs(dec))
        if dec>=0:
                bin1 = bin(dec).split("0b")[1]
                while len(bin1)<digit:
                        bin1 = '0'+bin1
        else:
                bin1 = -1*dec
                bin1 = bin(bin1-pow(2,digit)).split("0b")[1]
        return bin1,dec

def dic_to_txt(a_dict,filename,digit):
    f= open(filename,"w+")
    a_dicti = copy.deepcopy(a_dict)
    for key in a_dict:
        f.write("******************************************************************"+'\n')
        f.write("************"+key+"**********"+'\n')
        f.write("******************************************************************"+'\n')
        tensor = a_dict[key]
        #print(tensor)
        np_tensor = tensor.numpy()
        for idx, x in np.ndenumerate(tensor):
            b,dec= twosCom_decBin(x, digit)
            val2=twosCom_binDec(b, digit)
            a_dicti[key][idx] = val2
            if val2 != dec:
                print(val2)
                print(dec)
                a=input("error conversion for binary "+str(digit))
            f.write(key+'_'+str(idx)+' = '+ str(b)+' ;'+'\n')
    return a_dicti
            
def dic_to_txt_int(a_dict,filename,digit):
    f= open(filename,"w+")
    a_dicti = copy.deepcopy(a_dict)
    for key in a_dict:
        f.write("******************************************************************"+'\n')
        f.write("************"+key+"**********"+'\n')
        f.write("******************************************************************"+'\n')
        tensor = a_dict[key]
        #print(tensor)
        np_tensor = tensor.numpy()
        for idx, x in np.ndenumerate(tensor):
            a_dicti[key][idx] = int(a_dict[key][idx]*(2**(digit-1)))
            f.write(key+'_'+str(idx)+' = '+ str(int(a_dict[key][idx]*(2**(digit-1))))+' ;'+'\n')
    return a_dicti
            
def dic_to_txt2(a_dict,filename,digit):
    f= open(filename,"w+")
    a_dicti = copy.deepcopy(a_dict)

    for key in a_dict:
        f.write("******************************************************************"+'\n')
        f.write("************"+key+"**********"+'\n')
        f.write("******************************************************************"+'\n')
        tensor = a_dict[key]
        #print(tensor)
        np_tensor = tensor.numpy()
        for idx, x in np.ndenumerate(tensor):
            val2=x*digit
            a_dicti[key][idx] = val2
            f.write(key+'_'+str(idx)+' = '+ str(val2)+' ;'+'\n')
    return a_dicti


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(20, 20)
    self.fc2 = nn.Linear(20, 2)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.softmax(self.fc2(x),dim=1)
    return x

net = Net().float()
print(net)
#network = DrawNN( [20,20,2] )
#network.draw()
import torch
import torchvision
import torchvision.transforms as transform

f = h5py.File('data/dark_uni.h5', 'r')

train_set =torch.from_numpy(np.array(f['train_data'],dtype=float)).float()
train_label=torch.from_numpy(np.array(f['train_label'],dtype=float)).type(torch.LongTensor)
size_data_set= int(len(train_set))
n_epoch = 100
criterion = nn.CrossEntropyLoss()
trainloader=[]
k=0
 

for i in range(size_data_set):
    trainloader.append([train_set[i],train_label[i,1]])
    
batch_size = 500

"""
loading data test
"""
test_set = torch.from_numpy(np.array(f['test_data'],dtype=float)).float()
test_label = torch.from_numpy(np.array(f['test_label'],dtype=float)).type(torch.LongTensor)
testloader=[]

for i in range(20000):
    testloader.append([test_set[i],test_label[i,1]]) 

testloader2 = torch.utils.data.DataLoader(testloader, batch_size=batch_size,
                                          shuffle=False, num_workers=0)


ls_batch=[]
ls_loss=[]
ls_epoch=[]
lstest=[]
ls_lr=[]
num_steps = 5
n_lr = 5
sep_lr = int(n_epoch/n_lr)
learning_rate = 0.00001#exp_decay(epoch)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1) # optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
steps =  int(size_data_set/batch_size/num_steps)

for epoch in range(n_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    trainloader2 = torch.utils.data.DataLoader(trainloader, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
    for i, data in enumerate(trainloader2, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs,labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #scheduler.step()
        # print statistics
        running_loss += loss.item()
        if i % steps == steps-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / steps))
            ls_loss.append(running_loss/steps)
            ls_batch.append(k)
            running_loss = 0.0
            k=k+1
            ls_lr.append(np.log(learning_rate))

    """
    testing of my neural network
    """
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        for data in testloader2:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    lstest.append(100.0 * correct / total)
    ls_epoch.append(epoch)
    print('Accuracy of the network on the 20000 test events: %d %%' % (
        100.0 * correct / total))


"""
graphic for the loss 
"""
fig = plt.figure()
plt.plot(ls_batch,ls_loss)
plt.xlabel(r"Number of mini-batches", **axis_font)
plt.ylabel("Loss", **axis_font)
plt.title(r"Loss in function of the number of mini-batches used for training", **title_font)
plt.legend(loc=2,prop =legend_font)
plt.grid(True)
plt.savefig('loss_uni.png', bbox_inches='tight')
plt.close(fig)

"""
graphic for the prediction 
"""
fig = plt.figure()
plt.plot(ls_epoch,lstest)
plt.xlabel(r"epoch", **axis_font)
plt.ylabel("validated prediction", **axis_font)
plt.title(r"testing results after each epoch", **title_font)
plt.legend(loc=2,prop =legend_font)
plt.grid(True)
plt.savefig('prediction_uni.png', bbox_inches='tight')
plt.close(fig)

a_dict_net = net.state_dict()
torch.save(a_dict_net,"cnn_noformat_uni.pt")
model_None= Net().float()
model_None.load_state_dict(torch.load('cnn_noformat_uni.pt'))

ls_error=[]
ls_batches=[]
ls_accuray=[]


ls_digit = [2,3,4,5,6,7,8,12,16,20,32]
for i in range(len(ls_digit)):
    correct=0.0
    total=0.0
    digit=ls_digit[i]
    filename = "To_FPGA_uni_"+str(digit)+".txt"
    filename3 = "To_FPGA_uni_"+str(digit)+"dec.txt"
    filename2 = "cnn_uni"+str(digit)+".pt"
    a_dict = model_None.state_dict()
    a_dicti= dic_to_txt(a_dict,filename,digit)
    a_dictii= dic_to_txt_int(a_dict,filename3,digit)
    torch.save(a_dictii,filename2)
    model_dicti = Net().float()
    model_dicti.load_state_dict(torch.load(filename2))
    a_dictiii = model_dicti.state_dict()
    print("coucou")
    for key in a_dicti:
        for idx, x in np.ndenumerate(a_dicti[key]):
            val = a_dictiii[key][idx]
            val_2 = a_dicti[key][idx]#int(a_dicti[key][idx]*(2**digit-1))
            if val_2 != val :
                print(val)
                print(val_2)

                a=input("error conversion "+str(digit))

    ls_i=[]
    ls_res=[]
    with torch.no_grad():
        for i,data in enumerate(testloader2, 0):
            images, labels = data
            outputs = model_dicti(images)
            #outputs=[float, float]
            #labels = [0,1] or [1,0]
            ls_i.append(i)
            results, prediction = torch.max(outputs.data, 1) #if the value predicted is a event return 1 into prediction varibale.
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        print('Accuracy of the network on the 20000 test events: %d %%' % (100.0 * correct / total))
        ls_accuray.append(100.0 * correct / total)

    ls_error.append(ls_res)
    ls_batches.append(ls_i)
    a_dicti = None
    model_dicti = None

ls_error2=[]
ls_batches2=[]
ls_accuray2 =[]
ls_digit2 = [100,1000,10000,100000,1000000]

for i in range(len(ls_digit)):
    correct=0.0
    total=0.0
    digit=ls_digit[i]

    filename = "To_FPGA_uni_"+str(digit)+".txt"
    filename3 = "To_FPGA_uni_"+str(digit)+"dec.txt"
    filename2 = "cnn_uni"+str(digit)+".pt"
    a_dict = model_None.state_dict()
    a_dicti= dic_to_txt2(a_dict,filename,digit)
    torch.save(a_dicti,filename2)
    model_dicti = Net().float()
    model_dicti.load_state_dict(torch.load(filename2))
    a_dictiii = model_dicti.state_dict()
    for key in a_dicti:
        for idx, x in np.ndenumerate(a_dicti[key]):
            val = a_dictiii[key][idx]
            val_2 = a_dict[key][idx]*digit
            if val_2 != val :
                print(val)
                print(val_2)

                a=input("error conversion "+str(digit))
    ls_i=[]
    ls_res=[]
    with torch.no_grad():
        for i,data in enumerate(testloader2, 0):
            images, labels = data
            outputs = model_dicti(images)
            #outputs=[float, float]
            #labels = [0,1] or [1,0]
            ls_i.append(i)
            results, prediction = torch.max(outputs.data, 1) #if the value predicted is a event return 1 into prediction varibale.
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        print('Accuracy of the network on the 20000 test events: %d %%' % (100.0 * correct / total))
        ls_accuray2.append(100.0 * correct / total)

    ls_error2.append(ls_res)
    ls_batches2.append(ls_i)
    a_dicti = None
    model_dicti = None





with torch.no_grad():
    ls_i=[]
    ls_res=[]
    correct=0.0
    total=0.0
    for i,data in enumerate(testloader2, 0):
        images, labels = data
        outputs = model_None(images)
        results, prediction = torch.max(outputs.data, 1) #if the value predicted is a event return 1 into prediction varibale.
        total += labels.size(0)
        correct += (prediction == labels).sum().item()
    print('Accuracy of the network on the 20000 test events: %d %%' % (100.0 * correct / total))          
    ls_accuray2.append(100.0 * correct / total)
    ls_accuray.append(100.0 * correct / total)
ls_error.append(ls_res)
ls_error2.append(ls_res)
ls_digit.append("None")
ls_digit2.append("None")



fig = plt.figure()
positions=[]
for i in range(len(ls_digit)):
    positions.append(i)
    plt.plot(i,ls_accuray[i],'o',label = str(ls_digit[i]))
plt.xticks(positions, ls_digit)
plt.xlabel(r"BITS", **axis_font)
plt.ylabel("Precision in \%", **axis_font)
plt.title(r"testing results for different data format uni", **title_font)
#plt.legend(loc=2,prop =legend_font)
plt.grid(True)
plt.savefig('comparaison_data_formats_otherform_uni.pdf', bbox_inches='tight')
plt.close(fig)

fig = plt.figure()
positions=[]
for i in range(len(ls_digit2)):
    positions.append(i)
    plt.plot(i,ls_accuray2[i],'o',label = str(ls_digit[i]))
plt.xticks(positions, ls_digit2)
plt.xlabel(r"mutlifly factor", **axis_font)
plt.ylabel("Precision in \%", **axis_font)
plt.title(r"testing results for different data format  uni", **title_font)
#plt.legend(loc=2,prop =legend_font)
plt.grid(True)
plt.savefig('comparaison_multiply_uni.pdf', bbox_inches='tight')
plt.close(fig)

