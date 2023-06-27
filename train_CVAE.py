from __future__ import print_function
import argparse
import torch
import torch.utils.data as Data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from torch.autograd import Variable
import os, sys, pickle, glob
import pandas as pd
import numpy
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.distributions as D
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools 
from sklearn.metrics import classification_report
import copy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import time 
class CVAE1(nn.Module):
    def __init__(self,num_classes):
        super(CVAE1, self).__init__()
        self.l_z_xy=nn.Sequential(nn.Linear(41+num_classes, 35), nn.Softplus(),nn.Linear(35, 20), nn.Softplus(), nn.Linear(20, 2*3))
        self.l_z_x=nn.Sequential(nn.Linear(41,36),nn.Softplus(),nn.Softplus(), nn.Linear(36,20),nn.Softplus(),nn.Linear(20, 2*3))
        self.l_y_xz=nn.Sequential(nn.Linear(41+3,35),nn.Softplus(), nn.Linear(35,20),nn.Softplus(),nn.Linear(20, num_classes),nn.Sigmoid())     
        self.lb = LabelBinarizer()
    def z_xy(self,x,y):
        #y_c = self.to_categrical(y)
        xy =  torch.cat((x, y), 1)
        h=self.l_z_xy(xy)
        mu, logsigma = h.chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu, logsigma       
    def z_x(self,x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())
        #return mu,logsigma      
    def y_xz(self,x,z):
        xz=torch.cat((x, z), 1)
        #return D.Bernoulli(self.y_xz(xz))
        return self.l_y_xz(xz)   
    def forward(self, x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return self.l_y_xz(torch.cat((x, mu), 1))
  
  
def loss_func(z_xy, z_x,y_xz,y):
    KLD = D.kl.kl_divergence(z_xy,z_x) 
    #KLD=torch.sum(KLD)
    loss=nn.BCELoss(reduction='sum').cuda()
    BCE = loss(y_xz, y)   
    return (torch.sum(KLD)+BCE)/y.size(0)

def Stage1Train(train_loader,device,cuda,num_epoch):
        #Sets the module in training mode.
    
    max_label = -1  # 初始化最大Label编号为-1
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data.float())
        label = Variable(label.long())
        label = torch.unsqueeze(label, 1)
        # print(label.size())
        max_label = max(max_label, torch.max(label).item())  # 更新最大Label编号
    
    num_classes = max_label + 1  # 确定One-Hot编码的类别数目
    model = CVAE1(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
    model.train()
    train_loss = 0
    for epoch in range(num_epoch):
        epoch_loss = 0.0  # 用于累计每个epoch的总损失

        for batch_idx, (data, label) in enumerate(train_loader):
            data = Variable(data.float())
            label = Variable(label.long())
            label = torch.unsqueeze(label, 1)
            
            label = torch.zeros(label.size()[0], num_classes).scatter_(1, label, 1).cuda()
            
            if cuda:
                data = data.cuda()
            
            optimizer.zero_grad()
            z_xy = model.z_xy(data, label)
            z_x = model.z_x(data)
            z = z_xy.rsample()
            y_xz = model.y_xz(data, z)

            loss = loss_func(z_xy, z_x, y_xz, label)
            loss.backward()
            train_loss += loss.item()
            epoch_loss += loss.item()  # 累计每个batch的损失

            optimizer.step()

        average_loss = epoch_loss / len(train_loader)  # 计算平均损失

        print('Epoch [%d/%d] Average Loss: %.4f' % (epoch + 1, num_epoch, average_loss))

    torch.save(model, 'minmax_f3_CVAE1_setting4.pkl')  




  
def main(argv):

    no_cuda = False
    cuda_available = not no_cuda and torch.cuda.is_available()

    BATCH_SIZE = 50
    EPOCH = 100
    SEED = 1234

    torch.manual_seed(SEED)

    device = torch.device("cuda" if cuda_available else "cpu")

    # 加载Label编号含义字典
    with open('./Data/value_dict.pkl', 'rb') as f:
        value_dicts = pickle.load(f)

    # 定义需要筛选的Label编号

    # setting1
    # setting1 = ['normal', 'back', 'ipsweep', 'neptune', 'nmap', 'portsweep', 'satan', 'smurf', 'teardrop']

    # 实际为setting2，变量名懒得改了
    # setting1 = ['normal','ipsweep', 'neptune', 'nmap', 'portsweep','smurf', 'teardrop','guess_passwd','warezclient']

    # setting3
    # setting1 =  ['normal','nmap', 'neptune','satan','pod','back','ipsweep','teardrop']

    # setting4
    setting1 = ['normal','ipsweep','portsweep','teardrop','nmap','pod','smurf','warezclient']
    

    datatrain = np.load('./Data/NSLKDD_train_setting4.npy')
    xtrain = datatrain[:,:-2]  
    ytrain = datatrain[:,-2]



    testdatatest = np.load('./Data/NSLKDD_test_new.npy')
    print(testdatatest.shape)

        # 定义未知攻击类型的编号
    unknown_label = len(value_dicts[41])  # 使用当前编号字典中的最大编号+1

    # 未知攻击标签处理
    selected_samples = []
    for sample in testdatatest:
        label_idx = int(sample[-2])
        if label_idx in value_dicts[41].values():
            label = list(value_dicts[41].keys())[list(value_dicts[41].values()).index(label_idx)]
            if label not in setting1:
                sample[-2] = unknown_label
                selected_samples.append(sample)
            else:
                selected_samples.append(sample)

    # 转换为ndarray对象
    testdatatest = np.array(selected_samples)
    print(testdatatest.shape)
    xtest = testdatatest[:,:-2]
    ytest1 = testdatatest[:,-2]
    xtest = preprocessing.MinMaxScaler().fit(xtrain).transform(xtest)
    #xunknown = preprocessing.MinMaxScaler().fit(xtrain).transform(xunknown)
    minmaxscaler=preprocessing.MinMaxScaler().fit(xtrain)
    scale=minmaxscaler.scale_ 
    scale=torch.Tensor(scale)
    
    xtrain = preprocessing.MinMaxScaler().fit_transform(xtrain)
    xtrain=torch.from_numpy(xtrain)
    ytrain = torch.from_numpy(ytrain)    
    train_dataset = Data.TensorDataset(xtrain, ytrain)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)                       
    Stage1Train(train_loader,device,cuda_available,num_epoch=150)  
    print("training done")


  

if __name__ == "__main__":
    start = time.time()
    main(sys.argv)                
    end = time.time()
    print('elapsed time:',end - start)