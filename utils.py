import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def data_process(dataname, data_dir, seed=0, batch_size=100):
    if dataname == "Boston":
        data =  pd.read_csv(data_dir + "/housing.data", header=None, sep="\s+")
        data_x = MinMaxScaler((0,1)).fit_transform(data.iloc[:, :-1].astype(np.float64))
        data_y=np.array(data.iloc[:,-1]).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 1

    elif dataname == "Concrete":
        data=pd.read_csv(data_dir + "/Concrete_Data.csv", header=None)
        data_x = MinMaxScaler((0,1)).fit_transform(data.iloc[:, :-1].astype(np.float64))
        data_y=np.array(data.iloc[:,-1]).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 1
    
    elif dataname == "Energy":
        data =  pd.read_csv(data_dir + "/ENB2012_data.csv", header=None)
        data_x = MinMaxScaler((0,1)).fit_transform(data.iloc[:, :-1].astype(np.float64))
        data_y=np.array(data.iloc[:,-1]).reshape(-1,1)   
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 1
    
    elif dataname == "Yacht":
        data =  pd.read_csv(data_dir + "/yacht_hydrodynamics.data", header=None, sep="\s+")
        data_x = MinMaxScaler((0,1)).fit_transform(data.iloc[:, :-1].astype(np.float64))
        data_y=np.array(data.iloc[:,-1]).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 1
        
    elif dataname == "Haberman":
        data = pd.read_csv(data_dir + "/haberman.data", header=None)
        data_x = MinMaxScaler((0,1)).fit_transform(pd.get_dummies(data.iloc[:, :-1]).astype(np.float64))
        data_y = np.array(data.iloc[:,-1]-1).reshape(-1,1)        
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 2
        
    elif dataname == "Retinopathy":
        data=pd.read_csv(data_dir + "/messidor_features.data", header=None)
        data_x = MinMaxScaler((0,1)).fit_transform(pd.get_dummies(data.iloc[:, :-1]).astype(np.float64))
        data_y = np.array(data.iloc[:,-1]).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 2
    
    elif dataname == "Tic-tac-toe":
        data=pd.read_csv(data_dir + "/tic-tac-toe.data", header=None)
        data_x = MinMaxScaler((0,1)).fit_transform(pd.get_dummies(data.iloc[:, :-1]).astype(np.float64))
        data_y = np.array(pd.Series(np.where(data.iloc[:,-1]=="positive",1,0))).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 2
        
    elif dataname == "Promoter":
        data=pd.read_csv(data_dir + "/promoters.data", header=None)
        gene=[]
        for i in range(data.shape[0]):
            gene.append(list(data.iloc[i,2].replace('\t', '')))
        data_x = MinMaxScaler((0,1)).fit_transform(pd.get_dummies(pd.DataFrame(gene),drop_first=True).astype(np.float64))
        data_y = np.array(pd.Series(np.where(data.iloc[:,0]=="+",1,0))).reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=seed)
        x_train, x_test, y_train, y_test = torch.tensor(x_train).float(), torch.tensor(x_test).float(), torch.tensor(y_train).float(), torch.tensor(y_test).float()

        if batch_size == -1:
            train_batch_size = x_train.shape[0]
        else:
            train_batch_size = batch_size
        trainloader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=train_batch_size)
        test_batch_size = x_test.shape[0]
        testloader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size)
        n_train, n_test, p_data, num_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], 2

        
    elif dataname == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if batch_size<0:
            raise Exception('Invalid batch size.')

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        n_train, n_test, p_data, num_classes = len(trainset), len(testset), None, 10
        
        

    elif dataname == "CIFAR100":
        
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])

        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        n_train, n_test, p_data, num_classes = len(trainset), len(testset), None, 100
        
        
    elif dataname == "SVHN":        
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        n_train, n_test, p_data, num_classes = len(trainset), len(testset), None, 100
        
        
    return trainloader, testloader, n_train, n_test, p_data, num_classes



def filter_ranks(conv, bn):
    bn_scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    new_filter = bn_scale.reshape(-1,1,1,1) * conv.weight.data
    new_bias = bn.bias.data - bn_scale*bn.running_mean
    return (torch.pow(new_filter, 2)).sum((1,2,3)) #+ torch.pow(new_bias,2)# 
   