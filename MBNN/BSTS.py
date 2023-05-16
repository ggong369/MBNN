import pandas as pd
import numpy as np

import argparse
import csv
from datetime import datetime
from torch import optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer
from Kalman_Filter import *
from MCMC import *
from Mask_Update import *
from models import *
import matplotlib.pyplot as plt
import seaborn as sns

import sys 
sys.path.append('..')

from utils import *


parser = argparse.ArgumentParser(description='BSTS')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--out', default='', help='Directory to output the result')
parser.add_argument('--data_dir', default='', help='Directory of dataset')

parser.add_argument('--method', type=str, default= 'mBNN', choices=['linear', 'BNN', 'mBNN'])

def logit(x):
    return np.log(x/(1-x))

args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():
    
    out_directory = args.out
    
    data_y = np.array(pd.read_csv(args.data_dir + '/y_search.csv'))
    data_x = np.array(pd.read_csv(args.data_dir + '/x_search.csv'))
    
    data_x = MinMaxScaler((0,1)).fit_transform(np.stack(data_x).T)
    data_y = np.sum(np.stack(data_y),0)/1000
    n = len(data_y)
    n_train = 240
    n_test = n - n_train

    data_x_train = data_x[:n_train]
    data_y_train = data_y[:n_train]
    data_x_test = data_x[n_train:]
    data_y_test = data_y[n_train:]

    Z = np.ones(shape=(1,1))
    H = np.ones(shape=(1,1))
    T = np.ones(shape=(1,1))*0.95
    Q = np.ones(shape=(1,1))*0.1
    a_0 = np.zeros(shape=(1))
    P_0 = np.ones(shape=(1,1))
    
    if args.method == linear:
        thinning_interval = 10
        burn_in_sample = 10
        num_sample = 20
        total_epoch = thinning_interval*(burn_in_sample+num_sample)
        burn_in_epoch = thinning_interval*burn_in_sample
        
        beta_list = []
        filtered_state_list = []
        filtered_state_cov_list = []

        data_x_train_one = np.c_[np.ones((data_x_train.shape[0], 1)),data_x_train]
        inv = np.linalg.inv(data_x_train_one.T @ data_x_train_one + np.identity(data_x_train_one.shape[1]))
        diff = data_y_train

        for epoch in range(total_epoch):
            beta_mean = (inv @ data_x_train_one.T @ diff.reshape(-1,1)).squeeze()
            beta = np.random.multivariate_normal(beta_mean, inv)

            predicted = (data_x_train_one @ beta).squeeze()

            ERROR = ((diff - predicted)**2).sum()
            gamma = np.random.gamma(1 + n_train/2, 1/(ERROR/2 + 1))
            H = 1/gamma

            y = (data_y_train-predicted).reshape(1,-1)
            filtered_state, filtered_state_cov, _, _, _, _, _, _ = kalman_filter(y, Z, H, T, Q, a_0, P_0)
            sampled_mu = np.random.normal(loc=filtered_state.squeeze(), scale = np.sqrt(filtered_state_cov.squeeze()))
            diff = data_y_train - sampled_mu

            if (epoch+1) % thinning_interval == 0:
                print("epoch : ", epoch, ", RMSE : ", np.sqrt(np.mean((diff-predicted)**2)))
                if (epoch+1) > burn_in_epoch:
                    beta_list.append(beta)
                    filtered_state_list.append(filtered_state)
                    filtered_state_cov_list.append(filtered_state_cov)  
                    
        torch.save({'beta_list' : beta_list, 'filtered_state_list': filtered_state_list, 'filtered_state_cov_list':filtered_state_cov_list}, out_directory + "/BSTS_linear.pth")
        
    else:
        L, p, prior, prior_scale = 2, 100, "Cauchy", 1.0
        lr = 1e-5
        thinning_interval = 10
        burn_in_sample = 10
        num_sample = 20
        lamb, N_max = 0.1, 3
        num_classes=1
        
        x_train, y_train, x_test, y_test = torch.tensor(data_x_train).float(), torch.tensor(data_y_train).float(), torch.tensor(data_x_test).float(), torch.tensor(data_y_test).float()
        trainloader, testloader = DataLoader(TensorDataset(x_train, y_train), batch_size=x_train.shape[0]), DataLoader(TensorDataset(x_test, y_test), batch_size=x_test.shape[0])
        p_data = x_train.shape[1]

        task = 'regression'
        model = M_MLP(p_data, num_classes, L, p, prior, prior_scale).cuda()  
        step_size = lr/n
        total_epoch = thinning_interval*(burn_in_sample+num_sample)
        burn_in_epoch = thinning_interval*burn_in_sample
        step_size_tuning = DualAveragingStepSize(initial_step_size=step_size)
        model_tmp = copy.deepcopy(model)
        
        model_list = []
        filtered_state_list=[]
        filtered_state_cov_list=[]

        for epoch in range(total_epoch):
            model.train()
            try:
                p_accept = HMC(model, task, trainloader, lr=step_size, lf_step = 50)
                if epoch < burn_in_epoch:
                    step_size, _ = step_size_tuning.update(p_accept)
                if epoch == burn_in_epoch:
                    _, step_size = step_size_tuning.update(p_accept)
            except:
                model = copy.deepcopy(model_tmp)
            model_tmp = copy.deepcopy(model)   

            if args.method == "mBNN":
                model.eval()   
                for _ in range(2):
                    mask_update_dataloader(model, task, trainloader, n, lamb, N_max)


            predicted = []
            for _, (inputs, _) in enumerate(trainloader):
                inputs = inputs.cuda()
                outputs = model(inputs)
                predicted.append(outputs)

            predicted = torch.vstack(predicted).detach().cpu().numpy().squeeze()
            sigma_update(model, trainloader, n_train)

            H = model.sigma.item()**2    

            y = (data_y_train-predicted).reshape(1,-1)
            filtered_state, filtered_state_cov, _, _, _, _, _, _ = kalman_filter(y, Z, H, T, Q, a_0, P_0)
            sampled_mu = np.random.normal(loc=filtered_state.squeeze(), scale = np.sqrt(filtered_state_cov.squeeze()))
            diff = data_y_train - sampled_mu
            y_train = torch.tensor(diff).float().reshape(-1,1)
            trainloader = DataLoader(TensorDataset(x_train, y_train), batch_size=n_train)

            if (epoch+1) % thinning_interval == 0:          
                if (epoch+1) > burn_in_epoch:
                    model_list.append(copy.deepcopy(model))
                    filtered_state_list.append(filtered_state)
                    filtered_state_cov_list.append(filtered_state_cov)
                    
        torch.save({'model_list' : model_list, 'filtered_state_list': filtered_state_list, 'filtered_state_cov_list':filtered_state_cov_list}, out_directory + "/BSTS_"+args.method".pth")

                        
if __name__ == '__main__':
    main()