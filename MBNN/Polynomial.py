import argparse
from datetime import date
import os
import numpy as np
import copy
from datetime import datetime
from progress.bar import ChargingBar as Bar
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import sys 
sys.path.append('..')

from utils import *
from evaluate import *
from MBNN.models import *
from MBNN.MCMC import *
from MBNN.Mask_Update import *

parser = argparse.ArgumentParser(description='mBNN noisy polynomial regression')
parser.add_argument('--death_method', type=str, default= 'proposed', choices=["proposed", "Oops", "random"])
parser.add_argument('--birth_method', type=str, default= 'random', choices=["proposed", "Oops", "random"])

parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--out', default='', help='Directory to output the result')
parser.add_argument('--data_dir', default='', help='Directory of dataset')

parser.add_argument('--method', type=str, default= 'mBNN', choices=['BNN', 'mBNN'])

args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():
    
    out_directory = args.out + "/method_" + args.method
    if args.method == 'mBNN':
        out_directory += "_birth_" + args.birth_method + "_death_" + args.death_method
    
    n = 20
    p_data = 1
    num_classes=1

    x = np.random.uniform(low=-4.0, high=4.0, size=n)
    eps = np.random.normal(loc=0.0, scale=3.0, size=n)
    y = x**3 + eps
    
    L, p, prior, prior_scale = 2, 1000, "Cauchy", 0.3
    lr = 0.1
    thinning_interval = 10
    burn_in_sample = 300
    num_sample = 1
    lamb, N_max = 0.1, 3
    
    task = 'regression'
    model = M_MLP(p_data, num_classes, L, p, prior, prior_scale).cuda()  

    step_size = lr/n
    total_epoch = thinning_interval*(burn_in_sample+num_sample)
    burn_in_epoch = thinning_interval*burn_in_sample
    step_size_tuning = DualAveragingStepSize(initial_step_size=step_size)
    model_tmp = copy.deepcopy(model)

    trainloader = DataLoader(TensorDataset(torch.tensor(x).float().reshape(-1,1), torch.tensor(y).float().reshape(-1,1)), batch_size=n)
    
    model_list=[]
    res = pd.DataFrame()
    for epoch in range(total_epoch):
        model.train()
        try:
            p_accept = HMC(model, task, trainloader, lr=step_size, lf_step = 30)
            if epoch < burn_in_epoch:
                step_size, _ = step_size_tuning.update(p_accept)
            if epoch == burn_in_epoch:
                 _, step_size = step_size_tuning.update(p_accept) 
        except:
            model = copy.deepcopy(model_tmp)
        model_tmp = copy.deepcopy(model)

        if (epoch+1) % 1 ==0:
            model.eval()
            for _ in range(2):
                mask_update_dataloader(model, task, trainloader, n, lamb, N_max, args.death_method, args.birth_method)

        if task == 'regression':
            sigma_update(model, trainloader, n)

        if args.method == "mBNN":
            model.eval()            
            with torch.no_grad():
                _,ERROR_train = log_likelihood_dataloader(model, task, trainloader)

                node_sparsity_l = []
                l=0
                for name, param in model.named_parameters():
                    if 'active' in name:
                        node_sparsity_l.append(torch.sum(param.data).item())
                        l+=1
                        
            node_sparsity = np.mean(np.stack(node_sparsity_l))   

            res.loc[epoch,'n1'] = node_sparsity_l[0]
            res.loc[epoch,'n2'] = node_sparsity_l[1]
            res.loc[epoch,'n_r'] = (node_sparsity_l[0] + node_sparsity_l[1])/(2*1000)
            res.to_csv(out_directory + '/node_sparsity.csv')                 

        if (epoch + 1) > burn_in_epoch and (epoch+1-burn_in_epoch) % thinning_interval == 0: 
            model_list.append(copy.deepcopy(model))
            
    torch.save({'model_list' : model_list}, out_directory + "/models.pth")
            
if __name__ == '__main__':
    main()