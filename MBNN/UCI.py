import argparse
from datetime import date
import os
import numpy as np
import copy
from datetime import datetime
from progress.bar import ChargingBar as Bar
import pickle

import sys 
sys.path.append('..')

from utils import *
from evaluate import *
from MBNN.models import *
from MBNN.MCMC import *
from MBNN.Mask_Update import *

parser = argparse.ArgumentParser(description='mBNN for UCI dataset')

########################## model setting ##########################
parser.add_argument('--model', type=str, default= 'MLP', choices=['MLP', 'resnet18'], help='architecture of model')
parser.add_argument('--L', type=int, default= 2, help='depth of MLP')
parser.add_argument('--p', type=int, default= 1000, help='width of MLP')
parser.add_argument('--prior', type=str, default= 'Cauchy', choices=['Normal', 'Cauchy'], help='type of prior')
parser.add_argument('--prior_scale', type=float, default= 1.0, help='scale of prior')


########################## basic setting ##########################
parser.add_argument('--start_seed', type=int, help='start_seed')
parser.add_argument('--end_seed', type=int, help='end_seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='', help='Directory of dataset')
parser.add_argument('--out', default='', help='Directory to output the result')


######################### Dataset setting #############################
parser.add_argument('--dataset', type=str, default= 'Boston', choices=['Boston', 'Concrete', 'Energy', 'Yacht'], help='dataset name')
parser.add_argument('--batch_size', default=-1, type=int, help='train batchsize')


######################### MCMC setting #############################
parser.add_argument('--num_sample', default=20, type=int, help='the number of MCMC sample')
parser.add_argument('--burn_in_sample', default=2, type=int, help='the number of MCMC burn_in sample')
parser.add_argument('--thinning_interval', default=200, type=int, help='thinning_interval epoch')
parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--lf_step', default=20, type=int, help='the number of leapfrog step')

######################### Mask setting ###############################
parser.add_argument('--update_period', default=1, type=int, help='period of mask update')
parser.add_argument('--num_update', default=10, type=int, help='number of times to update at a time')
parser.add_argument('--lamb', default=0.1, type=float, help='hyperparameter for the prior of sparsity')
parser.add_argument('--N_max', default=3, type=int, help='maximum of the number of updated mask')

parser.add_argument('--death_method', type=str, default= 'proposed', choices=["proposed", "Oops", "random"])
parser.add_argument('--birth_method', type=str, default= 'random', choices=["proposed", "Oops", "random"])

######################### add name #############################
parser.add_argument('--add_name', default='', type=str, help='add_name')


args = parser.parse_args()
print(args)

torch.cuda.set_device(args.gpu)

def main():
    
    out_directory = args.out + '/MBNN' + '/' + str(args.dataset)
    out_directory += '/' + str(date.today().strftime('%Y%m%d')[2:])
    
    out_directory += '/' + '_lam' + str(args.lamb) 

    out_directory += '_bm_' + args.birth_method + '_dm_' + args.death_method

    if args.add_name != '':
        out_directory +='_'+str(args.add_name)

    if not os.path.isdir(out_directory):
        mkdir_p(out_directory)
        
    result1_list = []
    result2_list = []
    result3_list = []
    result4_list = []
    result5_list = []      
    for seed in range(args.start_seed, args.end_seed+1):           
        print("seed : ", seed)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
            
        #dataset
        trainloader, testloader, n_train, n_test, p_data, num_classes = data_process(args.dataset, args.data_dir, seed, args.batch_size)
        std = torch.std(next(iter(trainloader))[1])                

        model = M_MLP(p_data, num_classes, args.L, args.p, args.prior, args.prior_scale).cuda()         
          
        #MCMC
        step_size = args.lr/n_train
        total_epoch = args.thinning_interval*(args.burn_in_sample+args.num_sample)
        burn_in_epoch = args.thinning_interval*args.burn_in_sample
        mt=0
        bar = Bar('{:>10}'.format('Training'), max=total_epoch)                
        step_size_tuning = DualAveragingStepSize(initial_step_size=step_size)
        model_tmp = copy.deepcopy(model)

        res = pd.DataFrame()
        for epoch in range(total_epoch):
            try:
                p_accept = HMC(model, task, trainloader, lr=step_size, lf_step = args.lf_step)
                if epoch < burn_in_epoch:
                    step_size, _ = step_size_tuning.update(p_accept)
                if epoch == burn_in_epoch:
                     _, step_size = step_size_tuning.update(p_accept) 
            except:
                model = copy.deepcopy(model_tmp)
            model_tmp = copy.deepcopy(model)

            if (epoch+1) % args.update_period ==0:
                model.eval()
                for _ in range(args.num_update):
                    mask_update_dataloader(model, task, trainloader, n_train, args.lamb, args.N_max, args.death_method, args.birth_method)

            sigma_update(model, trainloader, n_train, std)
                
            if (epoch+1)%(20)==0:
                model.eval()            
                with torch.no_grad():
                    _,ERROR_train = log_likelihood_dataloader(model, task, trainloader)
                    _,ERROR_test = log_likelihood_dataloader(model, task, testloader)

                    ERROR_train, ERROR_test = np.sqrt(ERROR_train.item()/n_train), np.sqrt(ERROR_test.item()/n_test)

                    node_sparsity_l = []
                    l=0
                    for n, p in model.named_parameters():
                        if 'active' in n:
                            node_sparsity_l.append(torch.sum(p.data).item())
                            l+=1
                    node_sparsity = np.mean(np.stack(node_sparsity_l))                    

                    bar.suffix  = '({epo}/{total_epo}) ERR_train:{ER_tr} | ERR_test:{ER_te} | {ns}'.format(
                                    epo=epoch + 1,
                                    total_epo=total_epoch,                            
                                    ER_tr=np.round(ERROR_train,3),
                                    ER_te=np.round(ERROR_test,3),
                                    ns = np.round(node_sparsity,2)
                                    )
                res.loc[epoch,'n1'] = node_sparsity_l[0]
                res.loc[epoch,'n2'] = node_sparsity_l[1]
                res.loc[epoch,'n_r'] = (node_sparsity_l[0] + node_sparsity_l[1])/(2*args.p)
                res.to_csv(out_directory + '/node_sparsity_seed_%d.csv'%(seed,))

            if (epoch + 1) > burn_in_epoch and (epoch+1-burn_in_epoch) % args.thinning_interval == 0: 
                torch.save(model.state_dict(), out_directory + '/seed_%d_mt_%d.pt'%(seed, mt))
                mt += 1

            bar.next()            
        bar.finish()    

        print("model testing")
        pred_list=[]
        target_list=[]
        sigma_list=[]        
        with torch.no_grad():
            for mt in range(args.num_sample):
                model = M_MLP(p_data, num_classes, args.L, args.p, args.prior, args.prior_scale).cuda()
                model.load_state_dict(torch.load(out_directory + '/seed_%d_mt_%d.pt'%(seed,mt)))
                model.eval()
                
                pred = []
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    pred.append(outputs.squeeze())                

                    if mt==0:
                        target_list.append(targets.squeeze())

                sigma_list.append(model.sigma.data)

                pred_list.append(torch.cat(pred,0))    
                
            pred_list = torch.stack(pred_list)
            target_list = torch.cat(target_list,0)
            sigma_list = torch.cat(sigma_list,0)            
            RMSE, m_NLL, CRPS = evaluate_averaged_model_regression(pred_list, target_list, sigma_list)
            
            print("RMSE : ", RMSE, " m_NLL : ", m_NLL, " CRPS : ", CRPS)
            result1_list.append(RMSE)
            result2_list.append(m_NLL)
            result3_list.append(CRPS)
            
    num_seed = args.end_seed - args.start_seed
    result1_list, result2_list, result3_list = np.stack(result1_list), np.stack(result2_list), np.stack(result3_list)
    print("%.3f(%.3f), %.3f(%.3f), %.3f(%.3f)" % (np.mean(result1_list), np.std(result1_list)/np.sqrt(num_seed),np.mean(result2_list), np.std(result2_list)/np.sqrt(num_seed),np.mean(result3_list), np.std(result3_list)/np.sqrt(num_seed)))              
                    
if __name__ == '__main__':
    main()
    
    
    
