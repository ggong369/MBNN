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

parser = argparse.ArgumentParser(description='mBNN for image dataset')

########################## model setting ##########################
parser.add_argument('--model', type=str, default= 'resnet18', choices=['resnet18'], help='architecture of model')
parser.add_argument('--prior', type=str, default= 'Cauchy', choices=['Cauchy'], help='type of prior')
parser.add_argument('--prior_scale', type=float, default= 0.1, help='scale of prior')


########################## basic setting ##########################
parser.add_argument('--start_seed', type=int, help='start_seed')
parser.add_argument('--end_seed', type=int, help='end_seed')
parser.add_argument('--gpu', default=0, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_dir', default='', help='Directory of dataset')
parser.add_argument('--out', default='', help='Directory to output the result')


######################### Dataset setting #############################
parser.add_argument('--dataset', type=str, default= 'CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='benchmark dataset')
parser.add_argument('--batch_size', default=100, type=int, help='train batchsize')


######################### MCMC setting #############################
parser.add_argument('--num_sample', default=5, type=int, help='the number of MCMC sample')
parser.add_argument('--burn_in_sample', default=5, type=int, help='the number of MCMC burn_in sample')
parser.add_argument('--thinning_interval', default=20, type=int, help='thinning_interval epoch')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--temperature', default=(1/50000)**.5, type=float, help='temperature for SGLD')

######################### Mask setting ###############################
parser.add_argument('--update_period', default=10, type=int, help='period of mask update (in batch)')
parser.add_argument('--num_update', default=1, type=int, help='number of times to update at a time')
parser.add_argument('--lamb', default=0.05, type=float, help='hyperparameter for the prior of sparsity')
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
        task = 'classification' 

        model = M_ResNet18(num_classes, args.prior, args.prior_scale).cuda()
        initial_nodes = []    
        for n, p in model.named_parameters():
            if 'active' in n:
                initial_nodes.append(torch.sum(p.data).item())   
        
        #MCMC
        step_size = args.lr/args.batch_size
        total_epoch = args.thinning_interval*(args.burn_in_sample+args.num_sample)
        burn_in_epoch = args.thinning_interval*args.burn_in_sample
        optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=burn_in_epoch, eta_min=step_size/100)
        
        mt=0
        bar = Bar('{:>10}'.format('Training'), max=total_epoch)              

        res = pd.DataFrame({'mean_ns':[],'total_ns':[]})        
        for epoch in range(total_epoch):
            model.train()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                optimizer.zero_grad()
                loss_noise = noise_loss(model,step_size)*args.temperature
                loss = -log_prob_data(model, task, inputs, targets, n_train) + loss_noise
                loss.backward()
                optimizer.step()

                if (batch_idx+1) % args.update_period ==0:
                    for _ in range(args.num_update):
                        mask_update_data(model, task, inputs, targets, n_train, args.lamb, args.N_max, args.death_method, args.birth_method)

            if epoch+1 <= burn_in_epoch:
                scheduler.step()

            model.eval()            
            with torch.no_grad():
                _,ERROR_train = log_likelihood_dataloader(model, task, trainloader)
                _,ERROR_test = log_likelihood_dataloader(model, task, testloader)

                ERROR_train, ERROR_test = ERROR_train.item()/n_train, ERROR_test.item()/n_test

                total_node_sparsity = []
                l=0
                for n, p in model.named_parameters():
                    if 'active' in n:
                        total_node_sparsity.append(torch.sum(p.data).item())
                        l+=1
                total_node_sparsity_ratio = np.sum(np.stack(total_node_sparsity))/np.sum(np.stack(initial_nodes)) 

                bar.suffix  = '({epo}/{total_epo}) ERR_train:{ER_tr} | ERR_test:{ER_te} | {ns}'.format(
                                epo=epoch + 1,
                                total_epo=total_epoch,
                                ER_tr=np.round(ERROR_train,3),
                                ER_te=np.round(ERROR_test,3),
                                ns = np.round(total_node_sparsity_ratio,3)
                                )                
        
            res.loc[epoch,'total_ns'] = total_node_sparsity_ratio
            res.loc[epoch,'Er_train'] = np.round(ERROR_train,3)
            res.loc[epoch,'Er_test'] = np.round(ERROR_test,3)
            
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
        macs_list=[]
        params_list=[]
        
        with torch.no_grad():
            for mt in range(args.num_sample):
                model = M_ResNet18(num_classes, args.prior, args.prior_scale).cuda()
                model.eval()
                
                n_h_nodes = [] 
                for name, param in model.named_parameters():
                    if 'active' in name:
                        n_h_nodes.append(int(param.sum().item()))
                
                model.load_state_dict(torch.load(out_directory + '/seed_%d_mt_%d.pt'%(seed,mt), map_location='cuda:'+str(args.gpu)))
                pred = []
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    pred.append(F.softmax(outputs,dim=1))              

                    if mt==0:
                        target_list.append(targets.squeeze())

                pred_list.append(torch.cat(pred,0))                            
                                       
                n_act_h_nodes = []
                for name, param in model.named_parameters():
                    if 'active' in name:
                        n_act_h_nodes.append(int(param.sum().item()))      
                n_act_i_nodes = p_data

                macs, params = profiling(args.model, p_data, n_act_i_nodes, num_classes, n_act_h_nodes, n_h_nodes=n_h_nodes)
                macs_list.append(macs)
                params_list.append(params)                 
                
            pred_list = torch.stack(pred_list)
            target_list = torch.cat(target_list,0)           
            ACC, m_NLL, ECE = evaluate_averaged_model_classification(pred_list, target_list)
            
            
            macs = np.stack(macs_list).mean()
            params = np.stack(params_list).mean()
            print("ACC : ", ACC, " m_NLL : ", m_NLL, " ECE : ", ECE, "FLOPs rate : ", 100 * (macs), " non-zero param rate : ", 100 * (params))
            result1_list.append(ACC)
            result2_list.append(m_NLL)
            result3_list.append(ECE)
            result4_list.append(100 * (macs))
            result5_list.append(100 * (params))
    
    num_seed = args.end_seed - args.start_seed
    result1_list, result2_list, result3_list, result4_list, result5_list = np.stack(result1_list), np.stack(result2_list), np.stack(result3_list), np.stack(result4_list), np.stack(result5_list)
    print("%.3f(%.3f), %.3f(%.3f), %.3f(%.3f), %.2f(%.2f), %.2f(%.2f)" % (np.mean(result1_list), np.std(result1_list)/np.sqrt(num_seed),np.mean(result2_list), np.std(result2_list)/np.sqrt(num_seed),np.mean(result3_list), np.std(result3_list)/np.sqrt(num_seed),np.mean(result4_list), np.std(result4_list)/np.sqrt(num_seed),np.mean(result5_list), np.std(result5_list)/np.sqrt(num_seed)))        
            
                    
if __name__ == '__main__':
    main()