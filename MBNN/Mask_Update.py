import copy
import numpy as np
import torch
from itertools import permutations

from MCMC import *

def mask_update_dataloader(model, task, dataloader, datasize, lam=0.1, N_max=3, death_method="proposed", birth_method="random"):
    model.zero_grad()
    for n, p in model.named_parameters():
        if 'active' in n:
            p.requires_grad = True

    log_prob = log_prob_dataloader(model, task, dataloader)
    log_prob.backward()  
    
    u = torch.bernoulli(torch.tensor(0.5))
    N = np.random.randint(1,N_max+1)
    if u==1:
        proposal_prob = proposal_probability(model, u, method=death_method)
    else:
        proposal_prob = proposal_probability(model, u, method=birth_method)
    ind = torch.multinomial(proposal_prob, N, replacement=False)   
    
    active_vectors, active_vectors_size, active_vectors_item, _, current_p = active_info(model)
    L = len(active_vectors_size)
    ind_L = [(np.cumsum(active_vectors_size)<=i).sum() for i in ind.cpu().numpy()]
    ind_p = ind.cpu().numpy() - np.insert(np.cumsum(active_vectors_size),0,0)[ind_L]
    nums = np.bincount(ind_L, minlength=L)
    
    prior_rate = 1
    for l in range(L):
        if nums[l]>0:
            prior_rate *= torch.exp(-(lam*np.log(datasize))**5 *((2-4*u)*current_p[l]*nums[l] + nums[l]**2))
            if u==0:
                prior_rate *= torch.prod(torch.arange(current_p[l]+1,current_p[l]+nums[l]+1)) / torch.prod(torch.arange(active_vectors_size[l]-current_p[l]-nums[l]+1,active_vectors_size[l]-current_p[l]+1))
            else:
                prior_rate *= torch.prod(torch.arange(active_vectors_size[l]-current_p[l]+1,active_vectors_size[l]-current_p[l]+nums[l]+1)) / torch.prod(torch.arange(current_p[l]-nums[l]+1,current_p[l]+1))
              
                
    model_tmp = copy.deepcopy(model)
    active_vectors_tmp = []
    for n, p in model_tmp.named_parameters():
        if 'active' in n:
            active_vectors_tmp.append(p)

    for k in range(len(ind_L)):
        active_vectors_tmp[ind_L[k]][ind_p[k]].data += (1-2*u)
        
    log_prob_tmp = log_prob_dataloader(model_tmp, task, dataloader)
    log_prob_tmp.backward()
    
    if (1-u)==1:
        proposal_prob_tmp = proposal_probability(model_tmp, 1-u, method=death_method)
    else:
        proposal_prob_tmp = proposal_probability(model_tmp, 1-u, method=birth_method) 
                
    accept_prob = torch.clamp(prior_rate*torch.exp(log_prob_tmp-log_prob)*prob_multi_wor(proposal_prob_tmp, ind)/prob_multi_wor(proposal_prob, ind), max=1)
 
    if torch.rand(1).cuda()<accept_prob:
        for k in range(len(ind_L)):
            if active_vectors[ind_L[k]].sum()>2*N_max:
                active_vectors[ind_L[k]][ind_p[k]].data += (1-2*u)    

    for n, p in model.named_parameters():
        if 'active' in n:
            p.requires_grad = False

            
def mask_update_data(model, task, inputs, targets, datasize, lam=0.1, N_max=3, death_method="proposed", birth_method="random"):
    model.zero_grad()
    for n, p in model.named_parameters():
        if 'active' in n:
            p.requires_grad = True

    log_prob = log_prob_data(model, task, inputs, targets, datasize)
    log_prob.backward()   
    
    u = torch.bernoulli(torch.tensor(0.5))
    N = np.random.randint(1,N_max+1)
    if u==1:
        proposal_prob = proposal_probability(model, u, method=death_method)
    else:
        proposal_prob = proposal_probability(model, u, method=birth_method)
    ind = torch.multinomial(proposal_prob, N, replacement=False)       
    
    active_vectors, active_vectors_size, active_vectors_item, _, current_p = active_info(model)
    L = len(active_vectors_size)
    ind_L = [(np.cumsum(active_vectors_size)<=i).sum() for i in ind.cpu().numpy()]
    ind_p = ind.cpu().numpy() - np.insert(np.cumsum(active_vectors_size),0,0)[ind_L]
    nums = np.bincount(ind_L, minlength=L)
    
    prior_rate = 1
    for l in range(L):
        if nums[l]>0:
            prior_rate *= torch.exp(-(lam*np.log(datasize))**5 *((2-4*u)*current_p[l]*nums[l] + nums[l]**2))
            if u==0:
                prior_rate *= torch.prod(torch.arange(current_p[l]+1,current_p[l]+nums[l]+1)) / torch.prod(torch.arange(active_vectors_size[l]-current_p[l]-nums[l]+1,active_vectors_size[l]-current_p[l]+1))
            else:
                prior_rate *= torch.prod(torch.arange(active_vectors_size[l]-current_p[l]+1,active_vectors_size[l]-current_p[l]+nums[l]+1)) / torch.prod(torch.arange(current_p[l]-nums[l]+1,current_p[l]+1))
              
                
    model_tmp = copy.deepcopy(model)
    active_vectors_tmp = []
    for n, p in model_tmp.named_parameters():
        if 'active' in n:
            active_vectors_tmp.append(p)

    for k in range(len(ind_L)):
        active_vectors_tmp[ind_L[k]][ind_p[k]].data += (1-2*u)
        
    log_prob_tmp = log_prob_data(model_tmp, task, inputs, targets, datasize)
    log_prob_tmp.backward()
    
    if (1-u)==1:
        proposal_prob_tmp = proposal_probability(model_tmp, 1-u, method=death_method)
    else:
        proposal_prob_tmp = proposal_probability(model_tmp, 1-u, method=birth_method)   
                
    accept_prob = torch.clamp(prior_rate*torch.exp(log_prob_tmp-log_prob)*prob_multi_wor(proposal_prob_tmp, ind)/prob_multi_wor(proposal_prob, ind), max=1)
 
    if torch.rand(1).cuda()<accept_prob:
        #print("jump")
        for k in range(len(ind_L)):
            active_vectors[ind_L[k]][ind_p[k]].data += (1-2*u)    

    for n, p in model.named_parameters():
        if 'active' in n:
            p.requires_grad = False
            
    return accept_prob

            
def active_info(model):
    active_vectors = []
    active_vectors_size = []
    active_vectors_item = []
    active_vectors_grad = []
    current_p = [] 
    for n, p in model.named_parameters():
        if 'active' in n:
            active_vectors.append(p)
            active_vectors_size.append(len(p))
            active_vectors_item.append(p.data)
            active_vectors_grad.append(p.grad)
            current_p.append(torch.sum(p).data)
    active_vectors_item = torch.hstack(active_vectors_item)
    active_vectors_grad = torch.hstack(active_vectors_grad)
    
    return(active_vectors, active_vectors_size, active_vectors_item, active_vectors_grad, current_p)

            
def prob_multi_wor(proposal_prob, ind):
    sum=0
    for ind_permute in permutations(ind):
        subtract = 0
        prod = 1
        proposal_prob_tmp = copy.deepcopy(proposal_prob)
        for k in range(len(ind_permute)):
            prod *= proposal_prob_tmp[ind_permute[k]]/torch.sum(proposal_prob_tmp)
            proposal_prob_tmp[ind_permute[k]] *= 0
        sum += prod
        
    return sum

def proposal_probability(model, u, method="random"):
    
    _, _, active_vectors_item, active_vectors_grad, _ = active_info(model)
    
    if method=='Oops':
        proposal_prob = torch.where(active_vectors_item==u, torch.clamp(torch.exp(-(2*active_vectors_item-1)*active_vectors_grad/2), max=1e35), torch.tensor([0.]).cuda())
        
    elif method=='proposed':
        proposal_prob = torch.where(active_vectors_item==u, torch.exp(-torch.abs(active_vectors_grad)/2), torch.tensor([0.]).cuda())
        
    elif method=='random':
        proposal_prob = torch.where(active_vectors_item==u, torch.tensor([1.]).cuda(), torch.tensor([0.]).cuda())
        
    proposal_prob = proposal_prob / torch.sum(proposal_prob)
    return proposal_prob
        