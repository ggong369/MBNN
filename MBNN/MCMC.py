import numpy as np
import torch
import copy


def ll_regression(output, target, sigma):
    exponent = -((target - output)**2).sum() / (2 * sigma**2)
    log_coeff = (-0.5*torch.log(2*torch.tensor(np.pi))-torch.log(sigma))*output.shape[0]
    return (log_coeff + exponent)

def log_prior(model):
    
    l_prior = torch.zeros(1, requires_grad=True).cuda()
    
    for n, p in model.named_parameters():
        if (not 'bn' in n) and (not 'active' in n) and (not 'sigma' in n) :
            l_prior = l_prior + model.prior.log_prob(p).sum()
            
    return l_prior


def log_likelihood_dataloader(model, task, dataloader):
    
    l_likelihood = torch.zeros(1, requires_grad=True).cuda()
    ERROR = torch.zeros(1, requires_grad=True).cuda()
    
    if task is 'regression':        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            l_likelihood = l_likelihood + ll_regression(outputs, targets, model.sigma)
            ERROR += ((targets - outputs)**2).sum()
                
    elif task is 'classification':        
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda().squeeze().long()
            outputs = model(inputs)
            l_likelihood = l_likelihood - criterion(outputs, targets)
            ERROR += (torch.argmax(outputs,1) != targets).sum()            
    return l_likelihood, ERROR


def log_prob_dataloader(model, task, dataloader):
    
    return log_prior(model) + log_likelihood_dataloader(model, task, dataloader)[0]



def log_likelihood_data(model, task, inputs, targets):    
    if task is 'regression':
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        l_likelihood = ll_regression(outputs, targets, model.sigma)
        ERROR = ((targets - outputs)**2).sum()
        
    elif task is 'classification':
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')        
        inputs, targets = inputs.cuda(), targets.cuda().squeeze().long()
        outputs = model(inputs)
        l_likelihood = - criterion(outputs, targets)
        ERROR = (torch.argmax(outputs,1) != targets).sum()
        
    return l_likelihood, ERROR
        
def log_prob_data(model, task, inputs, targets, datasize):
    return log_prior(model) + log_likelihood_data(model, task, inputs, targets)[0]*(datasize/targets.shape[0])


def HMC(model, task, dataloader, lr, lf_step):    
        model_tmp = copy.deepcopy(model).cuda()            
        for n, p in model_tmp.named_parameters():
            if p.requires_grad:
                if not hasattr(p, 'momentum'):
                    setattr(p, 'momentum', torch.zeros_like(p.data))                    
                p.momentum = torch.randn(p.size()).cuda()
                
        log_prob = log_prob_dataloader(model_tmp, task, dataloader)
        ham = -log_prob.data
        for n, p in model_tmp.named_parameters():
            if p.requires_grad:
                ham += (p.momentum * p.momentum).sum()/2                   
        
        model_tmp.zero_grad()
        log_prob.backward()        
        
        for n, p in model_tmp.named_parameters():
            if p.requires_grad:      
                p.momentum += lr*p.grad.data/2
                p.data = p.data + lr * p.momentum
                
        for step in range(lf_step-1):          
            model_tmp.zero_grad()
            log_prob = log_prob_dataloader(model_tmp, task, dataloader)
            log_prob.backward()
            
            for n, p in model_tmp.named_parameters():
                if p.requires_grad:
                    p.momentum += lr*p.grad.data
                    p.data = p.data + lr * p.momentum

        model_tmp.zero_grad()
        log_prob = log_prob_dataloader(model_tmp, task, dataloader)
        log_prob.backward()                    

        for n, p in model_tmp.named_parameters():
            if p.requires_grad:
                p.momentum += lr*p.grad.data/2
                
        ham_tmp = -log_prob.data
        for n, p in model_tmp.named_parameters():
            if p.requires_grad:
                ham_tmp += (p.momentum * p.momentum).sum()/2
                
        if np.isnan((-ham_tmp + ham).item()):
            return 0
        
        log_p_accept = min(0., float(-ham_tmp + ham))
        if log_p_accept >= torch.log(torch.rand(1)):
            model.load_state_dict(copy.deepcopy(model_tmp.state_dict()))
            
        return np.exp(log_p_accept)
    
    
class DualAveragingStepSize:
    def __init__(self, initial_step_size, target_accept=0.7, gamma=0.2, t0=10.0, kappa=0.75):
        self.mu = np.log(initial_step_size)  
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        self.t += 1
        return np.exp(log_step), np.exp(self.log_averaged_step)    
    

    
def sigma_update(model, dataloader, n, std=1.0):
    
    ERROR = torch.zeros(1, requires_grad=True).cuda()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        ERROR = ERROR + ((targets - outputs)**2).sum()

    gamma_sampler = torch.distributions.gamma.Gamma(torch.tensor([1.0 + n/2]).cuda(), ERROR/2 + std)
    model.sigma = torch.sqrt(1/gamma_sampler.sample())
    
    
def noise_loss(model, lr):
    noise_loss = 0.0
    noise_std = (2/lr)**0.5
    for var in model.parameters():
        means = torch.cuda.FloatTensor(var.size()).fill_(0)
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).cuda())
    return noise_loss