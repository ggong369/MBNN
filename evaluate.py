import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import calibration as cal
from scipy.stats import norm
from thop import profile


def ll_mixture_normal(output, target, sigma):        
    exponent = -((target - output)**2).T/(2 * sigma**2)
    log_coeff = -0.5*torch.log(2*torch.tensor(np.pi))-torch.log(sigma)    
    px = torch.mean(torch.exp(exponent + log_coeff),1)
    ll = torch.where(px!=0, torch.log(px), torch.mean(exponent + log_coeff,1))    
    return torch.sum(ll)

def A(mu, sigma2):
    sigma = torch.sqrt(sigma2)
    r = (mu/sigma).detach().cpu().numpy()    
    A1 = 2*sigma*(torch.from_numpy(norm.pdf(r)).float().cuda())
    A2 = mu*(torch.from_numpy(2*norm.cdf(r)-1).float().cuda())    
    return(A1 + A2)

def CRPS_mixnorm(w,mu,sigma2,x):
    M = len(w)
    if (len(mu)!=M or len(sigma2)!=M): return(None)
    if x.dim()>0 :
        if len(x)>1:
            return(None)
    w = w/torch.sum(w)     
    crps1 = torch.sum(w*A(x-mu, sigma2))    
    crps3=[]
    for m in range(M):
        crps3.append(torch.sum(w*A(mu[m]-mu,sigma2[m] + sigma2)))    
    crps3 = torch.stack(crps3)
    crps2 = torch.sum(crps3*w/2)    
    return crps1 - crps2

def CRPS_norm(mu,sigma2,x):    
    if x.dim()>0 :
        if len(x)>1:
            return(None)
    crps1 = A(x-mu, sigma2)    
    crps2 = 0.5*A(0,2*sigma2)    
    return crps1 - crps2

def evaluate_averaged_model_regression(pred_list, target_list, sigma_list): 
    CRPS_list=[]
    for i in range(len(target_list)):
        CRPS = CRPS_mixnorm(torch.ones(pred_list.shape[0]).cuda(),pred_list[:,i], sigma_list**2, target_list[i])
        CRPS_list.append(CRPS)
    CRPSs = torch.stack(CRPS_list)    
    RMSE = torch.sqrt(((torch.mean(pred_list,0) - target_list)**2).mean()).item()
    m_NLL = -ll_mixture_normal(pred_list, target_list, sigma_list).item() / pred_list.shape[1]
    CRPS = torch.mean(CRPSs).item()    
    return(RMSE, m_NLL, CRPS)

def evaluate_averaged_model_classification(pred_list, target_list):    
    target_list = target_list.long()
    outputs_mixture = torch.mean(pred_list, dim=0)    
    ACC= torch.mean((torch.argmax(outputs_mixture,1) == target_list).float()).item()    
    criterion = torch.nn.NLLLoss(reduction='mean')
    m_NLL = criterion(torch.log(outputs_mixture), target_list).item()    
    ECE = cal.get_calibration_error(outputs_mixture.detach().cpu().numpy(), target_list.detach().cpu().numpy())    
    return(ACC, m_NLL, ECE)
   
class MLP_customized(nn.Module):
    def __init__(self, input_dim, output_dim, h_vec):
        super(MLP_customized, self).__init__()
                
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.L = len(h_vec)
        self.p_vec = np.hstack([input_dim,h_vec,output_dim])
        self.layers = self._make_layer()  
            
    def _make_layer(self):
        layers = []
        for l in range(self.L):       
            layer = []
            layer.append(nn.Linear(self.p_vec[l], self.p_vec[l+1]))
            layer.append(nn.ReLU())
            layers.append(nn.Sequential(*layer))
            
        layer = []        
        layer.append(nn.Linear(self.p_vec[-2], self.output_dim))
        layers.append(nn.Sequential(*layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):        
        x = x.view(-1, self.input_dim)        
        x = self.layers(x)       
        return x
    
class Convert(nn.Module):
    def __init__(self, size):
        super(Convert, self).__init__()
        self.size = size

    def forward(self, x):
        s = x.shape
        if s[1]>self.size:
            return torch.split(x, self.size, 1)[0]
        elif s[1]<self.size:
            return torch.cat((x, torch.zeros(s[0],self.size-s[1],s[2],s[3]).cuda()), 1)
    
class BasicBlock_customized(nn.Module):
    expansion = 1

    def __init__(self, h, stride, empty_shortcut):
        super(BasicBlock_customized, self).__init__()
        self.conv1 = nn.Conv2d(h[0], h[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(h[1])
        self.conv2 = nn.Conv2d(h[1], h[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(h[2])

        self.shortcut = nn.Sequential()
        
        if not empty_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(h[0], h[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(h[2])
            )        
        elif h[0]!=h[2]:
            self.shortcut = nn.Sequential(
                Convert(h[2])
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out    
    
class ResNet18_customized(nn.Module):
    def __init__(self, num_classes, h_vec):
        super(ResNet18_customized, self).__init__()

        self.conv1 = nn.Conv2d(3, h_vec[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(h_vec[0])
        self.layer1 = self._make_layer(BasicBlock_customized, h_vec[0:5], stride=1)
        self.layer2 = self._make_layer(BasicBlock_customized, h_vec[4:9], stride=2)
        self.layer3 = self._make_layer(BasicBlock_customized, h_vec[8:13], stride=2)
        self.layer4 = self._make_layer(BasicBlock_customized, h_vec[12:17], stride=2)
        self.linear = nn.Linear(h_vec[16], num_classes)
        self.register_buffer('sigma', torch.tensor([1.0]).cuda())
        
    def _make_layer(self, block, h, stride):
        layers = []
        layers.append(block(h[0:3], stride, (stride==1)))
        layers.append(block(h[2:5], 1, True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def profiling(model_type, p_data, n_act_i_nodes, num_classes, n_act_h_nodes, MLP_L=None, MLP_p=None, n_h_nodes=None):    
    if model_type == "MLP":        
        full_model = MLP_customized(p_data, num_classes, [MLP_p]*MLP_L).cuda()
        macs_f, params_f = profile(full_model, torch.randn(1,p_data).cuda(), verbose=False)        
        compressed_model = MLP_customized(n_act_i_nodes, num_classes, n_act_h_nodes).cuda()
        macs_c, params_c = profile(compressed_model, torch.randn(1,n_act_i_nodes).cuda(), verbose=False)
        return macs_c/macs_f, params_c/params_f    
    elif model_type == "resnet18":
        full_model = ResNet18_customized(num_classes, n_h_nodes).cuda()
        macs_f, params_f = profile(full_model, (torch.randn(1,3,32,32).cuda(),), verbose=False)
        compressed_model = ResNet18_customized(num_classes, n_act_h_nodes).cuda()
        macs_c, params_c = profile(compressed_model, (torch.randn(1,3,32,32).cuda(),), verbose=False)
        return macs_c/macs_f, params_c/params_f