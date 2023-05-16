import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class M_relu(nn.Module):    
    def __init__(self, input_dim, init_active_dim):
            super().__init__()
            self.input_dim = input_dim
            self.init_active_dim = init_active_dim

            self.active = nn.Parameter(torch.cuda.FloatTensor([1]*self.init_active_dim + 
                                                              [0]*(self.input_dim-self.init_active_dim)), requires_grad=False)        

    def forward(self, x):
        if len(x.shape)==2:            
            M = self.active.view(1,-1)       
            return M * F.relu(x)
        elif len(x.shape)==4:            
            M = self.active.view(1,-1,1,1)       
            return M * F.relu(x)


class M_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, L, init_p, prior, prior_scale):
        super(M_MLP, self).__init__()
                
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.L = L
        self.p = np.floor(1.1*init_p).astype(int) # For additional space
        self.p_vec = np.hstack([input_dim,np.repeat(self.p,L),output_dim])
        self.init_active_dim = init_p
        self.layers = self._make_layer()
        self.register_buffer('sigma', torch.tensor([1.0]).cuda())
        
        if prior == 'Cauchy':
            self.prior = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]).cuda(), torch.tensor([prior_scale]).cuda())
        elif prior == 'Normal':
            self.prior = torch.distributions.normal.Normal(torch.tensor([0.0]).cuda(), torch.tensor([prior_scale]).cuda())
        self.prior_scale = prior_scale
          
    def _make_layer(self):
        layers = []
        for l in range(self.L):       
            layer = []
            layer.append(nn.Linear(self.p_vec[l], self.p_vec[l+1]))
            layer.append(M_relu(self.p_vec[l+1], self.init_active_dim))
            layers.append(nn.Sequential(*layer))
            
        layer = []        
        layer.append(nn.Linear(self.p, self.output_dim))
        layers.append(nn.Sequential(*layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):        
        x = x.view(-1, self.input_dim)        
        x = self.layers(x)       
        return x
    

class Masked_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Masked_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.M_relu1 = M_relu(planes,int(planes*4/5))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.M_relu2 = M_relu(planes,int(planes*4/5))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.M_relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.M_relu2(out)
        return out
    
class Masked_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, prior, prior_scale):
        super(Masked_ResNet, self).__init__()
        self.in_planes = int(64*(5/4))

        self.conv1 = nn.Conv2d(3, int(64*(5/4)), kernel_size=3, stride=1, padding=1, bias=False) # For additional space
        self.bn1 = nn.BatchNorm2d(int(64*(5/4)))
        self.M_relu = M_relu(int(64*(5/4)),64)
        self.layer1 = self._make_layer(block, int(64*(5/4)), num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, int(128*(5/4)), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*(5/4)), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*(5/4)), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(512*(5/4))*block.expansion, num_classes)
        
        if prior == 'Cauchy':
            self.prior = torch.distributions.cauchy.Cauchy(torch.tensor([0.0]).cuda(), torch.tensor([prior_scale]).cuda())
        elif prior == 'Normal':
            self.prior = torch.distributions.normal.Normal(torch.tensor([0.0]).cuda(), torch.tensor([prior_scale]).cuda())
        self.prior_scale = prior_scale 
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.M_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out    
    
def M_ResNet18(num_classes, prior, prior_scale):
    return Masked_ResNet(Masked_BasicBlock, [2,2,2,2], num_classes=num_classes, prior=prior, prior_scale=prior_scale) 