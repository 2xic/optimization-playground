import torch
import torch.nn as nn
import torch.nn.functional as F
"""
https://fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch/
https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
https://scikit-learn.org/1.5/modules/neural_networks_unsupervised.html
"""
class RBM(nn.Module):
    def __init__(self, nv, nh):
        super().__init__()
        # Weights of the model 
        self.W = nn.Parameter(torch.randn(nh, nv))
        self.a = nn.Parameter(torch.randn(1, nh))
        self.b = nn.Parameter(torch.randn(1, nv))

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def forward(self, v):
        # generative model
        _, h1 = self.sample_h(v)
        h_ = h1
        for _ in range(10):
            _,v_ = self.sample_v(h_)
            _,h_ = self.sample_h(v_)
        return v, v_

    def classify(self, v):
        # classify
        _, h1 = self.sample_h(v)
        return v, h1
        
    def free_energy(self,v):
        vbias_term = v.mv(self.b)
        wx_b = F.linear(v,self.W, self.a)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()
    
