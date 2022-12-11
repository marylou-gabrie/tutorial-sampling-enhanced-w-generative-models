'''
Simplified implementation of Real-NVPs borrowing from
https://github.com/chrischute/real-nvp.
Original paper:
Density estimation using Real NVP
Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio
arXiv:1605.08803
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions.multivariate_normal import MultivariateNormal


class MLP(nn.Module):
    """
    a MLP class inheriting from the parent class nn.Module. 
    nn.Module is the generic parent class of models in Pytorch.
    It requires a method called forward. 
    With , Pytorch will be able to recursively recover all parameters in the attributes of a nn.Module object provide the attributes have type nn.Modules or nn.ModuleList.
    """
    def __init__(self, layerdims, activation=torch.relu, init_scale=1e-3):
        super(MLP, self).__init__()

        self.layerdims = layerdims
        self.activation = activation

        linears = [
            nn.Linear(layerdims[i], layerdims[i + 1]) for i in range(len(layerdims) - 1)
        ]

        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(
                    layer.weight, std=init_scale / np.sqrt(layerdims[l])
                )
                torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y


class AffineCoupling(nn.Module):
    """ Affine Coupling layer 
    Implements coupling layers with a rescaling 
    Args:
        s (nn.Module): scale network
        t (nn.Module): translation network
        mask (binary tensor): binary array of same size as inputs
        dt (float): rescaling factor for s and t
    """

    def __init__(self, s=None, t=None, mask=None, dt=1):
        super(AffineCoupling, self).__init__()

        self.mask = mask
        self.scale_net = s
        self.trans_net = t
        self.dt = dt

    def forward(self, x, log_det_jac=None, inverse=False):
        if log_det_jac is None:
            log_det_jac = 0

        s = self.mask * self.scale_net(x * (1 - self.mask))
        s = torch.tanh(s) * self.dt
        t = self.mask * self.trans_net(x * (1 - self.mask)) * self.dt

        if inverse:
            log_det_jac -= s.view(s.size(0), -1).sum(-1)
            x = x * torch.exp(-s) - t

        else:
            log_det_jac += s.view(s.size(0), -1).sum(-1)
            x = (x + t) * torch.exp(s)

        return x, log_det_jac


class NormalizingFlow(nn.Module):
    """ Minimal Real NVP architecture
    Args:
        dims (int,): input dimension
        n_blocks (int): number of pairs of coupling layers
        hidden_dim (int): # of hidden neurones per layer (coupling MLPs)
    """

    def __init__(self, dim, n_blocks, 
                 hidden_dim=124,
                 hidden_activation=torch.relu,
                 device='cpu'):
        super(NormalizingFlow, self).__init__()

        self.device = device
        self.dim = dim
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        mask = torch.ones(dim, device=self.device)
        mask[:int(dim / 2)] = 0
        self.mask = mask.view(1, dim)

        self.coupling_layers = self.initialize()

        self.prior_prec =  torch.eye(dim).to(device)
        self.prior_log_det = 0
        self.prior_distrib = MultivariateNormal(
            torch.zeros((dim,), device=self.device), self.prior_prec)

    def build_coupling_block(self, layer_dims=None, nets=None, reverse=False):
        count = 0
        coupling_layers = []
        for count in range(2):
            s = MLP(layer_dims, init_scale=1e-3)
            s = s.to(self.device)
            t = MLP(layer_dims, init_scale=1e-3)
            t = t.to(self.device)

            if count % 2 == 0:
                mask = 1 - self.mask
            else:
                mask = self.mask
            
            dt = self.n_blocks * 2
            dt = 2 / dt
            coupling_layers.append(AffineCoupling(s, t, mask, dt=dt))

        return coupling_layers

    def initialize(self):
        dim = self.dim
        coupling_layers = []

        for block in range(self.n_blocks):
            layer_dims = [self.hidden_dim]
            layer_dims = [dim] + layer_dims + [dim]

            couplings = self.build_coupling_block(layer_dims)
            coupling_layers.append(nn.ModuleList(couplings))

        return nn.ModuleList(coupling_layers)

    def forward(self, x, return_ldj=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[block]
            for coupling_layer in couplings:
                x, log_det_jac = coupling_layer(x, log_det_jac)

        if return_ldj:
            return x, log_det_jac
        else:
            return x

    def backward(self, x, return_ldj=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)
        
        for block in range(self.n_blocks):
            couplings = self.coupling_layers[::-1][block]
            for coupling_layer in couplings[::-1]:
                x, log_det_jac = coupling_layer(
                    x, log_det_jac, inverse=True)

        if return_ldj:
            return x, log_det_jac
        else:
            return x

    def log_prob(self, x):
        z, log_det_jac = self.backward(x, return_ldj=True)
        prior_ll = - 0.5 * torch.einsum('ki,ij,kj->k', z, self.prior_prec, z)
        prior_ll -= 0.5 * (self.dim * np.log(2 * np.pi) + self.prior_log_det)

        ll = prior_ll + log_det_jac
        return ll
    
    def U(self, x):
        return - self.log_prob(x)

    def sample(self, n):
        z = self.prior_distrib.rsample(torch.Size([n, ])).to(self.device)

        return self.forward(z)


class MoG():
    def __init__(self, means, covars, weights=None,
                 dtype=torch.float32, device='cpu'):
        """
        Class to handle operations around mixtures of multivariate
        Gaussian distributions
        Args:
            means: list of 1d tensors of centroids
            covars: list of 2d tensors of covariances
            weights: list of relative statistical weights (does not need to sum to 1)
        """
        self.device = device
        self.beta = 1.  # model 'temperature' for sampling with langevin and mh
        self.means = means
        self.covars = covars
        self.dim = means[0].shape[0]
        self.k = len(means)  # number of components in the mixture

        if weights is not None:
            self.weights = torch.tensor(weights, dtype=dtype, device=device)
        else:
            self.weights = torch.tensor([1 / self.k] * self.k,
                                        dtype=dtype, device=device)

        self.cs_distrib = td.categorical.Categorical(probs=self.weights)
        self.normal_distribs = []
        for c in range(self.k):
            c_distrib = td.multivariate_normal.MultivariateNormal(
                self.means[c].to(device),
                covariance_matrix=self.covars[c].to(device)
                )
            self.normal_distribs.append(c_distrib)

        self.covars_inv = torch.stack([torch.inverse(cv) for cv in covars])
        self.dets = torch.stack([torch.det(cv) for cv in covars])

    def sample(self, n):
        cs = self.cs_distrib.sample_n(n).to(self.device)

        samples = torch.zeros((n, self.dim), device=self.device)
        for c in range(self.k):
            n_c = (cs == c).sum()
            samples[cs == c, :] = self.normal_distribs[c].sample_n(n_c)
        return samples.to(self.device)

    def log_prob(self, x):
        x = x.unsqueeze(1)
        m = torch.stack(self.means).unsqueeze(0)
        args = - 0.5 * torch.einsum('kci,cij,kcj->kc', x-m, self.covars_inv, x-m)
        args += torch.log(self.weights)
        args -= torch.log((self.weights.sum() * torch.sqrt((2 * np.pi) ** self.dim * self.dets)))
        return  torch.logsumexp(args, 1)
    
    def U(self, x):
        return -self.log_prob(x)

