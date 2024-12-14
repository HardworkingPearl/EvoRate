
import numpy as np
import torch
import torch.nn as nn
import math
Sequence = True

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    if Sequence:
        # TODO: shift with 2, mi->20

        T = 10
        eps = torch.randn((T * batch_size,dim)).view(batch_size, T,  dim)  
        x = torch.empty((batch_size,T-1,dim)).float()
        for i in range(T-1):
            x[:, i] = eps[:, i]-0.5  

        y = math.sqrt(1 - rho**2) * eps[:, -1] + rho * torch.sum(x,dim=1)  / math.sqrt(T-1) +1

    else:    
        x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
        y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

    if cubic is not None:
        y = y ** 3

    return x, y


def rho_to_mi(dim, rho):
    """Obtain the ground truth mutual information from rho."""
    return -0.5 * np.log(1 - rho**2) * dim


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian give ground truth mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


def mi_schedule(n_iter):
    """Generate schedule for increasing correlation over time."""
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 20  
    return mis.astype(np.float32)


def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)
        if Sequence:
            self.seq_proj = nn.LSTM(input_size=dim, hidden_size=dim,num_layers=layers,batch_first=True)

    def forward(self, x, y):
        output, _ = self.seq_proj(x)
        yhat = output[:,-1,:]  
        scores = torch.matmul(self._h(y), self._g(yhat).t())
        return scores


class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        if Sequence:
            self.seq_proj = nn.LSTM(input_size=dim, hidden_size=dim,num_layers=layers,batch_first=True)

    def forward(self, x, y):
        if Sequence:
            output, _ = self.seq_proj(x)
            yhat = output[:,-1,:]  
        scores = -torch.cdist(yhat, y)**2
        batch_size = x.size(0)
        return scores

               
def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)
