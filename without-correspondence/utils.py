import numpy as np
import torch
import torch.nn as nn
import math
Sequence = True


def generate_invertible_matrix(n):
    while True:
        # Generate a random matrix
        matrix = torch.randn(n, n)
        # Check if the diagonal elements of R are non-zero
        if torch.linalg.matrix_rank(matrix)==n:
            return matrix

def generate_rotation_matrix(dim, angle):
    # Initialize identity matrix
    rotation_matrix = torch.eye(dim)

    # Iterate over pairs of dimensions
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            # Construct 2D rotation matrix for the pair of dimensions (i, j)
            givens_matrix = torch.eye(dim)
            givens_matrix[i, i] = torch.cos(angle)
            givens_matrix[j, j] = torch.cos(angle)
            givens_matrix[i, j] = -torch.sin(angle)
            givens_matrix[j, i] = torch.sin(angle)

            # Update rotation matrix by multiplying with the constructed 2D rotation matrix
            rotation_matrix = torch.matmul(rotation_matrix, givens_matrix)

    return rotation_matrix


# Example usage
angle = torch.tensor(30.0)  # Rotation angle in degrees

# Convert angle to radians
angle_rad = torch.deg2rad(angle)

# Generate rotation matrix
rotation_matrix = generate_rotation_matrix(128, angle_rad)

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None):
    """Generate samples from a correlated Gaussian distribution."""
    if Sequence:
        # TODO: shift with 2, mi->20

        T = 2
        relation = 1
        eps = torch.randn((T * batch_size,dim)).view(batch_size, T,  dim)  
        x = torch.empty((batch_size,T-1,dim)).float()
        for i in range(T-1):
            x[:, i] = eps[:, i]-0.5  # + self.rho * torch.sum(Xs[:, :i],dim=1) / math.sqrt(i)  # mean cause the variance shift!

        y = math.sqrt(1 - rho**2) * eps[:, -1] + relation *  rho * torch.sum(x,dim=1)  / math.sqrt(T-1) @ rotation_matrix +1

    else:    
        x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
        y = rho * x + relation * torch.sqrt(torch.tensor(1. - rho**2).float()) * eps

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

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden, heads=4):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, mask=None):
        attention, _ = self.multihead_attention(x, x, x)
        feedforward = self.feedforward(x)
        x = x + attention +feedforward
        x = self.layer_norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, hidden, num_layers, heads=8):
        super(Transformer, self).__init__()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim, hidden, heads) for _ in range(num_layers-1)])
        self.transformer_output = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, context):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x, _ = self.transformer_output(query=context, key=x, value=x)
        x = x[:,0]

        # x = self.fc(x)  # Only use the first token's output for classification
        return x

class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)
        if Sequence:
            self.seq_proj = nn.LSTM(input_size=dim, hidden_size=dim,num_layers=layers,batch_first=True)  
        self.enc = nn.Linear(dim, dim)

    def forward(self, x, y):
        #############
        len_x = x.shape[1]
        x = self.enc(x)
        x0 = x[:,0]
        y = self.enc(y)
        ###############
        if Sequence:
            output, _ = self.seq_proj(x)
            output += x
            x = output[:,-1,:] 

        scores = torch.matmul(self._h(y), self._g(x).t())

        scores_lst = [scores]
        return scores_lst
        # return scores



class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        # self.inv_mat = None
        self.rec = False
        if Sequence:
            if self.rec:
                self.enc = nn.Linear(dim, hidden_dim)   
                self.dec = nn.Linear(hidden_dim, dim)  
                self.seq_proj = nn.Linear(hidden_dim, hidden_dim) 
            else: 
                self.seq_proj = nn.Linear(dim,dim)
                
    def forward(self, x, y):
        
        x = x[:,0]
        ###############
        if Sequence:
            if self.rec:
                xhat = self.enc(x); x_rec = self.dec(xhat)
                yhat = self.enc(y); y_rec = self.dec(yhat)
                ytilde = self.seq_proj(xhat)
                scores = -torch.cdist(ytilde, yhat.detach())**2# y)**2 # 
            else:
                ytilde = self.seq_proj(x)
                scores = -torch.cdist(ytilde, y)**2
                x_rec, y_rec = x, y
        return scores, torch.cat([x,y]), torch.cat([x_rec, y_rec])
        

def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)