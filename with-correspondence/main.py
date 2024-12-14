import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from estimators import estimate_mutual_information
from tqdm import tqdm
# define the dimension of the Gaussian

dim = 128  # 20

# define the training procedure

CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
}

BASELINES = {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(dim=dim, hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    'gaussian': lambda: log_prob_gaussian,
}


def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()

    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'], cubic=data_params['cubic'])
        mi = estimate_mutual_information(
            mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None), **kwargs)
        loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()

        return mi

    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    estimates = []
    for i in tqdm(range(opt_params['iterations'])): # 
        mi = train_step(rhos[i], data_params, mi_params)
        mi = mi.detach().cpu().numpy()
        estimates.append(mi)

    return np.array(estimates)

data_params = {
    'dim': dim,
    'batch_size': 64,  # 64,
    'cubic': None
}

critic_params = {
    'dim': dim,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}

opt_params = {
    'iterations': 50000,  #0,  # 20000
    'learning_rate': 5e-4,
}

# Train for 20000 steps for each case.

mi_numpys = dict()

for critic_type in ['concat']:#, 'separable']:
    mi_numpys[critic_type] = dict()
    for estimator in ['smile']: #'infonce', 'nwj', 'js', 
        mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
        mis = train_estimator(critic_params, data_params, mi_params, opt_params)
        mi_numpys[critic_type][f'{estimator}'] = mis


# Plotting helper functions.

def find_name(name):
    if 'smile_' in name:
        clip = name.split('_')[-1]
        return f'SMILE ($\tau = {clip}$)'
    else:
        return {
            'infonce': 'CPC',
            'js': 'JS',
            'nwj': 'NWJ',
            'flow': 'GM (Flow)',
            'smile': 'SMILE (tau=infty)'
        }[name]

def find_legend(label):
    return {'concat': 'Joint critic', 'separable': 'Separable critic'}[label]


# Plot 5 of the results, InfoNCE, NWJ, Smile 1.0, 5.0, infty

ncols = 1  # 5
nrows = 1
EMA_SPAN = 200
fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axs = np.ravel(axs)

mi_true = mi_schedule(opt_params['iterations'])


estimator = 'smile'
for i, clip in enumerate([ None]):  # 1.0, 5.0,
    if clip is None:
        key = estimator
    else:
        key = f'{estimator}_{clip}'

    plt.sca(axs[0])  
    for net in ['concat']:
        mis = mi_numpys[net][key]
        EMA_SPAN = 200
        p1 = plt.plot(mis, alpha=0.3)[0]
        mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        plt.plot(mis_smooth, c=p1.get_color(), label="EvoRate") 
    plt.plot(mi_true, color='k', label='True MI')
    plt.ylim(0, 150)  ##22)  # 11
    plt.xlim(0, opt_params['iterations'])
    plt.ylabel('MI (nats)')
    plt.xlabel('Steps')
plt.gcf().tight_layout()
plt.legend()
plt.savefig('foo.pdf')
# np.save('', )