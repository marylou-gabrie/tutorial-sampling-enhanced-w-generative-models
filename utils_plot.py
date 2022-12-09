import matplotlib.pyplot as plt
import torch
import numpy as np

plot_lim = 5
lims = {'x_min':- plot_lim, 'x_max': plot_lim, 'y_min':- plot_lim, 'y_max': plot_lim}
device = 'cpu'

def grab(x):
    return x.detach().cpu().numpy()

def plot_density(log_prob_1, log_prob_2=None, lims=lims, n_points=100, ax=None,         
                 title=''):
    
    x_range = torch.linspace(lims['x_min'], lims['x_max'], n_points, device=device)

    if lims['y_min'] is None:
        y_range = x_range.clone()
    else:
        y_range = torch.linspace(lims['y_min'], lims['y_max'], n_points, device=device)

    grid = torch.meshgrid(x_range, y_range)
    xys = torch.stack(grid).reshape(2, n_points ** 2).T.to(device)

    Us_1 = grab(log_prob_1(xys).reshape(n_points, n_points).T)

    plt.figure() if ax is None else plt.sca(ax)
    plt.contourf(x_range, y_range, np.exp(Us_1), 20, cmap='GnBu')
    plt.gca().set_aspect('equal')
    plt.gca().set_title(title)

    if log_prob_2 is not None:
        Us_2 = grab(log_prob_2(xys).reshape(n_points, n_points).T)
        plt.contour(x_range, y_range, np.exp(Us_2), 20, colors='k', linestyles=':', alpha=0.5)
    
