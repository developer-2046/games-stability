import torch

@torch.no_grad()
def proj_euclid(x, proj=None):
    return x if proj is None else proj(x)

@torch.no_grad()
def gda_step(x, F, eta, proj=None):
    return proj_euclid(x - eta * F(x), proj)

@torch.no_grad()
def ogda_step(x, x_prev, F, eta, proj=None):
    g, g_prev = F(x), F(x_prev)
    return proj_euclid(x - 2*eta*g + eta*g_prev, proj)

@torch.no_grad()
def extragradient_step(x, F, eta, proj=None):
    y = proj_euclid(x - eta * F(x), proj)
    return proj_euclid(x - eta * F(y), proj)
