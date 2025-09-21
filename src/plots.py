import torch, matplotlib.pyplot as plt

def phase_portrait_2d(F, xlim=(-2,2), ylim=(-2,2), n=21, ax=None):
    if ax is None: fig, ax = plt.subplots()
    xs = torch.linspace(xlim[0], xlim[1], n)
    ys = torch.linspace(ylim[0], ylim[1], n)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    P = torch.stack([X.flatten(), Y.flatten()], dim=1)
    V = F(P)
    U = V[:,0].reshape(n,n).cpu()
    W = V[:,1].reshape(n,n).cpu()
    ax.quiver(X.cpu(), Y.cpu(), U, W, angles='xy', scale_units='xy', scale=1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.set_title('Vector field F(x)')
    return ax
