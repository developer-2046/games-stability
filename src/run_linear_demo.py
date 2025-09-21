import torch, math, os
from maps import LinearGame, simplex_proj
from methods import gda_step, ogda_step, extragradient_step
from metrics import residual
from plots import phase_portrait_2d
import matplotlib.pyplot as plt

torch.manual_seed(0)

DEVICE = "cpu"
D = 2
N_STEPS = 2000
ETA = 0.05
ALPHA = 0.9   # skew weight; closer to 1 => more rotation
SYM_MU = 0.2
SKEW_W = 1.0

game = LinearGame(d=D, sym_mu=SYM_MU, skew_omega=SKEW_W, alpha=ALPHA, device=DEVICE)
F = lambda x: game.F(x)

# init
x0 = torch.tensor([[1.2, -0.8]], device=DEVICE)
x_prev = x0.clone()
x = x0.clone()

def run(method="gda"):
    global x, x_prev
    x = x0.clone(); x_prev = x0.clone()
    traj = [x.clone()]
    res = []
    for k in range(N_STEPS):
        if method == "gda":
            x = gda_step(x, F, ETA)
        elif method == "ogda":
            xn = ogda_step(x, x_prev, F, ETA)
            x_prev, x = x, xn
        elif method == "eg":
            x = extragradient_step(x, F, ETA)
        else:
            raise ValueError
        traj.append(x.clone())
        res.append(residual(F, x).item())
    return torch.cat(traj, dim=0), res

# Vector field
fig, ax = plt.subplots()
phase_portrait_2d(F, ax=ax)
fig.savefig("experiments/figs/field.png", dpi=160)

# Runs
os.makedirs("experiments/figs", exist_ok=True)
for m in ["gda", "ogda", "eg"]:
    traj, res = run(m)
    # trajectory
    fig, ax = plt.subplots()
    ax.plot(traj[:,0].cpu(), traj[:,1].cpu(), '-o', markersize=2)
    ax.set_aspect('equal'); ax.set_title(f"Trajectory: {m.upper()}")
    fig.savefig(f"experiments/figs/traj_{m}.png", dpi=160)
    # residual
    plt.figure()
    plt.semilogy(res)
    plt.xlabel("k"); plt.ylabel("residual"); plt.title(f"Residual: {m.upper()}")
    plt.grid(True, which='both', ls=':')
    plt.savefig(f"experiments/figs/res_{m}.png", dpi=160)
    plt.close('all')

print("Saved figs to experiments/figs/")
