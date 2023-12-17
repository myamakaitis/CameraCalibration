import numpy as np
import matplotlib.pyplot as pyp
import pandas as pd
from CamModels import Pinhole
from numpy.random import random

from matplotlib.cm import get_cmap

cmap = get_cmap('viridis')


def ErrorColor(error):
    error /= 25
    error[error>1] = 1

    return cmap(error)

File = "Marks_A.csv"
Data = pd.read_csv(File)

# Data = Data[(1 < Data["Z"]) & (Data["Z"] < 11)]

Data.sort_values(by="u", axis=0, inplace=True)

u, v = Data["u"].values, Data["v"].values
X, Y, Z = Data["X"].values, Data["Y"].values, Data["Z"].values

# axes = axes.reshape(2, 8).T

Ratio = 0.1
Sample = random(size=u.size) < Ratio

Sample[:] = False
Sample[::10] = True

RMSE = []

fig, ax = pyp.subplots(2, 3, figsize=(3*4, 8), sharex=True, sharey=True)
ax = ax.ravel()

for axis in ax:
    axis.set_aspect('equal')
    axis.grid(True)

ax[0].set_title("Original Points")
ax[0].scatter(u[Sample], v[Sample], marker="o", color='k',
              edgecolor='k', facecolor="None")
ax[0].set_ylim(-1, 1001)
ax[0].set_xlim(-1, 1001)


for i, (cx, cy) in enumerate(zip(np.linspace(0, 2*u.mean(), 5), np.linspace(0, 2*v.mean(), 5))):

    Cam1 = Pinhole((cx, cy), 1)

    Cam1.Fit(u, v, X, Y, Z)
    Cam1.k = 0

    u_rp, v_rp = Cam1.Map(X, Y, Z)

    U_e, V_e = u_rp - u, v_rp - v
    e = (U_e ** 2 + V_e ** 2)

    RMSE.append(np.sqrt(np.sum(1 / len(e) * e)))

    print(np.linalg.det(Cam1.R) - 1)

    ax[i+1].set_title(f"Center Shift: ({cx:.0f}, {cy:.0f})")
    ax[i+1].scatter(u_rp[Sample], v_rp[Sample], color=ErrorColor(np.sqrt(e[Sample])), edgecolor='k')
    # PlotProjErrors(U_e, V_e, e, ax, axh, f"Error", "Histogram")

fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])


fig_fake, im_ax = pyp.subplots(1)
im = im_ax.contourf(np.zeros((2, 2)), np.zeros((2, 2)), np.array([[0, 25], [0, 0]]),
                    vmin=0, vmax=25, cmap='viridis', levels=501, extend='max')

fig.colorbar(im, cax=cbar_ax)

fig.show()

print(RMSE)

shiftX = np.linspace(0, 800)
shiftY = np.linspace(0, 1000)

ShiftX, ShiftY = np.meshgrid(shiftX, shiftY)

shape = ShiftX.shape
RMSE = []

for sx, sy in zip(ShiftX.ravel(), ShiftY.ravel()):

    Cam1 = Pinhole((sx, sy), 1)

    try:
        Cam1.Fit(u, v, X, Y, Z)
    except ValueError:
        RMSE.append(None)
        continue
    Cam1.k = 0

    u_rp, v_rp = Cam1.Map(X, Y, Z)

    U_e, V_e = u_rp - u, v_rp - v
    e = (U_e ** 2 + V_e ** 2)

    RMSE.append(np.sqrt(np.sum(1 / len(e) * e)))

RMSE = np.array(RMSE).reshape(shape).astype(np.float64)

fig, ax = pyp.subplots(figsize=(5,5))


ax.set_title("$\log_{10}($RMSE$)$ Reprojection Error")
ax.set_aspect('equal')

ax.set_xlabel("Center X")
ax.set_ylabel("Center Y")

log10rmse = np.log10(RMSE)

log10rmse[log10rmse > 3] = 3

cb = ax.contourf(ShiftX, ShiftY, log10rmse, vmin=0.5, vmax=2.5,
                 extend='max', levels=501)
fig.colorbar(cb, ticks=(0.8, 1, 1.5, 2, 2.5))

fig.tight_layout()
fig.show()
