import numpy as np
import matplotlib.pyplot as pyp
import pandas as pd
from CamModels import Pinhole, OPENCV_cam, FisheyeThinPrism, PinholeDLT
from numpy.random import random
from scipy.spatial.transform import Rotation

from matplotlib.cm import get_cmap

cmap = get_cmap('viridis')

MaxErrorColor = 1

def PlotProjErrors(u_e, v_e, e, ax_scatter, ax_hist, scatter_label, hist_label):
    axh = ax_hist
    ax = ax_scatter

    ax.set_title(scatter_label)
    ax.scatter(u_e, v_e, s=1, label='Fit')

    ax.set_ylabel("$\Delta v_e$ [px]")
    ax.set_xlabel("$\Delta u_e$ [px]")
    ax.set_aspect("equal")

    ax.hlines(0, -5, 5, color='k')
    ax.vlines(0, -5, 5, color='k')

    # ax.set_ylim(-2.1, 2.1)
    # ax.set_xlim(-2.1, 2.1)
    ax.grid(True)
    ax.legend(loc='lower right')

    axh.set_title(hist_label)
    logbins = np.logspace(np.log10(np.sqrt(e).min()), np.log10(np.sqrt(e).max()), 20)
    axh.hist(np.sqrt(e), bins=logbins, label='Fit')

    # axh.set_ylim(0, 210)

    axh.set_ylabel("Number (Fit)")

    axh.set_xlabel("error [px]")
    axh.set_xscale('log')
    axh.grid(True)
    # axh.legend(loc=(0.71, 0.89), frameon=False)

    fig.tight_layout()

def ErrorColor(error):
    error /= MaxErrorColor
    error[error>1] = 1

    return cmap(error)

# File = "TestFits/Marks_D.csv"
# Data = pd.read_csv(File)

# # Data = Data[(1 < Data["Z"]) & (Data["Z"] < 11)]

# # Data.sort_values(by="u", axis=0, inplace=True)

# u, v = Data["u"].values, Data["v"].values
# # u, v = Data["Xcam2"].values, Data["Ycam2"].values
# X, Y, Z = Data["X"].values, Data["Y"].values, Data["Z"].values
# X, Y, Z = Data["x"].values, Data["y"].values, Data["z"].values
# axes = axes.reshape(2, 8).T


File = "InvMic_MarkLocs_CamF.csv"
Data = pd.read_csv(File)
u, v = Data['u'], Data['v']
X, Y, Z = Data["x"].values, Data["y"].values, Data["z"].values

fig, ax = pyp.subplots(figsize=(16, 16), sharex=True, sharey=True)

figh, axh = pyp.subplots(figsize=(8,8))
figs, axs = pyp.subplots(figsize=(8,8))

ax.set_aspect('equal')
ax.grid(True)

ax.set_title("Original Points")
ax.scatter(u, v, marker="o", color='k',
              edgecolor='k', facecolor="None")

# for i, (cx, cy) in enumerate(zip(np.linspace(0, 2*u.mean(), 13), np.linspace(0, 2*v.mean(), 13))):

    # Cam1 = FisheyeThinPrism((cx, cy), 1)
Cam1 = PinholeDLT()
Cam1.Fit(u, v, X, Y, Z)

u_rp, v_rp = Cam1.Map(X, Y, Z)

U_e, V_e = u_rp - u, v_rp - v
e = np.sqrt(U_e ** 2 + V_e ** 2)

rmse = (np.sqrt(np.sum(e**2) / e.size))
print(rmse)
# print(np.linalg.det(Cam1.R) - 1)

# label = f"Center Shift: ({cx:.0f}, {cy:.0f})"

# ax.set_title(label)
ax.scatter(u_rp, v_rp, color=ErrorColor(np.sqrt(e)), edgecolor='k')
# PlotProjErrors(U_e, V_e, e, ax, axh, f"Error", "Histogram")


PlotProjErrors(U_e, V_e, e, axs, axh, 'scatter', 'hist')

fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])


fig_fake, im_ax = pyp.subplots(1)
im = im_ax.contourf(np.zeros((2, 2)), np.zeros((2, 2)), np.array([[0, MaxErrorColor], [0, 0]]),
                    vmin=0, vmax=MaxErrorColor, cmap='viridis', levels=501, extend='max')

fig.colorbar(im, cax=cbar_ax)

fig.show()
figh.show()
figs.show()

print(f"RMSE = {Cam1.RMSE(X, Y, Z, u, v)}")


print(Cam1.T)
print(Cam1.R)
print(Cam1.K)
