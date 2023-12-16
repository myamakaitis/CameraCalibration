import numpy as np
import matplotlib.pyplot as pyp
import pandas as pd
from CamModels import Pinhole
from numpy.random import random


File = "Marks_B.csv"
Data = pd.read_csv(File)

Data = Data[(1 < Data["Z"]) & (Data["Z"] < 11)]

u, v = Data["u"].values, Data["v"].values
X, Y, Z = Data["X"].values, Data["Y"].values, Data["Z"].values

fig, ax = pyp.subplots()
for z_plane in np.sort(list(set(Z)))[::2]:

    uplane, vplane = u[Z == z_plane], v[Z == z_plane]

    ax.scatter(uplane, vplane, s=1, label=f"z = {z_plane}")

ax.legend()
fig.show()

Ratio = 0.1

Witheld = random(size=u.size) < Ratio
Used = np.invert(Witheld)

fig, (ax, axh) = pyp.subplots(1, 2, dpi=150, figsize=(8, 4))
fig.suptitle(f"Reprojection Errors")

# axes = axes.reshape(2, 8).T

RMSEf = []
RMSEt = []

def PlotProjErrors(u_e, v_e, e, ax_scatter, ax_hist, scatter_label, hist_label):
    axh = ax_hist
    ax = ax_scatter

    ax.set_title(scatter_label)
    ax.scatter(u_e[Used], v_e[Used], s=1, label='Fit')
    ax.scatter(u_e[Witheld], v_e[Witheld], s=1.5, label='Test', color='red')

    ax.set_ylabel("$\Delta v_e$ [px]")
    ax.set_xlabel("$\Delta u_e$ [px]")
    ax.set_aspect("equal")

    ax.set_ylim(-2.1, 2.1)
    ax.set_xlim(-2.1, 2.1)
    ax.grid(True)
    ax.legend(loc='lower right')

    axh.set_title(hist_label)
    logbins = np.logspace(np.log10(0.01), np.log10(5), 10)
    axh.hist(e[Used], bins=logbins, label='Fit')

    axh.set_ylim(0, 210)

    axhtest = axh.twinx()
    axhtest.hist(e[Witheld], bins=logbins, label="Test", alpha=0.8, color='red')

    axh.set_ylabel("Number (Fit)")
    axhtest.set_ylabel("Number (Test)")
    axhtest.set_ylim(axh.get_ylim()[0], 2 * Ratio * axh.get_ylim()[1])
    axh.set_xlabel("error [px]")
    axh.set_xlim(0.01, 5)
    axh.set_xscale('log')
    axh.grid(True)
    axh.legend(loc=(0.71, 0.89), frameon=False)
    axhtest.legend(frameon=False, loc=(0.71, 0.84))

    fig.tight_layout()


Cam1 = Pinhole((u.mean(), v.mean()), 6.5e-3)

Cam1.Fit(u, v, X, Y, Z)
Cam1.k = 0

u_rp, v_rp = Cam1.Map(X, Y, Z)

# u_rp, v_rp = polyCam.Map(X, Y, Z)

U_e, V_e = u_rp - u, v_rp - v
e = (U_e ** 2 + V_e ** 2)

RMSE_test = np.sqrt(np.sum(1 / len(e[Witheld]) * e[Witheld]))
RMSE_fit = np.sqrt(np.sum(1 / len(e[Used]) * e[Used]))

print(f"{RMSE_test=}")
print(f"{RMSE_fit=}")

PlotProjErrors(U_e, V_e, e, ax, axh, f"Error", "Histogram")

fig.show()
