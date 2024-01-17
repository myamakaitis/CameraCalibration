import numpy as np
import matplotlib.pyplot as pyp
import pandas as pd
from CamModels import Polynomial
from numpy.random import random


File = "C_cal.csv"
Data = pd.read_csv(File)

# Data = Data[(1 < Data["Z"]) & (Data["Z"] < 11)]

# u, v = Data["u"].values, Data["v"].values
# X, Y, Z = Data["X"].values, Data["Y"].values, Data["Z"].values

u, v = Data["Xcam1"].values, Data["Ycam1"].values
X, Y, Z = Data["x"].values, Data["y"].values, Data["z"].values

fig, ax = pyp.subplots()
for z_plane in np.sort(list(set(Z)))[::2]:

    uplane, vplane = u[Z == z_plane], v[Z == z_plane]

    ax.scatter(uplane, vplane, s=1, label=f"z = {z_plane}")

ax.legend()
fig.show()

Ratio = 0.1

polyCam = Polynomial(MaxOrders=(3, 3, 3))
polyCam.FitCam(u, v, X, Y, Z)

u_rp, v_rp = polyCam.Map(X, Y, Z)

U_e, V_e = u_rp - u, v_rp - v
e = (U_e ** 2 + V_e ** 2)

MaxError = 30
u, v = u[e < MaxError], v[e < MaxError]
X, Y, Z = X[e < MaxError], Y[e < MaxError], Z[e < MaxError]

Witheld = random(size=u.size) < Ratio
Used = np.invert(Witheld)

fig, axes = pyp.subplots(4, 4, dpi=150, figsize=(16, 16))
fig.suptitle(f"Reprojection Errors")

# axes = axes.reshape(2, 8).T

Orders = np.arange(1, 9)
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
    logbins = np.logspace(np.log10(0.05), np.log10(3), 20)
    axh.hist(e[Used], bins=logbins, label='Fit')

    axh.set_ylim(0, 210)

    axhtest = axh.twinx()
    axhtest.hist(e[Witheld], bins=logbins, label="Test", alpha=0.8, color='red')

    axh.set_ylabel("Number (Fit)")
    axhtest.set_ylabel("Number (Test)")
    axhtest.set_ylim(axh.get_ylim()[0], 2 * Ratio * axh.get_ylim()[1])
    axh.set_xlabel("error [px]")
    axh.set_xlim(0.05, 10)
    axh.set_xscale('log')
    axh.grid(True)
    axh.legend(loc=(0.71, 0.89), frameon=False)
    axhtest.legend(frameon=False, loc=(0.71, 0.84))

    fig.tight_layout()


for d in Orders:
    polyCam = Polynomial(MaxOrders=(d, d, np.min([3, d])))

    polyCam.FitCam(u[Used], v[Used],
                   X[Used], Y[Used], Z[Used])

    u_rp, v_rp = polyCam.Map(X, Y, Z)

    U_e, V_e = u_rp - u, v_rp - v
    e = (U_e ** 2 + V_e ** 2)

    RMSE_test = np.sqrt(np.sum(1 / len(e[Witheld]) * e[Witheld]))
    RMSE_fit = np.sqrt(np.sum(1 / len(e[Used]) * e[Used]))


    RMSEt.append(RMSE_test)
    RMSEf.append(RMSE_fit)

    ax, axh = axes[2 * ((d-1)//4), (d-1) % 4], axes[1 + 2 * ((d-1)//4), (d-1) % 4]

    e = np.sqrt(e)
    PlotProjErrors(U_e, V_e, e, ax, axh, f"Order = {d}", "Histogram")
    print(polyCam.RowLabels)


fig.show()

fig, ax = pyp.subplots()
ax.plot(Orders, RMSEf, label="Fit", marker='x')
ax.plot(Orders, RMSEt, label="Test", marker='o', color='r')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel("Polynomial Order")
ax.set_ylabel("RMSE [px]")

ax.set_yticks((0.5, 1, 2, 4))
ax.set_yticklabels((0.5, 1, 2, 4))
ax.grid(True)
fig.show()
print(RMSEf)

