# images.txt
# Two Lines Per Image
# Image ID Qw Qx QY QZ TX TY TZ CamID FileName
# (u, v, Point3D_ID), (u, v, Point3D_ID)....

# cameras.txt
# Camera ID, Model (Radial), Width, Height, fx, fy, cx, cy, k1, k2, p1, p2

# points3D.txt
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)

# Load D_Cal.csv

# Initiate 5 Camera Models
# Find Cx, and Cy which minimize error and use those

# Put camera parameters in cameras.txt
# Put rotation and translation parameters in images.txt

# For each point in D_cal.csv
# number it
# add its 3D points to points3D.txt
# give it an arbitrary color
# give it an arbitrary error

# for each image add its u, v coordinates to the images.txt file and reference the 3d point number
# keep track of the position of these 2d points and add them to the points3D.txt file with the image #

import numpy as np
import matplotlib.pyplot as pyp
import pandas as pd
from CamModels import OPENCV_cam, Pinhole, RADIAL, SIMPLE_RADIAL
from numpy.random import random
from scipy.spatial.transform import Rotation
from time import perf_counter

File = "C_Cal.csv"
Data = pd.read_csv(File)

Xw, Yw, Zw = Data["x"].values, Data["y"].values, Data["z"].values

Ucams = []
Vcams = []
Cams = []

Ncams = 4

RMSEs = np.full(4, np.inf, dtype=np.float64)

for i in range(Ncams):
    print(i)

    u, v = Data[f"Xcam{i}"].values, Data[f"Ycam{i}"].values

    Ucams.append(np.copy(u))
    Vcams.append(np.copy(v))

    Ncoarse = 35
    shiftX = np.linspace(0, 2 * u.mean(), Ncoarse)
    shiftY = np.linspace(0, 2 * v.mean(), Ncoarse)
    ShiftX, ShiftY = np.meshgrid(shiftX, shiftY)

    t0 = perf_counter()

    for sx, sy in zip(ShiftX.ravel(), ShiftY.ravel()):

        Cami = Pinhole((sx, sy), 1)
        Cami.Fit(u, v, Xw, Yw, Zw)

        rmse = Cami.RMSE(Xw, Yw, Zw, u, v)

        if rmse < RMSEs[i]:
            RMSEs[i] = rmse
            Cx, Cy = sx, sy

    tf = perf_counter()

    print(f"coarse done {tf - t0}")
    t0 = perf_counter()

    du1 = shiftX[1] - shiftX[0]
    dv1 = shiftY[1] - shiftY[0]

    Nfine = 31
    shiftX = np.linspace(-2*du1, 2*du1, Nfine)
    shiftY = np.linspace(-2*dv1, 2*dv1, Nfine)
    ShiftX, ShiftY = np.meshgrid(shiftX, shiftY)
    for sx, sy in zip(ShiftX.ravel(), ShiftY.ravel()):

        Cami = Pinhole((sx, sy), 1)
        Cami.Fit(u, v, Xw, Yw, Zw)

        rmse = Cami.RMSE(Xw, Yw, Zw, u, v)

        if rmse < RMSEs[i]:
            RMSEs[i] = rmse
            Cx, Cy = sx, sy

    tf = perf_counter()
    print(f"fine done {tf - t0}")

    Cam_i = Pinhole((Cx, Cy), 1)
    Cam_i.Fit(u, v, Xw, Yw, Zw)

    Cams.append(Cam_i)

with open("cameras.txt", 'w') as cameras, open("images.txt", "w") as images, open("points3D.txt", 'w') as points3d:

    imageCamLines = dict()
    imagePointLines = dict()

    for i in range(4):

        cami = Cams[i]
        cameras.write(f"{i+1} PINHOLE 4512 800 ")

        fx, fy = Cams[i].f*Cams[i].sx, Cams[i].f

        Cx, Cy = Cams[i].Cx, Cams[i].Cy
        # k1 = Cams[i].k1
        # k1, k2 = Cams[i].k1, Cams[i].k2
        # p1, p2 = Cams[i].p1, Cams[i].p2

        # cameras.write(f"{fx:.10f} {fy:.10f} {Cx:.10f} {Cy:.10f} {k1:.15f} {k2:.15f} {p1:.15f} {p2:.15f} \n")
        # cameras.write(f"{Cams[i].f:.10f} {Cx:.10f} {Cy:.10f} {k1:.15f}\n")
        cameras.write(f"{fx:.10f} {fy:.10f} {Cx:.10f} {Cy:.10f}\n")

        Qx, Qy, Qz, Qw = Rotation.from_matrix(Cams[i].R).as_quat()
        Tx, Ty, Tz = Cams[i].T

        imageCamLines[i] = f"{i+1} {Qw:.15f} {Qx:.15f} {Qy:.15f} {Qz:.15f} {Tx:.15f} {Ty:.15f} {Tz:.15f} {i+1} C_cam{i}_0001a.png\n"
        imagePointLines[i] = ""

    for j in range(len(Xw)):
        points3d.write(f"{j+1} {Xw[j]} {Yw[j]} {Zw[j]} 128 128 128 1")

        for i in range(Ncams):
            points3d.write(f" {i} {j}")
            imagePointLines[i] += f"{Ucams[i][j]:.10f} {Vcams[i][j]:.10f} {j} "

        points3d.write("\n")

    for i in range(Ncams):
        images.write(imageCamLines[i])
        images.write(imagePointLines[i])
        images.write("\n")

print(RMSEs)







