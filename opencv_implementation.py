import cv2
from cv2 import calibrateCamera
import pandas as pd
import numpy as np

df = pd.read_csv("TestFits/TestPinhole4545.csv")
X, Y, Z = df["X"].values, df["Y"].values, df["Z"].values
u, v = df["u"].values, df["v"].values


obj_pts = []
obj_pts.append(np.array([X, Y, Z], dtype=np.float32).T)

img_pts = []
img_pts.append(np.array([u, v], dtype=np.float32).T)

fx = 7
fy = 7
Cx = 0
Cy = 0

intrinsics = np.array([[fx, 0, Cx],
                       [0, fy, Cy],
                       [0, 0, 1]],
                      dtype=np.float32)

ret, mtx, dist, rvecs, tvecs = calibrateCamera(obj_pts, img_pts, intrinsics, None, None, cv2.CALIB_USE_INTRINSIC_GUESS)

print(f"{ret=}")
print(f"{mtx=}")
print(f"{dist=}")
print(f"{rvecs=}")
print(f"{tvecs=}")