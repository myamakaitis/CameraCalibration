import numpy as np

class Pinhole:
    def __init__(self, f, theta, phi, psi):

        self.theta = theta
        self.phi = phi
        self.psi = psi

        self.dx, self.dy # pixel pitch in x and y
        self.Cx, self.Cy # coordinates of center pixel

        self.tx, self.ty, self.tz

        self.f = f

    def RotMat_EulerAngles(self, phi, psi, theta):

        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_tht, cos_tht = np.sin(theta), np.cos(theta)

        self.R = np.array([
            [cos_psi*cos_tht, sin_psi*cos_tht, -sin_tht],
            [-sin_psi*cos_phi + cos_psi*sin_tht*cos_phi, cos_psi*cos_phi + sin_psi*sin_tht*sin_phi, cos_tht*sin_phi],
            [sin_psi*sin_phi + cos_psi*sin_tht*cos_phi, -cos_tht*sin_phi + sin_psi*sin_tht*cos_phi, cos_tht*cos_phi ]
        ])

        return self.R

    def TransVec(self):

        self.T = np.array([self.tx, self.ty, self.tz])

        return self.T

    def RigidBodyTransform(self, Xw, Yw, Zw):
        # 3D world coordiantes -> 3D cam coordinates
        return self.R @ np.array([Xw, Yw, Zw]) + self.T

    def PerspectiveEqn(self, x, y, z):
        # 3d cam coordinates -> Undistorted Image Coordinates
        Xu, Yu = self.f * (x/z), self.f * (y/z)

        return Xu, Yu

    def RadialDistortion(self, Xu, Yu):
        # undistorted image coordinates -> distorted coordinates

        R2 = Xu**2 + Yu**2

        X = Xu/(1 + self.k*R2)
        Y = Yu/(1 + self.k*R2)

        return X, Y

    def RealCoordinates(self, X, Y):
        # distorted cam coordinates -> pixel coordinates

        Xf = self.dx*X + self.Cx
        Yf = self.dy*Y + self.Cy

        return Xf, Yf

    def Map(self, Xw, Yw, Zw):

        x, y, z = self.RigidBodyTransform(Xw, Yw, Zw)
        Xu, Yu = self.PerspectiveEqn(x, y, z)
        X, Y = self.RadialDistortion(Xu, Yu)
        Xf, Yf = self.RealCoordinates(X, Y)

        return Xf, Yf

class Polynomial:

    def __init__(self, MaxOrders = (3, 3, 3)):


        MaxOrders = np.array([1, 1, 1])
        for m in range(3):
            self.MaxOrders = MaxOrders[m] + 1

        self.Cx = np.zeros(self.MaxOrders, np.float64) # Polynomial Coefficients for u-coordinate
        self.Cy = np.zeros(self.MaxOrders, np.float64)

        self.CoefficientIndices()

    def CoefficientIndices(self):
        Cx = np.full(self.MaxOrders, -1, np.float64)

        for i in range(0, self.MaxOrders[0]):
            for j in range(0, self.MaxOrders[1] - i):
                for k in range(0, self.MaxOrders[2] - i - j):
                    Cx[i, j, k] = 100 * i + 10 * j + k
        (i, j, k) = np.where(Cx != -1)

        # The index pairs of the non-zero coefficients
        self.Ci = i
        self.Cj = j
        self.Ck = k
        self.N_coefficients = len(i)

    def Poly3d_3rdOrder(self, Cijk, x, y, z):

        val = 0
        for i in range(4):
            for j in range(4 - i):
                for k in range(4 - i - j):
                    val += Cijk[i, j, k]*x**i * y**j * z**k

    def Map(self, Xw, Yw, Zw):

        u, v = self.Upoly(Xw, Yw, Zw), self.Vpoly(Xw, Yw, Zw)

        return u, v

    def FitPoly(self, x, y, z, val):

        Cijk = np.zeros((4, 4, 4), dtype=np.float64)

        # combinations of i, j, k to create third order polynomial \sum C_{ijk} (x^i + y^j + z^k)

        X = np.empty((len(val), self.N_coefficients), dtype=np.float64)

        for l in range(self.N_coefficients):
            X[:, l] = x**self.Ci[l] * y**self.Cj[l] * z**self.Ck[l]

        Cs, e_px, rank, _ = np.linalg.lstsq(X, val)

        if rank < self.N_coefficients:
            raise Warning("Feature Matrix is Rank Deficient")

        Cijk[self.Ci, self.Cj, self.Ck] = Cs

        return Cs, e_px
