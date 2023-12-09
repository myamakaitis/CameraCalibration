import numpy as np


class Pinhole:
    def __init__(self, f, Center, Angles, k):
        self.theta = Angles[0]
        self.phi = Angles[1]
        self.psi = Angles[2]

        self.tx, self.ty, self.tz = Center

        self.f = f
        self.k = k

        self.RotMat_EulerAngles(*Angles)
        self.TransVec()
        self.ACalc()
        self.RtCalc()

    def RotMat_EulerAngles(self, phi, psi, theta):
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_tht, cos_tht = np.sin(theta), np.cos(theta)

        self.R = np.array([
            [cos_psi * cos_tht, sin_psi * cos_tht, -sin_tht],
            [-sin_psi * cos_phi + cos_psi * sin_tht * sin_phi, cos_psi * cos_phi + sin_psi * sin_tht * sin_phi,
             cos_tht * sin_phi],
            [sin_psi * sin_phi + cos_psi * sin_tht * cos_phi, -cos_psi * sin_phi + sin_psi * sin_tht * cos_phi,
             cos_tht * cos_phi]
        ])

        return self.R

    def TransVec(self):
        self.T = np.array([self.tx, self.ty, self.tz])

        return self.T

    def RtCalc(self):
        self.Rt = np.zeros((3, 4), dtype=np.float64)
        self.Rt[:3, :3] = self.R
        self.Rt[:, 3] = self.T

    def ACalc(self):
        self.A = np.array([[-self.f, 0, 0],
                           [0, self.f, 0, ],
                           [0, 0, 1]])

    def PCalc(self):
        return self.A @ self.Rt

    def RigidBodyTransform(self, Xw, Yw, Zw):
        # 3D world coordiantes -> 3D cam coordinates
        return self.Rt @ np.array([Xw, Yw, Zw, 1])

    def PerspectiveEqn(self, x, y, z):
        # 3d cam coordinates -> Undistorted Image Coordinates
        Xu, Yu = -self.f * (x / z), -self.f * (y / z)

        return Xu, Yu

    def RadialDistortion(self, Xu, Yu):
        # undistorted image coordinates -> distorted coordinates

        R2 = Xu ** 2 + Yu ** 2

        X = Xu / (1 + self.k * R2)
        Y = Yu / (1 + self.k * R2)

        return X, Y

    def RealCoordinates(self, X, Y):
        # distorted cam coordinates -> pixel coordinates

        Xf = X
        Yf = Y

        return Xf, Yf

    def Map(self, Xw, Yw, Zw):
        x, y, z = self.RigidBodyTransform(Xw, Yw, Zw)
        Xu, Yu = self.PerspectiveEqn(x, y, z)
        X, Y = self.RadialDistortion(Xu, Yu)
        Xf, Yf = self.RealCoordinates(X, Y)

        return Xf, Yf

class Polynomial:

    def __init__(self, MaxOrders = (3, 3, 3)):

        self.MaxOrders = np.array(MaxOrders) + 1
        self.CoefficientIndices()

    @property
    def RowLabels(self):
        self._rl = []

        for l in range(self.N_coefficients):
            row_str = ""
            if self.Ci[l] > 0:
                row_str = row_str + f"x^{self.Ci[l]} "
            if self.Cj[l] > 0:
                row_str = row_str + f"y^{self.Cj[l]} "
            if self.Ck[l] > 0:
                row_str = row_str + f"z^{self.Ck[l]}"
            if row_str == "":
                row_str = "1"
            self._rl.append(row_str)

        return self._rl

    def CoefficientIndices(self):
        Cx = np.full(self.MaxOrders, -1, np.float64)

        for i in range(0, self.MaxOrders[0]):
            for j in range(0, self.MaxOrders[1]):
                for k in range(0, self.MaxOrders[2]):
                    if (i + j + k) < self.MaxOrders.max():
                        Cx[i, j, k] = 100 * i + 10 * j + k
        (i, j, k) = np.where(Cx != -1)

        # The index pairs of the non-zero coefficients
        self.Ci = i
        self.Cj = j
        self.Ck = k
        self.N_coefficients = len(i)

    def NewPoly3rdOrder(self, coeffs):
        def Poly(x, y, z):
            val = 0
            for l in range(self.N_coefficients):
                val += coeffs[l] * x ** self.Ci[l] * y ** self.Cj[l] * z ** self.Ck[l]

            return val

        return Poly

    def Map(self, Xw, Yw, Zw):

        u, v = self.Upoly(Xw, Yw, Zw), self.Vpoly(Xw, Yw, Zw)

        return u, v

    def FitPoly(self, x, y, z, val):

        # Cijk = np.zeros((self.MaxOrders), dtype=np.float64)

        # combinations of i, j, k to create third order polynomial \sum C_{ijk} (x^i + y^j + z^k)

        X = np.empty((len(val), self.N_coefficients), dtype=np.float64)

        for l in range(self.N_coefficients):
            X[:, l] = x**self.Ci[l] * y**self.Cj[l] * z**self.Ck[l]

        Cs, e_px, rank, _ = np.linalg.lstsq(X, val, rcond=None)

        if rank < self.N_coefficients:
            raise Warning("Feature Matrix is Rank Deficient")

        # Cijk[self.Ci, self.Cj, self.Ck] = Cs

        return Cs, e_px

    def FitCam(self, u, v, x, y, z):

        self.u_coeffs, u_e_px = self.FitPoly(x, y, z, u)
        self.v_coeffs, v_e_px = self.FitPoly(x, y, z, v)

        self.Upoly = self.NewPoly3rdOrder(self.u_coeffs)
        self.Vpoly = self.NewPoly3rdOrder(self.v_coeffs)

        return np.sqrt(u_e_px**2 + v_e_px**2)


if __name__ == '__main__':

    for o in range(1, 9):
        polyCam = Polynomial(MaxOrders=(o, o, 2))
        print(polyCam.N_coefficients)
        print(polyCam.RowLabels)

        import matplotlib.pyplot as pyp

        Nper = 51

        seq = np.arange(0, Nper, dtype=np.float64)
        seq0 = np.zeros_like(seq)
        seqN = np.full_like(seq, Nper)

        t = np.linspace(-10, 10, 11)

        x, y = np.meshgrid(t, t)

        z = np.zeros_like(x).ravel()

        Xtest = np.concatenate([x.ravel(), x.ravel(), x.ravel()])
        Ytest = np.concatenate([y.ravel(), y.ravel(), y.ravel()])
        Ztest = np.concatenate([z - 5, z, z + 5])

        TestPoints = (Xtest, Ytest, Ztest)

        colors = len(z) * ['red'] + len(z) * ['seagreen'] + len(z) * ['navy']

        fig = pyp.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Xtest, Ytest, Ztest, color=colors)

        ax.set_title("Simulated Points")
        # fig.show()

        Angles = np.radians(np.array([15, 0, 45]))
        Cam1 = Pinhole(10, (0, 0, -30), Angles, .01)
        P1 = Cam1.PCalc()

        u1, v1 = Cam1.Map(Xtest, Ytest, Ztest)

        e_poly = polyCam.FitCam(u1[::2], v1[::2], Xtest[::2], Ytest[::2], Ztest[::2])

        u2, v2 = polyCam.Map(Xtest, Ytest, Ztest)

        fig, (ax1, ax2) = pyp.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
        fig.suptitle(f"Order = {o}")
        ax1.scatter(u1, v1, color=colors, alpha=0.5, s=3)

        ax1.set_title("Projected Points Cam 1")
        ax1.set_aspect('equal')
        ax1.grid(True)

        ax2.set_title("Projected Points Cam 2")
        ax2.scatter(u2, v2, color=colors, alpha=0.5, s=3)
        ax2.set_aspect('equal')
        ax2.grid(True)

        fig.show()

        e_u, e_v = u2 - u1, v2 - v1
        e = np.sqrt(np.sum(1/len(e_u) * (e_u**2 + e_v**2)))

        e_train = np.sqrt(np.sum(1 / len(e_u[::2]) * (e_u[::2] ** 2 + e_v[::2] ** 2)))
        e_test = np.sqrt(np.sum(1 / len(e_u[1::2]) * (e_u[1::2] ** 2 + e_v[1::2] ** 2)))

        print(f"\nOrder : {o}")
        print(f"Error: {e}")
        print(f"Train Error {e_train}")
        print(f"Test Error {e_test}")



