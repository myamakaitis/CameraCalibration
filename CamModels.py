import numpy as np
from scipy.optimize import minimize


class Pinhole:
    def __init__(self, Center, PixelPitch):

        self.Cx, self.Cy = Center[0], Center[1]
        self.dx = PixelPitch

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

    def CombineRT(self):
        self.Rt = np.zeros((3, 4), dtype=np.float64)
        self.Rt[:3, :3] = self.R
        self.Rt[:, 3] = self.T

    def PCalc(self):
        return self.A @ self.Rt

    def RigidBodyTransform(self, Xw, Yw, Zw):
        # 3D world coordiantes -> 3D cam coordinates
        ones = np.ones_like(Xw)
        return self.Rt @ np.array([Xw, Yw, Zw, ones])

    def PerspectiveEqn(self, x, y, z):
        # 3d cam coordinates -> Undistorted Image Coordinates
        Xu, Yu = self.f * (x / z), self.f * (y / z)

        return Xu, Yu

    def Distortion(self, Xu, Yu):
        # undistorted image coordinates -> distorted coordinates

        # R2 = Xu ** 2 + Yu ** 2

        # X = Xu / (1 + self.k * R2)
        # Y = Yu / (1 + self.k * R2)

        return Xu, Yu

    def RealCoordinates(self, X, Y):
        # distorted cam coordinates -> pixel coordinates

        u = self.sx*X/self.dx + self.Cx
        v = Y/self.dx + self.Cy

        return u, v

    def Map(self, Xw, Yw, Zw):
        x, y, z = self.RigidBodyTransform(Xw, Yw, Zw)
        Xu, Yu = self.PerspectiveEqn(x, y, z)
        X, Y = self.Distortion(Xu, Yu)
        u, v = self.RealCoordinates(X, Y)

        return u, v

    def RMSE(self, Xw, Yw, Zw, u_actual, v_actual):

        u_rp, v_rp = self.Map(Xw, Yw, Zw)
        U_e, V_e = u_rp - u_actual, v_rp - v_actual
        e = (U_e ** 2 + V_e ** 2)

        rmse = np.sqrt(1 / len(e) * np.sum(e))
        return rmse

    def Fit(self, u, v, Xw, Yw, Zw):

        up = (u - self.Cx)*self.dx
        vp = (v - self.Cy)*self.dx

        Tr = self.ComputeTr(up, vp, Xw, Yw, Zw)

        testpoint = np.array([Xw[0], Yw[0], Zw[0]])
        testmap = np.array([up[0], vp[0]])
        Tx, Ty, sx = self.CalcRT(Tr, testpoint, testmap)

        f, Tz = self.Approx_f_Tz(Ty, vp, Xw, Yw, Zw)

        self.T = np.array([Tx, Ty, Tz])
        self.f = f
        self.sx = sx

        self.CombineRT()

        self.Refine_f_Tz(Xw, Yw, Zw, u, v)

        self.CalcDistortion(Xw, Yw, Zw, u, v)

    def CalcDistortion(self, Xw, Yw, Zw, u_act, v_act):
        pass

    def Refine_f_Tz(self, Xw, Yw, Zw, u_act, v_act):

        guess = np.array([self.f, self.T[2]])

        minimize(self.Adjust_f_Tz, guess,
                 args=(Xw, Yw, Zw, u_act, v_act))

    def Adjust_f_Tz(self, fTz, Xw, Yw, Zw, u_act, v_act):

        self.f = fTz[0]
        self.T[2] = fTz[1]

        self.CombineRT()

        return self.RMSE(Xw, Yw, Zw, u_act, v_act)

    def Approx_f_Tz(self, Ty, vp, xw, yw, zw):

        yi = self.R[1, 0] * xw + self.R[1, 1]*yw + self.R[1, 2]*zw + Ty
        wi = self.R[2, 0] * xw + self.R[2, 1]*yw + self.R[2, 2]*zw


        sys = np.zeros((len(vp), 2))
        sys[:, 0] = yi
        sys[:, 1] = -vp

        rhs = vp*wi

        (f, Tz), _, _, _ = np.linalg.lstsq(sys, rhs, rcond=None)

        return f, Tz

    def ComputeTr(self, up, vp, xw, yw, zw):

        System = np.zeros((len(up), 7))

        System[:, 0] = vp*xw
        System[:, 1] = vp*yw
        System[:, 2] = vp*zw
        System[:, 3] = vp
        System[:, 4] = -up*xw
        System[:, 5] = -up*yw
        System[:, 6] = -up*zw

        Tr, _, _, _ = np.linalg.lstsq(System, up, rcond=None)

        # Tr = [ sx r1 / Ty, sx r2 / Ty, sx r3 / Ty,
        #        sx Tx / Ty, r4 / Ty, r5 / Ty, r6 / Ty ]

        return Tr

    def CalcRT(self, Tr, TestXYZ, TestUV):

        abs_Ty = (Tr[4]**2 + Tr[5]**2 + Tr[6]**2)**(-0.5)
        sx = (Tr[0]**2 + Tr[1]**2 + Tr[2]**2)**(0.5) * abs_Ty

        r1 = Tr[0]*abs_Ty/sx
        r2 = Tr[1]*abs_Ty/sx
        r3 = Tr[2]*abs_Ty/sx

        r4 = Tr[4]*abs_Ty
        r5 = Tr[5]*abs_Ty
        r6 = Tr[6]*abs_Ty

        Tx = Tr[3]*abs_Ty


        self.R = np.array([[r1, r2, r3],
                      [r4, r5, r6],
                      [0.0, 0.0, 0.0]])

        self.R[-1, :] = np.cross(self.R[0, :], self.R[1, :])

        if np.allclose(np.sign(np.linalg.det(self.R)), -1):
            self.R[-1, :] *= -1
        if np.abs(np.linalg.det(self.R) - 1) > 1e-2:
            raise ValueError("Rotation Matrix Determinant != 1")

        sign = self.GetSign(abs_Ty, Tx, self.R, TestXYZ, TestUV)

        Ty = sign*abs_Ty
        # sx *= sign

        return Tx, Ty, sx

    def GetSign(self, abs_Ty, Tx, R, testxyz, testuv):

        x, y, _ = R @ testxyz

        Xmatch = (np.sign(x + Tx) == np.sign(testuv[0]))
        Ymatch = (np.sign(y + abs_Ty) == np.sign(testuv[1]))

        if Xmatch == Ymatch:
            return 1
        else:
            return -1


class PinholeCV2(Pinhole):

    def __init__(self, *args):
        super().__init__(*args)

        self.k1 = 0
        self.k2 = 0

        self.p1 = 0
        self.p2 = 0

    def Distortion(self, Xu, Yu):

        R2 = Xu**2 + Yu**2

        X = (1 + self.k1 * R2 + self.k2 * R2**2)*Xu + 2*self.p1*Xu*Yu + self.p2*(R2 + 2*Xu)
        Y = (1 + self.k1 * R2 + self.k2 * R2**2)*Yu + self.p1*(R2 + 2*Yu) + 2*self.p2*Xu*Yu

        return X, Y

    def CalcDistortion(self, Xw, Yw, Zw, u_act, v_act):

        guess = np.array([0.0, 0.0, 0.0, 0.0])

        minimize(self.AdjustDistortion, guess,
                 args=(Xw, Yw, Zw, u_act, v_act), method='Nelder-Mead', tol=1e-8)


    def AdjustDistortion(self, Coeffs, Xw, Yw, Zw, u_act, v_act):

        self.k1 = Coeffs[0]
        self.k2 = Coeffs[1]

        self.p1 = Coeffs[2]
        self.p2 = Coeffs[3]

        return self.RMSE(Xw, Yw, Zw, u_act, v_act)

class PinholeCV2_ThinPrism(Pinhole):

    def __init__(self, *args):
        super().__init__(*args)

        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0

        self.p1 = 0
        self.p2 = 0

        self.s1 = 0
        self.s2 = 0

    def Distortion(self, Xu, Yu):

        R2 = Xu**2 + Yu**2

        X = (1 + self.k1 * R2 + self.k2 * R2**2 + self.k3 * R2**3 + self.k4 * R2**4)*Xu \
            + 2*self.p1*Xu*Yu + self.p2*(R2 + 2*Xu) \
            + self.s1*R2

        Y = (1 + self.k1 * R2 + self.k2 * R2**2 + self.k3 * R2**3 + self.k4 * R2**4)*Yu \
            + self.p1*(R2 + 2*Yu) + 2*self.p2*Xu*Yu \
            + self.s2*R2

        return X, Y

    def CalcDistortion(self, Xw, Yw, Zw, u_act, v_act):

        guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        minimize(self.AdjustDistortion, guess,
                 args=(Xw, Yw, Zw, u_act, v_act), method='Nelder-Mead', tol=1e-6)


    def AdjustDistortion(self, Coeffs, Xw, Yw, Zw, u_act, v_act):

        self.k1 = Coeffs[0]
        self.k2 = Coeffs[1]

        self.k3 = Coeffs[4]
        self.k4 = Coeffs[5]

        self.p1 = Coeffs[2]
        self.p2 = Coeffs[3]

        self.s1 = Coeffs[6]
        self.s2 = Coeffs[7]

        self.s3 = Coeffs[8]
        self.s4 = Coeffs[9]

        return self.RMSE(Xw, Yw, Zw, u_act, v_act)


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

        row_str = ""
        for l in range(self.N_coefficients):
            row_str = row_str + f"+ a_{l}"

            if self.Ci[l] > 0:
                row_str = row_str + f"x^{self.Ci[l]}"
            if self.Cj[l] > 0:
                row_str = row_str + f"y^{self.Cj[l]} "
            if self.Ck[l] > 0:
                row_str = row_str + f"z^{self.Ck[l]}"
            if row_str == "":
                row_str = "1"
        self._rl = row_str

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

        import matplotlib.pyplot as pyp
        import pandas as pd

        df = pd.read_csv("TestPinhole.csv")
        X, Y, Z = df["X"], df["Y"], df["Z"]
        u, v = df["u"], df["v"]

        Cam1 = Pinhole((0, 0), 6.5e-3)

        Cam1.Fit(u, v, X, Y, Z)
        Cam1.k = 0

        u1, v1 = Cam1.Map(X, Y, Z)

        fig, (ax1, ax2) = pyp.subplots(2, figsize=(4,8), sharex=True, sharey=True)

        ax1.scatter(u, v, s=1)
        ax2.scatter(u1, v1, s=1)

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')

        fig.show()

        print(Cam1.Rt)
        print(Cam1.f)
        print(Cam1.sx)





