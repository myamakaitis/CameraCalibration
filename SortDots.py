import numpy as np
import cv2 as cv
import matplotlib.pyplot as pyp


refPt = []


def click(event, u, v, flags, param):
    # grab references to the global variables
    global refPt
    if event == cv.EVENT_LBUTTONDOWN:
        refPt.append(np.array([u, v]))

    if len(refPt) == 3:
        cv.destroyAllWindows()


def ShowTable(img, table):

    n_rows, n_cols = table.values.shape

    img_8bit = (img.astype(np.float32)*(255/img.max())).astype(np.uint8)

    img_8bit_c = np.zeros((*img_8bit.shape, 3), dtype=np.uint8)
    for i in range(3):
        img_8bit_c[:, :, i] = img_8bit

    color_value = 255

    for l in range(0, n_rows):

        u1, v1 = table.iloc[l, 0], table.iloc[l, 1]
        u1, v1 = int(u1), int(v1)

        i1, j1 = table.iloc[l, 2], table.iloc[l, 3]

        cv.circle(img_8bit_c, (int(u1), int(v1)), 17, (0, color_value, 0))

        if i1 == 0 and j1 == 0:
            cv.circle(img_8bit_c, (int(u1), int(v1)), 17, (color_value, 0, color_value))

        for m in range(l, n_rows):
            u2, v2 = table.iloc[m, 0], table.iloc[m, 1]
            u2, v2 = int(u2), int(v2)

            i2, j2 = table.iloc[m, 2], table.iloc[m, 3]

            if np.abs(i2 - i1) == 2 and j2 == j1:
                cv.line(img_8bit_c, (u1, v1), (u2, v2), (color_value, 0, 0))
            elif np.abs(j2 - j1) == 2 and i2 == i1:
                cv.line(img_8bit_c, (u1, v1), (u2, v2), (0, 0, color_value))

    cv.imshow("Extracted Locations", img_8bit_c)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_8bit_c


class PatternGrid:
    def __init__(self, WorldSpacing,
                 u0=None, v0=None,
                 d_px_x=None, d_px_y=None):

        self.dWorld = WorldSpacing / 2

        self.Origin = np.array([u0, v0])

        self.d_px_x = d_px_x
        self.d_px_y = d_px_y

        if d_px_x is not None and d_px_y is not None:
            self.CalcRefVecMag()

    def CalcRefVecMag(self):
        self.x_mag_px = self.d_px_x.dot(self.d_px_x)
        self.y_mag_px = self.d_px_y.dot(self.d_px_y)

    def SortMark(self, u, v):

        du = u - self.Origin[0]
        dv = v - self.Origin[1]
        duv = np.array([du, dv]).T

        row = 2*duv.dot(self.d_px_y) / self.y_mag_px
        col = 2*duv.dot(self.d_px_x) / self.x_mag_px

        return np.around(row).astype(np.int32), np.around(col).astype(np.int32)

    def WorldPos(self, row, col):

        return row*self.dWorld, col*self.dWorld

    def PosSort(self, u, v):

        row, col = self.SortMark(u, v)

        row, col = self.FixRowCol(row, col)
        row, col = self.FixRowCol(row, col)

        X, Y = self.WorldPos(row, col)

        # self.RemoveNonAdj(row, col, X, Y)

        invalid = ((row % 2 == 1) + (col % 2 == 1))
        Y[invalid], X[invalid] = np.nan, np.nan

        return row, col, X, Y

    def RemoveNonAdj(self, row, col, x, y):

        valid = np.full(len(row), True)

        for l in range(0, len(row)):
            adj = 0

            i1, j1 = col[l], row[l]

            for m in range(l, len(col)):

                i2, j2 = col[m], row[m]

                if np.abs(i2 - i1) == 2 and j2 == j1:
                    adj += 1
                elif np.abs(j2 - j1) == 2 and i2 == i1:
                    adj += 1
        if adj > 4 or adj < 1:
            y[l] = np.nan
            x[l] = np.nan


    def FixRowCol(self, row, col):

        for j in range(int(np.max(row) / 2) + 2):
            RowOdd_jm1 = (row == 2 * j)
            RowOdd_j = (row == 2 * j + 1)
            for i in range(int(np.min(col)), int(np.max(col))):

                Coli = (col == i)
                row[RowOdd_j*Coli] += -1 + 2*np.any(RowOdd_jm1 * Coli)

            RowOdd_jm1 = (row == -2 * j)
            RowOdd_j = (row == -2 * j - 1)
            for i in range(int(np.min(col)), int(np.max(col))):
                Coli = (col == i)
                row[RowOdd_j * Coli] += 1 - 2 * np.any(RowOdd_jm1 * Coli)

        for i in range(int(np.max(col) / 2) + 2):
            ColOdd_im1 = (col == 2 * i)
            ColOdd_i = (col == 2 * i + 1)
            for j in range(int(np.min(row)), int(np.max(row))):
                Rowj = (row == j)
                row[ColOdd_i * Rowj] += -1 + 2 * np.any(ColOdd_im1 * Rowj)

            ColOdd_im1 = (col == -2 * i)
            ColOdd_i = (col == -2 * i - 1)
            for j in range(int(np.min(row)), int(np.max(row))):
                Rowj = (row == j)
                row[ColOdd_i * Rowj] += 1 - 2 * np.any(ColOdd_im1 * Rowj)

        return row, col


    def CreateRefs(self, IMG):

        global refPt
        refPt = []

        cv.namedWindow("ReferenceImage")
        cv.setMouseCallback("ReferenceImage", click)
        cv.imshow("ReferenceImage", IMG)
        cv.waitKey(0)

        self.Origin = refPt[0]
        self.d_px_x = refPt[1] - self.Origin
        self.d_px_y = refPt[2] - self.Origin

        self.CalcRefVecMag()

    def AdjustRefs(self, table):

        global refPt
        refPt = []

        u, v = table['u'].values, table['v'].values

        origin_index = ((u - self.Origin[0])**2 + (v - self.Origin[1])**2).argmin()

        self.Origin = np.array([u[origin_index], v[origin_index]])

        du = u - self.Origin[0]
        dv = v - self.Origin[1]
        duv = np.array([du, dv]).T

        row = duv.dot(self.d_px_y) / self.y_mag_px
        col = duv.dot(self.d_px_x) / self.x_mag_px

        col_i = ((row)**2 + (col - 1)**2).argmin()
        row_i = ((row - 1)**2 + (col)**2).argmin()

        self.d_px_x = np.array([u[col_i], v[col_i]]) - self.Origin
        self.d_px_y = np.array([u[row_i], v[row_i]]) - self.Origin

        self.CalcRefVecMag()

    def AvgSeparation(self):

        return 0.5 * (np.sqrt(self.x_mag_px) + np.sqrt(self.y_mag_px))


if __name__ == "__main__":

    img = cv.imread("TestImages/Good.png")

    pg = PatternGrid(3.0)

    pg.CreateRefs(img)

    for (u, v) in refPt:
        cv.circle(img, (u, v), 17, (255, 0, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

