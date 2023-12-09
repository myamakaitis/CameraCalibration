import cv2 as cv
import numpy as np
import pandas as pd
from SortDots import PatternGrid, ShowTable
from TemplateMatch import GetMarkLocs


def MakeTable(u_marks, v_marks, translation):

    size = len(u_marks)
    table_vals = np.zeros((size, 7))

    mark_table = pd.DataFrame(data=table_vals, columns=["u", "v", "j", "i", "X", "Y", "Z"])
    mark_table["u"] = u_marks
    mark_table["v"] = v_marks

    mark_table["X"] = translation[0]
    mark_table["Y"] = translation[1]
    mark_table["Z"] = translation[2]

    return mark_table


def CalibrateGrid(ImagesPath, translations, GridSpacing, PM_Channel=0):

    for ImgName, shift in zip(ImagesPath, translations):

        Img = cv.imread(ImgName)
        ImgPM = Img[:, :, PM_Channel]

        pattern = PatternGrid(GridSpacing)
        pattern.CreateRefs(np.copy(Img))

        spacing_px = pattern.AvgSeparation()
        u_marks, v_marks, _ = GetMarkLocs(ImgPM, int(0.7*spacing_px))

        table = MakeTable(u_marks, v_marks, shift)

        j, i, x, y = pattern.PosSort(u_marks, v_marks)
        table["j"] = j
        table["i"] = i

        table["X"] += x
        table["Y"] += y

        table = table.dropna()

        ShowTable(Img, table)


if __name__ == "__main__":

    imgs = ["Good.png", "Perspective.png", "DistortedVignetting.png"]

    z = (0, 0, 0)
    translations = [np.array([0, 0, zi]) for zi in z]

    CalibrateGrid(imgs, translations, 1.5)

    print("done")













