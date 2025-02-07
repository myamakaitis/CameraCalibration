import cv2 as cv
import numpy as np
import pandas as pd
from SortDots import PatternGrid, ShowTable
from TemplateMatch import GetMarkLocs, MakeTemplate


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


def CalibrateGrid(ImagesPath, translations, GridSpacing, stack_template=None):

    table = pd.DataFrame(columns=["u", "v", "j", "i", "X", "Y", "Z"])
    labelled_imgs = []

    pattern = PatternGrid(GridSpacing)
    for ImgName, shift in zip(ImagesPath, translations):

        Img = cv.imread(ImgName, -1)
        # ImgPM = Img[:, :, PM_Channel]

        if pattern.Origin[0] is None:
            pattern.CreateRefs(np.copy(Img))

        spacing_px = pattern.AvgSeparation()
        u_marks, v_marks, _ = GetMarkLocs(Img, int(0.7*spacing_px), template=stack_template)

        table_z = MakeTable(u_marks, v_marks, shift)

        pattern.AdjustRefs(table_z)

        j, i, x, y = pattern.PosSort(u_marks, v_marks)
        table_z["j"] = j
        table_z["i"] = i

        table_z["X"] += x
        table_z["Y"] += y

        table_z = table_z.dropna()

        img_labelled = ShowTable(Img, table_z)
        labelled_imgs.append(img_labelled)

        table = table.append(table_z)

    return table, labelled_imgs


if __name__ == "__main__":

    Case = "D"

    # Folder = f"TestImages/Perspective{Case}/"
    # imgs = [Folder + f"cal_{i:03d}_{Case}.tif" for i in range(12)]

    Folder = f"TestImages/split2/"
    imgs = [Folder + f"cal_{i:03d}_{Case}.tif" for i in np.arange(7, 16)]

    z = np.arange(0, 5.0, 0.5)

    print(z)
    Translations = [np.array([0, 0, zi]) for zi in z]

    InFocusTemplate = MakeTemplate(cv.imread(imgs[len(imgs)//2], -1))

    Table, colored_images = CalibrateGrid(imgs, Translations, 1, stack_template=InFocusTemplate)

    # Table.to_csv(f"Marks_{Case}.csv")
    Table.to_csv(f"Marks2.csv")

    # for i in range(12):
    #     cv.imwrite(Folder + f"AfterCal/Cal_z={z[i]}.png", colored_images[i])

    print("done")
