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


def CalibrateGrid(ImagesPath, translations, GridSpacing, stack_template=None, mask=None):

    table = pd.DataFrame()
    table.dropna()
    labelled_imgs = []

    pattern = PatternGrid(GridSpacing)
    for ImgName, shift in zip(ImagesPath, translations):

        Img = cv.imread(ImgName, -1)
        # ImgPM = Img[:, :, PM_Channel]

        if mask is not None:
            Img *= mask
  
        if pattern.Origin[0] is None:
            pattern.CreateRefs(np.copy(Img))

        spacing_px = pattern.AvgSeparation()
        u_marks, v_marks, _ = GetMarkLocs(Img, int(0.8*spacing_px), template=stack_template, MinContrast=45)

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

        table = pd.concat([table, table_z], axis=0)

    return table, labelled_imgs


if __name__ == "__main__":

    # Case = "A"

    # Folder = f"TestCases/Perspective{Case}/"
    # imgs = [Folder + f"cal_{i:03d}_{Case}.tif" for i in range(12)]

    

    img_folder = "InvertedMicroCal/split/"

    
    img_prefix = "img"
    img_postfix = ""
    img_ext = ".tif"
    label = "A"

    # z = np.arange(0, 12)
    
    z = np.arange(0, 25, 2.5)
    n_array = np.arange(3, 13)
    n_array = np.arange(6, 11)

    imgs = [f"{img_folder}{label}_{img_prefix}{n:03d}{img_postfix}{img_ext}" for n in n_array]

    print(imgs[0])

    # mask = cv.imread(Folder + f"Masks/mask{n}.png", -1)

    Translations = [np.array([0, 0, zi]) for zi in z]

    InFocusTemplate = MakeTemplate(cv.imread(imgs[3], -1))

    Table, colored_images = CalibrateGrid(imgs, Translations, 10, stack_template=InFocusTemplate)

    Table.to_csv(img_folder + f"{label}_Marks.csv")

    for i in range(len(imgs)):
        cv.imwrite(img_folder + f"AfterCal/{label}_Cal_z={z[i]}.png", colored_images[i])

    print("done")

