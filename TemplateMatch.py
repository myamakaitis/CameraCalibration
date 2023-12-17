import numpy as np
import cv2 as cv
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as pyp

rmPt = []


def click_rm(event, u, v, flags, param):
    # grab references to the global variables
    global rmPt
    if event == cv.EVENT_LBUTTONDOWN:
        rmPt.append(np.array([u, v]))


def MakeTemplate(img):

    imgd = img-img.min()

    imgd = (imgd.astype(np.float32)*255/imgd.max()).astype(np.uint8)

    r = cv.selectROI("Select Template", imgd)

    cv.destroyWindow("Select Template")

    template = img[int(r[1]):int(r[1] + r[3]),
                   int(r[0]):int(r[0] + r[2])]

    return template


def GetMarkLocs(img, MinSeparation, template=None, AvgWindow=9, MinContrast=2.1):

    if template is None:
        template = MakeTemplate(img)

    peaks, _, _ = DotPeaks(img, template)
    local_peaks = MaxFind(peaks, MinSeparation, MinContrast)
    peak_indices = np.where(local_peaks)

    u_int = peak_indices[1]
    v_int = peak_indices[0]

    bad_points = ShowCircs(img, u_int, v_int)
    u_int, v_int = u_int[bad_points], v_int[bad_points]

    u, v = SubPixelPeaks(peaks, u_int, v_int, MinSeparation//3)

    return u, v, template


def ShowCircs(img_gs, u, v):
    global rmPt
    rmPt = []

    color_value = np.iinfo(img_gs.dtype).max

    g = np.zeros((*img_gs.shape, 3), dtype=np.uint16)
    for i in range(3):
        g[:, :, i] = img_gs

    for k in range(len(u)):
        g[v[k] - 1:v[k] + 2, u[k] - 1:u[k] + 2, :] = (color_value, color_value, color_value)
        cv.circle(g, (u[k], v[k]), 17, (color_value, color_value, color_value), 3)

    cv.namedWindow("Found Marks")
    cv.setMouseCallback("Found Marks", click_rm)

    cv.imshow("Found Marks", g)
    cv.waitKey(0)
    cv.destroyWindow("Found Marks")

    bad_points = np.full(len(u), True)

    for pt in rmPt:
        diff = ((u-pt[0])**2 + (v-pt[1])**2)

        bad_points[diff == diff.min()] = False

    return bad_points


def CovarianceTemplate(img, template):

    img = img.astype(np.float64)
    template = template.astype(np.float64)

    n_rows, n_cols = img.shape[0], img.shape[1]

    template -= np.mean(template)

    row_pad = n_rows - template.shape[0]
    col_pad = n_cols - template.shape[1]

    row_pad_half = row_pad // 2
    col_pad_half = col_pad // 2

    # pad the template to the image size
    if template.shape != img.shape:

        template = np.pad(template,
                              ((row_pad_half, row_pad_half + row_pad % 2),
                               (col_pad_half, col_pad_half + col_pad % 2)),
                              mode='constant')

    square_window = np.copy(template)
    square_window[square_window > 0] = 1
    # Correlation

    # Calculate local average over a square window
    EX = np.real(fftshift(ifft2(fft2(img) * np.conjugate(fft2(square_window)))))


    #Subtract Local Average
    img -= EX

    #Compute Covariance
    C = np.real(fftshift(ifft2(fft2(img) * np.conjugate(fft2(template)))))

    return C


def CorrelateTemplate(img, template):

    img = img.astype(np.float64)
    template = template.astype(np.float64)

    n_rows, n_cols = img.shape[0], img.shape[1]

    row_pad = n_rows - template.shape[0]
    col_pad = n_cols - template.shape[1]

    row_pad_half = row_pad // 2
    col_pad_half = col_pad // 2

    # pad the template to the image size
    if template.shape != img.shape:

        template = np.pad(template,
                              ((row_pad_half, row_pad_half + row_pad % 2),
                               (col_pad_half, col_pad_half + col_pad % 2)),
                              mode='constant')

    # Correlation
    #EX = np.real(fftshift(ifft2(fft2(Img) * np.conjugate(fft2(square_window)))))
    # Img -= EX
    C = np.real(fftshift(ifft2(fft2(img) * np.conjugate(fft2(template)))))

    #EX2 = np.real(fftshift(ifft2(fft2(Img**2) * np.conjugate(fft2(square_window)))))

    #template_std = np.sqrt(np.sum((Template - np.mean(Template))**2))

    #windowstd = np.sqrt(np.abs(EX2 - EX**2))

    return C / np.max(C) #(template_std*windowstd+1e-5)


def DotPeaks(img, template):

    correlation = CovarianceTemplate(img, template)

    # imgFH = ForstnerHarris(img, 16)
    # templateFH = ForstnerHarris(template, 16)

    imgFH = np.abs(cv.cornerHarris(img.astype(np.float32), 2, 7, 0.05))
    templateFH = np.abs(cv.cornerHarris(template.astype(np.float32), 2, 7, 0.05))

    # templateFH = templateFH - templateFH.min()
    # templateFH /= templateFH.max()
    # imgFH /= imgFH.max()

    correlationFH = CorrelateTemplate(imgFH, templateFH)

    # combined = correlation[window_size//2:-window_size//2, window_size//2:-window_size//2] * correlationFH
    combined = correlation*correlationFH

    return combined, correlation, correlationFH


def MaxFind(img, search_size, minthresh = 2.5):

    PeakLocs = np.zeros(img.shape, dtype=np.bool)

    mask = (img > minthresh*img.mean()).astype(np.bool)

    hss = search_size//2

    for jj in range(hss, img.shape[0] - hss):
        for ii in range(hss, img.shape[1] - hss):
            PeakLocs[jj, ii] = img[jj, ii] >= img[jj-hss:jj+hss, ii-hss:ii+hss].max()

    PeakLocs *= mask

    return PeakLocs


def SubPixelPeaks(PeakImg, u_int, v_int, Window):
    # Find the subpixel peak location using the centroid

    hw = Window//2

    i = np.arange(0, PeakImg.shape[1])
    j = np.arange(0, PeakImg.shape[0])

    ii, jj = np.meshgrid(i, j)

    iiPeakImg = ii*PeakImg
    jjPeakImg = jj*PeakImg

    u_subpixel = np.empty(len(u_int), dtype=np.float32)
    v_subpixel = np.empty(len(v_int), dtype=np.float32)

    for k, (u, v) in enumerate(zip(u_int, v_int)):
        W = np.sum(PeakImg[v-hw: v+hw+1, u-hw: u+hw+1])

        u_subpixel[k] = np.sum(iiPeakImg[v-hw: v+hw+1, u-hw: u+hw+1])/W
        v_subpixel[k] = np.sum(jjPeakImg[v-hw: v+hw+1, u-hw: u+hw+1])/W

    return u_subpixel, v_subpixel


if __name__ == "__main__":
    from cv2 import imread

    Img = imread("TestCases/Perspective.png")[:, :, 0]
    Template = imread("TestCases/dot_template.png")[:, :, 0]

    Peaks, c, cFH = DotPeaks(Img, Template)

    PeakLocs = MaxFind(Peaks, 64, 5)



    fig, ax = pyp.subplots(1, 5, figsize = (5*3, 4), dpi=300)
    ax[0].imshow(Img, cmap='Greys_r')
    ax[0].set_title("\nOriginal Image")
    ax[1].imshow(c)
    ax[1].set_title("\nTemplate Covariance")
    ax[2].imshow(cFH)
    ax[2].set_title("\nHarris CD Covariance")
    ax[3].imshow(Peaks)
    ax[3].set_title("\nEstimates Multiplied")
    ax[4].imshow(cv.dilate(PeakLocs.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)))
    ax[4].set_title("\nRecovered Peaks")

    for axis in ax:
        axis.axis('off')

    fig.tight_layout()
    fig.show()

    # GetMarkLocs(Img, 64)

    #

    # ImgCD = ForstnerHarris(Img, 16)
    TemplateCD = cv.cornerHarris(Template.astype(np.float32), 2, 5, 0.05)
    fig, ax = pyp.subplots()
    ax.imshow(TemplateCD)
    fig.show()
    #
    # TemplateCD = TemplateCD - TemplateCD.min()
    # TemplateCD /= TemplateCD.max()
    #
    # Correlation = CovarianceTemplate(Img, template)
    #
    # import matplotlib.pyplot as pyp
    # fig, ax = pyp.subplots()
    # ax.imshow(Correlation)
    # ax.set_aspect('equal')
    # fig.show()
    #
    # print(Correlation.max())
    #
    # fig, ax = pyp.subplots(2)
    # ax[0].imshow(ImgCD)
    # ax[1].imshow(TemplateCD)
    # fig.show()
    #
    # CorrelationCD = CorrelateTemplate(ImgCD, TemplateCD)
    #
    # import matplotlib.pyplot as pyp
    # fig, ax = pyp.subplots()
    # ax.imshow(CorrelationCD)
    # ax.set_aspect('equal')
    # fig.show()
    #
    # print(TemplateCD.min())
    #
    # fig, ax = pyp.subplots(2, 2, dpi=180, figsize = (6, 8))
    # fig.tight_layout()
    #
    # ax[0, 0].axis('off')
    # ax[0, 0].set_title("Calibration Image")
    # ax[0, 0].imshow(Img, cmap='Greys_r')
    #
    # ax[0, 1].axis('off')
    # ax[0, 1].set_title("Cal Image\nCorrelation Template Matched")
    # ax[0, 1].imshow(Correlation)
    #
    # ax[1, 0].axis('off')
    # ax[1, 0].set_title("Calibration Image\nHarris Corner Detected")
    # ax[1, 0].imshow(ImgCD)
    #
    # ax[1, 1].axis('off')
    # ax[1, 1].set_title("Corner Detected\nTemplate Matched")
    # ax[1, 1].imshow(CorrelationCD)
    #
    # fig.show()
    #
    # fig, ax = pyp.subplots()
    #
    # ax.set_title("Combined")
    #
    # Combined = Correlation
    # Correlation[:-16, :-16]*=CorrelationCD
    #
    # ax.imshow(Combined[:-16, :-16])
    # fig.show()
