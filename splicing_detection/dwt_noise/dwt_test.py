'''
@Description: image splicing detection using discrete wavelet transform(DWT) noise
(Paper: Using noise inconsistencies for blind image forensics. Babak Mahdian 2009)
@Author: gtwell
@Date: 2019-08-22 16:05:19
@LastEditTime: 2019-09-10 19:27:49
@LastEditors: Please set LastEditors
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image

def SplitImages(path, maskpath):
    img = cv2.imread(path, 0)
    h, w = img.shape
    # print(h, w)
    if h < 300 and w < 300:
        BLOCK = 8
    elif h > 1000 or w > 1000:
        BLOCK = 24
    else:
        BLOCK = 16
    # mask = cv2.imread(maskpath, 0)
    # img_mask = np.concatenate([img, mask], 1)
    # cv2.imshow("img", img_mask)
    # cv2.waitKey(20)

    coeffs2 = pywt.dwt2(img, 'db8')
    LL, (LH, HL, HH) = coeffs2
    HH = HH[0:int(np.floor(HH.shape[0]/BLOCK)) * BLOCK, 0:int(np.floor(HH.shape[1]/BLOCK)) * BLOCK]

    blockMatrix = []

    for i in range(0, HH.shape[0], BLOCK):
        for j in range(0, HH.shape[1], BLOCK):
            blockWindow = HH[i:i+BLOCK, j:j+BLOCK]
            blockMatrix.append(blockWindow)
    label = [i for i in range(len(blockMatrix))]
    blockMatrix = np.array(blockMatrix)
    label = np.array(label)
    label_idx = list(label.copy())

    dicts = {"blk": blockMatrix, "label": label, "label_id": label_idx, "HH": HH, "img": img, "BLOCK": BLOCK}

    return dicts

def MAD(matrix):
    return np.mean(np.abs(matrix))/0.6745

def MergeBlock(kwargs):
    blockMatrix = kwargs["blk"]
    label       = kwargs["label"]
    label_idx   = kwargs["label_id"]
    HH          = kwargs["HH"]
    img         = kwargs["img"]
    blocksize   = kwargs["BLOCK"]
    heatmap = np.array([MAD(blockMatrix[i]) for i in range(len(blockMatrix))]).reshape(int(HH.shape[0]/blocksize), int(HH.shape[1]/blocksize))

    plt.subplot(1, 2, 1)
    plt.imshow(heatmap, cmap="gray")

    # generated prediction mask method 1
    heatmap = heatmap.flatten()
    # THRESH = np.abs(np.percentile(heatmap[:-1]-heatmap[1:], 25))
    THRESH = 0.15
    print("---------start----------")
    for i in range(len(heatmap) - 1):
        for j in range(i+1, len(heatmap)):
            if np.abs(heatmap[i] - heatmap[j]) < THRESH:
                label[j] = label[i]
    print("----------end----------")
    label = label.reshape(int(HH.shape[0]/blocksize), int(HH.shape[1]/blocksize))

    """
    # generated prediction mask method 2
    # if you want to use method2, please comment method 1
    label = label.reshape(int(HH.shape[0]/blocksize), int(HH.shape[1]/blocksize))
    # THRESH = np.abs(np.percentile(heatmap[:-1]-heatmap[1:], 25))
    THRESH = 1
    for i in range(int(HH.shape[0]/blocksize)):
        for j in range(int(HH.shape[1]/blocksize)):
            if i == 0 and j == 0:
                pass
            elif i == 0 and j != 0:
                if (np.abs(heatmap[i][j] - heatmap[i][j-1]) < THRESH):
                    label[i][j] = label[i][j-1]
            elif j == 0 and i != 0:
                value = min(np.abs(heatmap[i][j] - heatmap[i-1][j]), np.abs(heatmap[i][j] - heatmap[i-1][j+1]))
                if (value < THRESH):
                    if (np.abs(heatmap[i][j] - heatmap[i-1][j]) < np.abs(heatmap[i][j] - heatmap[i-1][j+1])):
                        label[i][j] = label[i-1][j]
                    else:
                        label[i][j] = label[i-1][j+1]
            elif j == int(HH.shape[1]/blocksize)-1:
                value = min(np.abs(heatmap[i][j] - heatmap[i-1][j]), np.abs(heatmap[i][j] - heatmap[i][j-1]), np.abs(heatmap[i][j] - heatmap[i-1][j-1]))
                if (value < THRESH):
                    if (np.abs(heatmap[i][j] - heatmap[i-1][j-1]) == value):
                        label[i][j] = label[i-1][j-1]
                    elif (np.abs(heatmap[i][j] - heatmap[i-1][j]) == value):
                        label[i][j] = label[i-1][j]
                    elif (np.abs(heatmap[i][j] - heatmap[i-1][j-1]) == value):
                        label[i][j] = label[i][j-1]
            else:
                value = min(np.abs(heatmap[i][j] - heatmap[i-1][j]), np.abs(heatmap[i][j] - heatmap[i-1][j+1]), \
                    np.abs(heatmap[i][j] - heatmap[i][j-1]), np.abs(heatmap[i][j] - heatmap[i-1][j-1]))
                if (value < THRESH):
                    if (np.abs(heatmap[i][j] - heatmap[i-1][j-1]) == value):
                        label[i][j] = label[i-1][j-1]
                    elif (np.abs(heatmap[i][j] - heatmap[i-1][j]) == value):
                        label[i][j] = label[i-1][j]
                    elif (np.abs(heatmap[i][j] - heatmap[i-1][j+1]) == value):
                        label[i][j] = label[i-1][j+1]
                    elif (np.abs(heatmap[i][j] - heatmap[i][j-1]) == value):
                        label[i][j] = label[i][j-1]
    """

    repeatLabelCounts = {i: np.sum(label == i) for i in label_idx}
    maxNum = 0
    maxNumIndex = 0
    for key, value in repeatLabelCounts.items():
        if value > maxNum:
            maxNum = value
            maxNumIndex = key

    for i in range(int(HH.shape[0]/blocksize)):
        for j in range(int(HH.shape[1]/blocksize)):
            if label[i][j] == maxNumIndex or repeatLabelCounts[label[i][j]] < 3:
                label[i][j] = 0
            else:
                label[i][j] = 255

    label = label.astype(np.uint8)
    label = cv2.resize(label, (img.shape[1], img.shape[0]))
    label[label<128] = 0
    label[label>=128] = 255
    cv2.imwrite("prediction.jpg", label)
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap="gray")
    plt.show()

    return label

if __name__ == "__main__":
    import glob
    # path = glob.glob("F:/dataset/forgery_dataset/Columbia Uncompressed Image Splicing Detection/4cam_splc/*")
    # path = glob.glob("./sp/*")
    # mask = glob.glob("./mask/*")
    path = glob.glob("./sp0.tif")
    for i in range(len(path)):
        # if no groundtruth, the second parameter defined None and commented imread maskpath in the function.
        dicts = SplitImages(path[i], None)
        label = MergeBlock(dicts)