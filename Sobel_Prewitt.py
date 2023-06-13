import cv2
import numpy as np
import matplotlib.pyplot as plt
from auto_threshold import auto_threshold


def convolution(img, mask):
    img_fix = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    img_new = np.zeros([img.shape[0], img.shape[1]])
    for i in range(img_fix.shape[0] - 2):
        for j in range(img_fix.shape[1] - 2):
            img_new[i, j] = np.sum(img_fix[i: i + 3, j: j + 3] * mask)
    return img_new

def threshold(img, threshold):
    m, n = img.shape
    img_new = np.zeros([m, n], np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold:
                img_new[i, j] = 255
            else:
                img_new[i, j] = 0
    return img_new

# Lọc Prewitt
def Prewitt(threshold_img):
    # Bộ lọc Prewitt theo hướng X
    Prewitt_X = np.array(([-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]))
    # Bộ lọc Prewitt theo hướng Y
    Prewitt_Y = np.array(([-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]))

    # Nhân tích chập Prewitt theo hướng X
    imgPrewitt_X = convolution(threshold_img, Prewitt_X)

    # Nhân tích chập Prewitt theo hướng Y
    imgPrewitt_Y = convolution(threshold_img, Prewitt_Y)

    # Ảnh tổng Prewitt theo hướng X và Sobel theo hướng Y
    Prewitt_img = np.sqrt(imgPrewitt_X ** 2) + np.sqrt(imgPrewitt_Y ** 2)
    Prewitt_img = Prewitt_img / np.max(Prewitt_img) * 255
    Prewitt_img = threshold(Prewitt_img, 111)

    return Prewitt_img

def Sobel(threshold_img):
    # Lọc Sobel
    # Bộ lọc Sobel theo hướng X
    Sobel_X = np.array(([-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]))
    # Bộ lọc Sobel theo hướng Y
    Sobel_Y = np.array(([-1, 0, 1],
                        [-2, 0, 2],
                        [1, 0, 1]))

    # Nhân tích chập Sobel theo hướng X
    imgSobel_X = convolution(threshold_img, Sobel_X)

    # Nhân tích chập Sobel theo hướng Y
    imgSobel_Y = convolution(threshold_img, Sobel_Y)

    # Ảnh tổng Sobel theo hướng X và Sobel theo hướng Y
    sobel_img = np.sqrt(imgSobel_X ** 2) + np.sqrt(imgSobel_Y ** 2)
    sobel_img = sobel_img / np.max(sobel_img) * 255
    sobel_img = threshold(sobel_img, 111)

    return sobel_img



if __name__ == "__main__":
    img = cv2.imread("D:\Image_Processing\Xu.png", 0)
    threshold_img = threshold(img, auto_threshold(img))
    sobel_img = Sobel(threshold_img)
    prewitt_img = Prewitt(threshold_img)
    cv2.imshow("Sobel", sobel_img)
    cv2.imshow("Prewitt", prewitt_img)
    cv2.waitKey()
    cv2.destroyAllWindows()