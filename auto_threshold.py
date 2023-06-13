import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_threshold(img):
    m, n = img.shape
    img_arr = np.array(img).flatten()

    # Sử dụng hàm unique() để lấy ra các giá trị điểm ảnh duy nhất trong mảng
    img_count = np.unique(img_arr)

    # Sử dụng hàm sort() của NumPy để sắp xếp các giá trị điểm ảnh trong mảng theo thứ tự tăng dần
    g = np.sort(img_count).astype(float)

    # Số lần xuất hiện của mỗi giá trị trong mảng
    hg = np.zeros(len(g)).astype(float)
    for i in range(len(hg)):
        hg[i] = np.count_nonzero(img_arr == g[i])

    # Tong xich ma so lan xuat hien ( hg )
    tg = np.cumsum(hg).astype(float)

    # g * hg
    g_hg = np.zeros(len(g)).astype(float)
    for i in range(len(g_hg)):
        g_hg[i] = g[i] * hg[i]

    # Tổng xích ma của g_hg
    xichma_g_hg = np.cumsum(g_hg).astype(float)

    # mg = (1 / tg) * xichma_g_hg
    mg = np.zeros(len(g)).astype(float)
    for i in range(len(mg)):
        mg[i] = (1 / tg[i]) * xichma_g_hg[i]

    # fg = (tg / (m * n - tg)) * ((mg - max(mg)) ** 2)
    fg = np.zeros(len(g) - 1).astype(float)
    for i in range(len(fg)):
        fg[i] = (tg[i] / (m * n - tg[i])) * (((mg[i] - mg[len(mg) - 1]) ** 2))
    return g[np.where(fg == max(fg))]


