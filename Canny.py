import numpy as np
import cv2


def convolution(img, mask):
    pad_img = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    new_img = np.zeros([pad_img.shape[0], pad_img.shape[1]])
    for i in range(pad_img.shape[0] - 2):
        for j in range(pad_img.shape[1] - 2):
            new_img[i + 1, j + 1] = np.sum(pad_img[i: i + 3, j: j + 3] * mask)

    return new_img


def sobel_filter(blur_img):
    x = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    y = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])

    sobel_x = convolution(blur_img, x)
    sobel_y = convolution(blur_img, y)
    sobel_img = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    theta = np.arctan2(sobel_x, sobel_y)

    return sobel_img, theta


def non_maximum_suppression(sobel_img, theta):
    m, n = sobel_img.shape
    non_maximum_suppression_img = np.zeros_like(sobel_img)

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            direction = theta[i, j]
            # Góc 0, 180 độ
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                prev_pixel = sobel_img[i, j - 1]
                next_pixel = sobel_img[i, j + 1]
            # Góc 45 độ
            elif (22.5 <= direction < 67.5):
                prev_pixel = sobel_img[i + 1, j - 1]
                next_pixel = sobel_img[i - 1, j + 1]
            # Góc 90 độ
            elif (67.5 <= direction < 112.5):
                prev_pixel = sobel_img[i - 1, j]
                next_pixel = sobel_img[i + 1, j]
            # Góc 135 độ
            else: # 112.5 <= direction < 157.5
                prev_pixel = sobel_img[i - 1, j - 1]
                next_pixel = sobel_img[i + 1, j + 1]

            current_pixel = sobel_img[i, j]
            if (current_pixel >= prev_pixel) and (current_pixel >= next_pixel):
                non_maximum_suppression_img[i, j] = current_pixel

    return non_maximum_suppression_img


def double_threshold(non_maximum_suppression_img, low_threshold_ratio, high_threshold_ratio):
    high_threshold = non_maximum_suppression_img.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    double_threshold_img = np.zeros_like(non_maximum_suppression_img, dtype=np.uint8)
    weak_i = []
    weak_j = []
    for i in range(non_maximum_suppression_img.shape[0]):
        for j in range(non_maximum_suppression_img.shape[1]):
            if non_maximum_suppression_img[i, j] >= high_threshold:
                double_threshold_img[i, j] = 255
            elif non_maximum_suppression_img[i, j] < high_threshold and non_maximum_suppression_img[i, j] >= low_threshold:
                double_threshold_img[i, j] = non_maximum_suppression_img[i, j]
                weak_i.append(i)
                weak_j.append(j)

    return double_threshold_img, weak_i, weak_j



def edge_tracking(threshold_img, weak_i, weak_j):
    edge_tracking_img = np.copy(threshold_img)
    for i, j in zip(weak_i, weak_j):
        region = threshold_img[i - 1:i + 2, j - 1:j + 2]
        if np.any(region == 255):
            edge_tracking_img[i, j] = 255
        else:
            edge_tracking_img[i, j] = 0

    return edge_tracking_img



img = cv2.imread("D:\Image_Processing\canny.png", 0)
# Bước 1: Làm mờ ảnh
blur_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
# Bước 2: Tính Gradient độ xám
sobel_filter_img, theta = sobel_filter(blur_img)
# Bước 3: Thực hiện thuật toán Non-maximum suppression để làm mỏng biên
non_maximum_suppression_img = non_maximum_suppression(sobel_filter_img, theta)
# Bước 4: Ngưỡng kép tùy vào từng ảnh mà chọn tỉ lệ ngưỡng cao, thấp
double_threshold_img, weak_i, weak_j = double_threshold(non_maximum_suppression_img, 0.03, 0.09)
# Bước 5: Theo dõi cạnh theo độ trễ (Edge Tracking by Hysteresis)
edge_tracking_img = edge_tracking(double_threshold_img, weak_i, weak_j)


edge_Canny = cv2.Canny(blur_img, 100, 200)
cv2.imshow("Code chay", edge_tracking_img)
cv2.imshow("Canny", edge_Canny)
cv2.waitKey()
cv2.destroyAllWindows()