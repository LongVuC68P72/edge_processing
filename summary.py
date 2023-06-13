from Canny import sobel_filter, non_maximum_suppression, double_threshold, edge_tracking
from Sobel_Prewitt import threshold, Prewitt, Sobel
from auto_threshold import auto_threshold
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img = cv2.imread("D:\Image_Processing\\test1.png", 0)
    blur_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1, sigmaY=1)
    sobel_filter_img, theta = sobel_filter(blur_img)
    non_maximum_suppression_img = non_maximum_suppression(sobel_filter_img, theta)
    double_threshold_img, weak_i, weak_j = double_threshold(non_maximum_suppression_img, 0.03, 0.07)
    edge_tracking_img = edge_tracking(double_threshold_img, weak_i, weak_j)

    threshold_img = threshold(img, auto_threshold(img))
    sobel_img = Sobel(threshold_img)
    prewitt_img = Prewitt(threshold_img)

    plt.subplot(2, 3, 2)
    plt.title("Origin")
    plt.imshow(img, cmap="gray")

    plt.subplot(2, 3, 4)
    plt.title("Prewitt")
    plt.imshow(Sobel(prewitt_img), cmap="gray")

    plt.subplot(2, 3, 5)
    plt.title("Sobel")
    plt.imshow(Sobel(sobel_img), cmap="gray")

    plt.subplot(2, 3, 6)
    plt.title("Canny")
    plt.imshow(Sobel(edge_tracking_img), cmap="gray")
    plt.tight_layout()
    plt.show()
