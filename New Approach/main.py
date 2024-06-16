import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("E:\Sid Folder\Random Python Scripts\Farm Land Detection\Farmland2.png")

    # GrayScale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert
    thresh = cv2.bitwise_not(thresh)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    cv2.imwrite("E:\\Sid Folder\\Random Python Scripts\\Farm Land Detection\\Gray Scale\\thresh2.png", thresh)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
