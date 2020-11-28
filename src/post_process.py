import numpy as np
import cv2 as cv

def find_con(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    high_thresh, thresh_im = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    print(low_thresh, high_thresh)
    thresh = cv.Canny(img, low_thresh, high_thresh)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    ori = cv.imread('../data/001.png')
    img = cv.imread('../result/demo_err_all.png')
    contours = find_con(img)
    cv.drawContours(ori, contours, -1, (255, 0, 0), 2)
    # print(contours)
    cv.imwrite('../result/contour.png', ori)