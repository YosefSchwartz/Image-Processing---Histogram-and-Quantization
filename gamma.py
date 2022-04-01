"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np
from ex1_utils import LOAD_RGB, LOAD_GRAY_SCALE


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    im_cv = cv2.imread(img_path)# Load the image
    if im_cv is None:
        raise Exception("Cannot load image!\n")

    if rep < 1 or rep > 2:
        raise Exception("Invalid argument of representation!\n")

    if rep == LOAD_GRAY_SCALE:
        im_cv = cv2.cvtColor(im_cv,cv2.COLOR_BGR2GRAY)

    def on_trackbar(val):  # On change function
        # Avoid from divide zero
        if val == 0:
            img = np.ones(im_cv.shape).dot(255)
        else:
            img = ((im_cv / 255) ** (val / 100) * 255).astype(np.uint8)
        cv2.imshow(title_window, img)

    title_window = 'Gamma Correction'
    cv2.namedWindow(title_window)
    cv2.resizeWindow(title_window, im_cv.shape[1], im_cv.shape[0])
    trackbar_name = 'Gamma x 0.01'
    cv2.createTrackbar(trackbar_name, title_window, 100, 200, on_trackbar)
    cv2.imshow(title_window, im_cv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    gammaDisplay('ZOOM_IN.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
