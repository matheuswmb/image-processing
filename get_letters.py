import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

def show_and_wait(img_to_show, name="Display window"):
    cv.imshow(name, img_to_show)
    a = cv.waitKey(0)

def plot_histogram(image, channel, pixel_range=None, color='grey'):
    pixel_range = pixel_range or [0,256]

    hist = cv.calcHist([image], channel, None, [256], pixel_range)

    plt.plot(hist, color=color)
    plt.xlim([0,256])
    plt.show()

def pre_processin_image(image, bottom_limit=200, upper_limit=255):
    gray_image = get_gray_image(image)
    return get_black_white_image(gray_image, bottom_limit, upper_limit)

def get_gray_image(imagem):
    return cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

def get_black_white_image(image, bottom_limit):
    blackAndWhiteImage = image.copy()
    for line_index, line in enumerate(image):
        for collum_index, pixel in enumerate(line):
            if bottom_limit < pixel:
                blackAndWhiteImage[line_index][collum_index] = 255
            else:
                blackAndWhiteImage[line_index][collum_index] = 0

    return blackAndWhiteImage

def get_images():
    fist_image = cv.imread('./first.jpg')
    second_image = cv.imread('./second.jpg')
    third_image = cv.imread('./third.jpg')

    return fist_image, second_image, third_image

def segment():
    pass

def get_neighborhood_array(array, x, y, neighborhood=1, flatted=False):
    min_y = y-neighborhood
    if min_y < 0:
        min_y = 0

    min_x = x-neighborhood
    if min_x < 0:
        min_x = 0

    neighborhood_array = array[min_y: y+neighborhood+1, min_x: x+neighborhood+1]

    if flatted:
        return neighborhood_array.flatten()
    else:
        return neighborhood_array

if __name__ == "__main__":
    fist_image, second_image, third_image = get_images()

    fist_image = get_gray_image(fist_image)
    second_image = get_gray_image(second_image)
    third_image = get_gray_image(third_image)

    # show_and_wait(get_black_white_image(fist_image, 220), 'primeira')
    # show_and_wait(cv.threshold(fist_image, 220, 255, cv.THRESH_BINARY)[1], 'primeira opencv')

    # show_and_wait(get_black_white_image(second_image, 200), 'segunda')
    # show_and_wait(cv.threshold(second_image, 200, 255, cv.THRESH_BINARY)[1], 'segunda opencv')

    # show_and_wait(get_black_white_image(third_image, 100),  'terceira')
    # show_and_wait(cv.threshold(third_image, 100, 255, cv.THRESH_BINARY)[1], 'terceira opencv')

    a = np.array([[7,20,32,40,5,6],[1,2,3,4,5,6],[3,30,300,3000,30000,300000],[1,2,3,4,5,6],[1,20,3,4,5,60], [1,2,3,4,5,6]])
    print(get_neighborhood_array(a, 5, 2))
