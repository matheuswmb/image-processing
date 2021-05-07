import cv2 as cv
import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt

Pixel = namedtuple('Pixel', ['x', 'y', 'value'])
BLACK = 0
WHITE = 255
GRAY = 100
WHITOUT_COLOR = -1

def show_and_wait(img_to_show, name="Display window"):
    cv.imshow(name, img_to_show)
    a = cv.waitKey(0)

def plot_histogram(image, channel, pixel_range=None, color='grey'):
    pixel_range = pixel_range or [0,256]

    hist = cv.calcHist([image], channel, None, [256], pixel_range)

    plt.plot(hist, color=color)
    plt.xlim([0,256])
    plt.show()

def pre_processing_image(image, bottom_limit=200):
    gray_image = get_gray_image(image)
    return get_black_white_image(gray_image, bottom_limit)

def get_black_white_image(image, bottom_limit):
    blackAndWhiteImage = image.copy()
    for row_index, row in enumerate(image):
        for column_index, pixel in enumerate(row):
            if bottom_limit < pixel:
                blackAndWhiteImage[row_index][column_index] = 255
            else:
                blackAndWhiteImage[row_index][column_index] = 0

    return blackAndWhiteImage

def get_images():
    first_image = cv.imread('./first.jpg', cv.IMREAD_GRAYSCALE)
    second_image = cv.imread('./second.jpg', cv.IMREAD_GRAYSCALE)
    third_image = cv.imread('./third.jpg', cv.IMREAD_GRAYSCALE)

    return first_image, second_image, third_image

def default_group_index_and_color(pixel_value):
    if pixel_value == 0:
        return 0, BLACK
    elif pixel_value == 255:
        return 1, WHITE
    else:
        return -1, WHITOUT_COLOR

def segment(image, groups_number=None, groups_color=None, get_group_index=default_group_index_and_color):
    padded_image = np.pad(first_image, 1, mode='constant', constant_values=(GRAY))
    groups_number = groups_number or 2
    groups_color = groups_color or [255, 0]
    groups = []

    for group_index in range(groups_number):
        groups.append(np.full(image.shape, groups_color[group_index], 'uint8'))

    initial_pixel = Pixel(0, 0, image[0, 0])
    stack = [initial_pixel]
    next_stacks = []

    while len(stack) != 0:
        pixel = stack.pop()

        pixel_group, pixel_color = get_group_index(pixel.value)
        groups[pixel_group][pixel.y, pixel.x] = pixel_color

        padded_image[pixel.y + 1, pixel.x + 1] = GRAY
        pixel_neighborhood = get_neighborhood_array(padded_image, pixel.x, pixel.y, padded=True)
        row_len, column_len = pixel_neighborhood.shape
        for row_index in range(row_len):
            for column_index in range(column_len):
                neighborhood_pixel_value = pixel_neighborhood[row_index][column_index]
                neighborhood_pixel_group, neighborhood_pixel_color = get_group_index(neighborhood_pixel_value)
                if neighborhood_pixel_group == pixel_group:
                    stack.append(
                        Pixel(pixel.x-1+column_index, pixel.y-1+row_index, neighborhood_pixel_value)
                    )
                elif neighborhood_pixel_color != WHITOUT_COLOR:
                    next_stacks.append(
                        [Pixel(pixel.x-1+column_index, pixel.y-1+row_index, neighborhood_pixel_value)]
                    )

        # print(len(stack), pixel.x, pixel.y)
        if len(stack) == 0 and len(next_stacks) > 0:
            stack = next_stacks.pop()

    return groups

def get_neighborhood_array(array, x, y, neighborhood=1, flatted=False, padded=False):
    if padded:
        x += neighborhood
        y += neighborhood

    min_y = y-neighborhood
    min_x = x-neighborhood

    if min_y < 0:
        min_y = 0

    if min_x < 0:
        min_x = 0

    # numpy slices handle if max range when matrix is not padded
    neighborhood_array = array[min_y: y+neighborhood+1, min_x: x+neighborhood+1]

    if flatted:
        return neighborhood_array.flatten()
    else:
        return neighborhood_array

def erosion(image):
    for line_index in range(linha):
        for column in range(coluna):
            x = column
            y = line_index 

if __name__ == "__main__":
    first_image, second_image, third_image = get_images()

    first_image = get_black_white_image(first_image, 220)
    # second_image = get_black_white_image(second_image, 200)
    # third_image = get_black_white_image(third_image, 100)
    imagens = segment(first_image)
    show_and_wait(imagens[0], "imagem 1")
    show_and_wait(imagens[1], "imagem 2")