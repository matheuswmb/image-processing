import math
import cv2 as cv
import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt

Pixel = namedtuple('Pixel', ['x', 'y', 'value'])
RGB_RED = (255, 0, 0)
BLACK = 0
WHITE = 255
# 107 is prime, so the rule in get_next_color "never" get this value.
INVALID = 107
WHITOUT_COLOR = -1

HIT = 2
FIT = 1
HIT_RULE = 1


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
                blackAndWhiteImage[row_index][column_index] = WHITE
            else:
                blackAndWhiteImage[row_index][column_index] = BLACK

    return blackAndWhiteImage

def get_images():
    first_image = cv.imread('./first.jpg', cv.IMREAD_GRAYSCALE)
    second_image = cv.imread('./second.jpg', cv.IMREAD_GRAYSCALE)
    third_image = cv.imread('./third.jpg', cv.IMREAD_GRAYSCALE)

    return first_image, second_image, third_image

def get_next_color(init_value=None, step=50):
    init_value = init_value or WHITE

    if step == None:
        while True:
            yield init_value
            if init_value == WHITE:
                init_value = BLACK
    else:
        while init_value > 0 + step:
            yield init_value
            init_value -= step

def create_group(shape, fill_value):
    return np.full(shape, fill_value, 'uint8')

def segment(image):
    padded_image = np.pad(image, 1, mode='constant', constant_values=(INVALID))
    get_color = get_next_color(step=None)

    group_color = None
    groups = []
    groups_rect = []
    group_value = None

    initial_pixel = Pixel(0, 0, image[0, 0])
    stack = [initial_pixel]
    added_points = {}
    next_inits = []

    while len(stack) != 0:
        pixel = stack.pop()

        if group_value is None:
            group_value = pixel.value
            group_color = next(get_color)
            if group_color == WHITE:
                groups.append(create_group(image.shape, BLACK))
            else:
                groups.append(create_group(image.shape, WHITE))

            groups_rect.append((pixel.x, pixel.y, pixel.x, pixel.y))

        groups[-1][pixel.y, pixel.x] = group_color

        react = groups_rect[-1]
        groups_rect[-1] = update_react(react, pixel)
        padded_image[pixel.y + 1, pixel.x + 1] = INVALID

        pixel_neighborhood = get_neighborhood_array(padded_image, pixel.x, pixel.y, padded=True)
        row_len, column_len = pixel_neighborhood.shape
        for row_index in range(row_len):
            for column_index in range(column_len):
                neighborhood_pixel_value = pixel_neighborhood[row_index][column_index]
                if neighborhood_pixel_value == group_value:
                    stack.append(
                        Pixel(pixel.x-1+column_index, pixel.y-1+row_index, neighborhood_pixel_value)
                    )
                elif neighborhood_pixel_value != INVALID:
                    next_init_x = pixel.x-1+column_index
                    next_init_y = pixel.y-1+row_index
                    new_key = f'{next_init_x}, {next_init_y}'

                    if new_key not in added_points:
                        added_points.update({new_key: True})
                        next_inits.append(
                            Pixel(next_init_x, next_init_y, neighborhood_pixel_value)
                        )

        if len(stack) == 0 and len(next_inits) > 0:
            possible_init = next_inits.pop()
            while len(next_inits) != 0 and padded_image[possible_init.y+1, possible_init.x+1] == INVALID:
                possible_init = next_inits.pop()

            stack = [possible_init]
            group_value = None
            group_color = None

    return groups, groups_rect

def update_react(react, pixel):
    x_init, y_init, x_final, y_final = react

    if pixel.x < x_init:
        x_init = pixel.x
    if pixel.x > x_final:
        x_final = pixel.x

    if pixel.y < y_init:
        y_init = pixel.y
    if pixel.y > y_final:
        y_final = pixel.y
    
    return (x_init, y_init, x_final, y_final)

def get_neighborhood_array(array, x, y, neighbourhood=1, flatted=False, padded=False):
    if padded:
        x += neighbourhood
        y += neighbourhood

    min_y = y-neighbourhood
    min_x = x-neighbourhood

    if min_y < 0:
        min_y = 0

    if min_x < 0:
        min_x = 0

    # numpy slices handle if max range when matrix is not padded
    neighborhood_array = array[min_y: y+neighbourhood+1, min_x: x+neighbourhood+1]

    if flatted:
        return neighborhood_array.flatten()
    else:
        return neighborhood_array

def find_start_point(image):
    len_row_image, len_column_image = image.shape

    for row_index in range(0, len_row_image):
        for column_index in range(0, len_column_image):
            if image[row_index][column_index] == BLACK:
                return row_index, column_index

def check_neigh(image, neighborhood_length, x, y, check_by=WHITE):
    neighborhood = get_neighborhood_array(image, x, y, neighborhood_length, flatted=True)
    count = 0

    for pixel in neighborhood:
        if pixel == check_by:
            count += 1

    if count == len(neighborhood):
        return 1
    if count >= HIT_RULE:
        return 2
    else:
        return 0

def erosion(image, neighborhood_length=1, erode_by=BLACK, positive_color=BLACK, negative_color=WHITE):
    return __erode_or_dilate__(image, neighborhood_length, 'erode', dilate_by, positive_color, negative_color)

def dilate(image, neighborhood_length=1, dilate_by=BLACK, positive_color=BLACK, negative_color=WHITE):
    return __erode_or_dilate__(image, neighborhood_length, 'dilate', dilate_by, positive_color, negative_color)

def __erode_or_dilate__(image, neighborhood_length, mode, check_by, positive_color, negative_color):
    dilated_image = image.copy() 

    len_image_row, len_image_column = dilated_image.shape
    for row_index in range(len_image_row):
        for column_index in range(len_image_column):
            to_check = check_neigh(image, neighborhood_length, column_index, row_index, check_by)
            
            if mode == 'dilate':
                hit_or_fit = HIT
            elif mode == 'erode':
                hit_or_fit = FIT

            if to_check == hit_or_fit:
                dilated_image[row_index, column_index] = positive_color
            else:
                dilated_image[row_index, column_index] = negative_color

    return dilated_image

def plot_grouped_image(image, groups_rect, color=RGB_RED, title='Imagem com Grupos Demarcados', show=True):
    groups_length = len(groups_rect)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    for group in groups_rect:
        image = cv.rectangle(image, (group[0] , group[1]), (group[2] , group[3]), color, 1)

    plot, img = plt.subplots()
    plot.suptitle(title)
    img.imshow(image)
    if show:
        plt.show()


if __name__ == "__main__":
    first_image, second_image, third_image = get_images()
    teste = cv.imread('./teste1.jpg', cv.IMREAD_GRAYSCALE)

    first_image = get_black_white_image(first_image, 210)
    second_image = get_black_white_image(second_image, 200)
    third_image = get_black_white_image(third_image, 100)
    teste = get_black_white_image(teste, 100)

    plot, img = plt.subplots()
    plot.suptitle('teste dilatado com 3x3')
    test_react = cv.cvtColor(first_image, cv.COLOR_BGR2RGB)
    test_react = cv.rectangle(test_react, (2 , 10), (50 , 100), RGB_RED, 1)
    img.imshow(test_react)
    plt.show()

    teste_dilatado = dilate(teste)
    teste_dilatado_palavras = dilate(teste, 2)
    imagens, groups_rect_dilatado = segment(teste_dilatado)
    imagens, groups_rect_dilatado_palavras = segment(teste_dilatado_palavras)

    plot_grouped_image(teste_dilatado, groups_rect_dilatado, show=False)
    plot_grouped_image(teste_dilatado_palavras, groups_rect_dilatado_palavras)
