import math
import cv2 as cv
import numpy as np

from collections import namedtuple
from matplotlib import pyplot as plt

Pixel = namedtuple('Pixel', ['x', 'y', 'value'])
BLACK = 0
WHITE = 255

# 107 is prime, so the rule in get_next_color "never" get this value.
INVALID = 107
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

        groups[-1][pixel.y, pixel.x] = group_color
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

        # print(len(stack), pixel.x , pixel.y)
        if len(stack) == 0 and len(next_inits) > 0:
            possible_init = next_inits.pop()
            while len(next_inits) != 0 and padded_image[possible_init.y+1, possible_init.x+1] == INVALID:
                possible_init = next_inits.pop()

            stack = [possible_init]
            group_value = None
            group_color = None

    return groups

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

def get_cross_neighbourhood(image, x, y):
    
    pos_x = x
    pos_y = y
    len_row_image, len_column_image = image.shape
    north_neigh, west_neigh, center_neigh, east_neigh, south_neigh = 0, 0, 0, 0, 0

    if pos_x > 0:
        if pos_y > 0:
            if pos_x < len_row_image - 1:
                if pos_y < len_column_image - 1:
                    north_neigh = image[pos_x - 1][pos_y]
                    west_neigh = image[pos_x][pos_y - 1]
                    center_neigh = image[pos_x][pos_y]
                    east_neigh = image[pos_x][pos_y + 1]
                    south_neigh = image[pos_x + 1][pos_y] 

    else:      
        if pos_x == 0:
            if pos_y == 0:
                center_neigh = image[pos_x][pos_y]
                east_neigh = image[pos_x][pos_y + 1]
                south_neigh = image[pos_x + 1][pos_y]

                
        if pos_x == 0:
            if pos_y > 0:
                if pos_y < len_column_image - 1:
                    west_neigh = image[pos_x][pos_y - 1]
                    center_neigh = image[pos_x][pos_y]
                    east_neigh = image[pos_x][pos_y + 1]
                    south_neigh = image[pos_x + 1][pos_y]


        if pos_x > 0:
            if pos_y == 0:
                if row_index < len_image_row - 1:
                    north_neigh = image[pos_x - 1][pos_y]
                    center_neigh = image[pos_x][pos_y]
                    east_neigh = image[pos_x][pos_y + 1]
                    south_neigh = image[pos_x + 1][pos_y]


        if pos_x == 0:
            if pos_y == len_image_row:
                center_neigh = image[pos_x][pos_y]
                east_neigh = image[pos_x][pos_y + 1]
                south_neigh = image[pos_x + 1][pos_y]


        if pos_x > 0:
            if pos_y < len_image_column - 1:

                north_neigh = image[pos_x - 1][pos_y]
                west_neigh = image[pos_x][pos_y - 1]
                center_neigh = image[pos_x][pos_y]
                south_neigh = image[pos_x + 1][pos_y]
                
                    
        if pos_x == len_image_row:
            if pos_y == len_image_column:
                north_neigh = image[pos_x - 1][pos_y]
                west_neigh = image[pos_x][pos_y - 1]
                center_neigh = image[pos_x][pos_y]

    return (north_neigh, west_neigh, center_neigh, east_neigh, south_neigh)

def find_start_point(image):

    len_row_image, len_column_image = image.shape

    for row_index in range(0, len_row_image):
        for column_index in range(0, len_column_image):
            if image[row_index][column_index] == BLACK:
                return row_index, column_index

def check_neigh(image, x, y):

    north, west, center, east, south = get_cross_neighbourhood(image, x, y)
    count = 0

    if north == BLACK:
        count += 1
    if west == BLACK:
        count += 1
    if center == BLACK:
        count += 1
    if east == BLACK:
        count += 1
    if south == BLACK:
        count += 1

    if count == 5:
        return 1

    if count >= 3:
        return 2

    else:
        return 0

def erosion(image):

    eroded_image = image.copy()
    len_image_row, len_image_column = eroded_image.shape
    x, y = find_start_point(eroded_image)

    for row_index in range(x, len_image_row):
        for column_index in range(y, len_image_row):
            count = check_neigh(eroded_image, len_image_row, len_image_column)

            if count < 5:
                if row_index > 0:
                    if column_index > 0:
                        if row_index < len_image_row - 1:
                            if column_index < len_image_column - 1:
                                eroded_image[row_index - 1][column_index] = WHITE
                                eroded_image[row_index][column_index - 1] = WHITE
                                eroded_image[row_index][column_index] = WHITE
                                eroded_image[row_index][column_index + 1] = WHITE
                                eroded_image[row_index + 1][column_index] = WHITE
                else:
                    if row_index == 0:
                        if column_index == 0:
                            eroded_image[row_index][column_index] = WHITE
                            eroded_image[row_index][column_index + 1] = WHITE
                            eroded_image[row_index + 1][column_index] = WHITE

                    if row_index == 0:
                        if column_index > 0:
                            if column_index < len_image_column - 1:
                                eroded_image[row_index][column_index - 1] = WHITE
                                eroded_image[row_index][column_index] = WHITE
                                eroded_image[row_index][column_index + 1] = WHITE
                                eroded_image[row_index + 1][column_index] = WHITE
                    
                    if row_index > 0:
                        if column_index == 0:
                            if row_index < len_image_row - 1:
                                eroded_image[row_index - 1][column_index] = WHITE
                                eroded_image[row_index][column_index] = WHITE
                                eroded_image[row_index][column_index + 1] = WHITE
                                eroded_image[row_index + 1][column_index] = WHITE

                    if row_index == 0:
                        if column_index == len_image_row:
                            eroded_image[row_index][column_index + 1] = WHITE
                            eroded_image[row_index][column_index] = WHITE
                            eroded_image[row_index + 1][column_index] = WHITE

                    if row_index > 0:
                        if column_index < len_image_column - 1:
                            eroded_image[row_index - 1][column_index] = WHITE
                            eroded_image[row_index][column_index - 1] = WHITE
                            eroded_image[row_index][column_index] = WHITE
                            eroded_image[row_index + 1][column_index] = WHITE

                    if row_index == len_image_row:
                        if column_index == len_image_column:
                            eroded_image[row_index - 1][column_index] = WHITE
                            eroded_image[row_index][column_index - 1] = WHITE
                            eroded_image[row_index][column_index] = WHITE
    return eroded_image

def dilate(image):
    dilated_image = image.copy() 

    len_image_row, len_image_column = dilated_image.shape
    x, y = find_start_point(dilated_image)

    for row_index in range(x, len_image_row):
        for column_index in range(y, len_image_row):
            count = check_neigh(dilated_image, len_image_row, len_image_column)

            if count >= 3:
                if row_index > 0:
                    if column_index > 0:
                        if row_index < len_image_row - 1:
                            if column_index < len_image_column - 1:
                                dilated_image[row_index - 1][column_index] = BLACK
                                dilated_image[row_index][column_index - 1] = BLACK
                                dilated_image[row_index][column_index] = BLACK
                                dilated_image[row_index][column_index + 1] = BLACK
                                dilated_image[row_index + 1][column_index] = BLACK
                else:
                    if row_index == 0:
                        if column_index == 0:
                            dilated_image[row_index][column_index] = BLACK
                            dilated_image[row_index][column_index + 1] = BLACK
                            dilated_image[row_index + 1][column_index] = BLACK

                    if row_index == 0:
                        if column_index > 0:
                            if column_index < len_image_column - 1:
                                dilated_image[row_index][column_index - 1] = BLACK
                                dilated_image[row_index][column_index] = BLACK
                                dilated_image[row_index][column_index + 1] = BLACK
                                dilated_image[row_index + 1][column_index] = BLACK
                    
                    if row_index > 0:
                        if column_index == 0:
                            if row_index < len_image_row - 1:
                                dilated_image[row_index - 1][column_index] = BLACK
                                dilated_image[row_index][column_index] = BLACK
                                dilated_image[row_index][column_index + 1] = BLACK
                                dilated_image[row_index + 1][column_index] = BLACK

                    if row_index == 0:
                        if column_index == len_image_row:
                            dilated_image[row_index][column_index + 1] = BLACK
                            dilated_image[row_index][column_index] = BLACK
                            dilated_image[row_index + 1][column_index] = BLACK

                    if row_index > 0:
                        if column_index < len_image_column - 1:
                            dilated_image[row_index - 1][column_index] = BLACK
                            dilated_image[row_index][column_index - 1] = BLACK
                            dilated_image[row_index][column_index] = BLACK
                            dilated_image[row_index + 1][column_index] = BLACK

                    if row_index == len_image_row:
                        if column_index == len_image_column:
                            dilated_image[row_index - 1][column_index] = BLACK
                            dilated_image[row_index][column_index - 1] = BLACK
                            dilated_image[row_index][column_index] = BLACK

    return dilated_image


if __name__ == "__main__":
    first_image, second_image, third_image = get_images()
    show_and_wait(first_image, "imagem total")
    first_image = get_black_white_image(first_image, 210)
    show_and_wait(first_image, "imagem preto e branco")

    cv.imwrite('./black_white.jpg', first_image)
    second_image = get_black_white_image(second_image, 200)
    third_image = get_black_white_image(third_image, 100)
    imagens = segment(third_image)
    print(len(imagens))
    # for i in range(len(imagens)):
    #     show_and_wait(imagens[i], f"imagem {i}")
