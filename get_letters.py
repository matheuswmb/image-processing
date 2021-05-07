import math
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
                blackAndWhiteImage[row_index][column_index] = WHITE
            else:
                blackAndWhiteImage[row_index][column_index] = BLACK

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

    # show_and_wait(get_black_white_image(first_image, 220), 'primeira')
    # show_and_wait(cv.threshold(first_image, 220, 255, cv.THRESH_BINARY)[1], 'primeira opencv')

    kernel = np.ones((2, 2), dtype=np.uint8)

    #show_and_wait(get_black_white_image(second_image, 200), 'segunda')
    #eroded = erosion(second_image, kernel)
    #show_and_wait(eroded)
    # show_and_wait(cv.threshold(second_image, 200, 255, cv.THRESH_BINARY)[1], 'segunda opencv')

    # show_and_wait(get_black_white_image(third_image, 100),  'terceira')
    # show_and_wait(cv.threshold(third_image, 100, 255, cv.THRESH_BINARY)[1], 'terceira opencv')

    #a = np.array([[7,20,32,40,5,6],[1,2,3,4,5,6],[3,30,300,3000,30000,300000],[1,2,3,4,5,6],[1,20,3,4,5,60], [1,2,3,4,5,6]])
    #print(get_neighborhood_array(a, 5, 2))
     
    teste1 = cv.imread('./teste1.jpg', cv.IMREAD_GRAYSCALE)
    teste2 = cv.imread('./teste2.jpg', cv.IMREAD_GRAYSCALE)
    teste1 = get_black_white_image(teste1, 140)
    teste2 = get_black_white_image(teste2, 210)    

    # -- teste1 becomes the new image and the return of the function is a safe copy of the image --
    #show_and_wait(teste1, 'teste1')
    #eroded = erosion(teste1)
    #erosaocv = cv.erode(teste1,kernel,iterations = 1)
    #show_and_wait(eroded, 'teste1 erodido')
    #show_and_wait(erosaocv, 'teste1 opencv')
    show_and_wait(teste1)
    dilatacao = dilate(teste1)
    show_and_wait(dilatacao, 'teste1 dilatacao')
    dilation = cv.dilate(teste1,kernel,iterations = 1)
    show_and_wait(dilation, 'teste1 dilatacao opencv')

    #erosao2 = erosion(teste2)
    #erosaocv2 = cv.erode(teste2,kernel,iterations = 1)
    #show_and_wait(teste2, 'teste2')
    #show_and_wait(erosao2, 'teste2 erodido')
    #show_and_wait(erosaocv2, 'teste2 opencv')
    #dilatacao2 = dilate(teste2)
    #dilation2 = cv.dilate(first_image,kernel,iterations = 1)
    #show_and_wait(dilatacao, 'teste2 dilatacao')
    #show_and_wait(dilation2, 'teste2 dilatacao opencv')

    #first_image = get_black_white_image(first_image, 220)
    #show_and_wait(first_image, 'preta e branca')
    #erosion(third_image)
    #dilate(first_image)
    #show_and_wait(first_image, 'dilatacao')

    
    #erosion = cv.erode(first_image,kernel,iterations = 1)
    #show_and_wait(erosion, 'opencv')
    #show_and_wait(dilation, 'opencv')
