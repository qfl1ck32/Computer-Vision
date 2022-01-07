import numpy as np
import cv2

from math import ceil

from constants import image_height, image_width


def does_not_intersect(boxes: [tuple], x: int, y: int):
    patch_x_max = x + image_width
    patch_y_max = y + image_height

    no_intersect = True

    for (x_min, y_min, x_max, y_max) in boxes:
        if not (x_min > patch_x_max or
                x_max < x or
                y_min > patch_y_max or
                y_max < y):
            no_intersect = False

            break

    return no_intersect


def add_bounding_box(image: np.ndarray, box: tuple, color=(255, 0, 0)):
    min_x, min_y, max_x, max_y = box

    return cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, 2)


def sliding_window(image: np.ndarray, step_size: int, window_size: (int, int)):
    window_size_x, window_size_y = window_size
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield x, y, image[y: y + window_size_y, x: x + window_size_x]


def get_yellow_percentage(image: np.ndarray):
    quantity = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if image[i][j][0] < 80 and image[i][j][1] > 100 and image[i][j][2] > 150:
                quantity += 1

    return quantity / (image.shape[0] * image.shape[1])


# This is not used, anymore. Will be implemented in the facial recognition algorithm later.
def get_resized_and_distorted_image(image: np.ndarray):
    for new_size in [1]:
        for xy_ratio in [(1, 1), (2, 1), (1, 2)]:
            yield new_size, xy_ratio, cv2.resize(image,
                                                 dsize=(int(new_size * image.shape[1] // xy_ratio[1]),
                                                        int(new_size * image.shape[0] // xy_ratio[0])))


# This is not used, anymore. Will be implemented in the facial recognition algorithm later.
def get_corresponding_box_after_resize_and_distortion(box: tuple, window_size: int, new_size: float, xy_ratio: tuple):
    box_x, box_y = box

    xy_ratio_float = ceil(xy_ratio[0] / xy_ratio[1])
    yx_ratio_float = ceil(xy_ratio[1] / xy_ratio[0])

    return box_x * yx_ratio_float // new_size, box_y * xy_ratio_float // new_size, \
           (box_x + window_size) * yx_ratio_float // new_size, (box_y + window_size) * xy_ratio_float // new_size
