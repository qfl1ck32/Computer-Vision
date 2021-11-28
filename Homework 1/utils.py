from typing import Callable

from numpy import zeros, ndarray, array, argmin, argmax, diff, sqrt
from cv2 import warpPerspective, getPerspectiveTransform


def apply_pipeline(image: ndarray, pipeline: [Callable]):
    assert (len(pipeline) > 0)

    last_result = pipeline[0](image)

    for func in pipeline[1:]:
        last_result = func(last_result)

    return last_result


def get_clockwise_sorted_points_in_rectangle(points: ndarray):
    """

    @param points: An array of 4 points
    @return: A copy of the array, sorted clockwise
    """

    assert (len(points) == 4)

    rectangle = zeros(
        shape=(4, 2),
        dtype="float32"
    )

    points_sum = points.sum(axis=1)

    rectangle[0] = points[argmin(points_sum)]
    rectangle[2] = points[argmax(points_sum)]

    points_diff = diff(
        a=points,
        axis=1
    )

    rectangle[1] = points[argmin(points_diff)]
    rectangle[3] = points[argmax(points_diff)]

    return rectangle


def get_euclidian_distance(points: list[array]):
    """

    :param points: A list of two points, e.g. [ [0, 1], [1, 0] ]
    :return: The euclidian distance between the two points
    """

    assert (len(points) == 2)

    ((x1, y1), (x2, y2)) = points

    return sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)


def get_bird_eye_view(image: ndarray, corner_points: ndarray):
    """

    :param image: An image
    :param corner_points: An array of 4 points, representing the corners of
        the ROI to extract
    :return: The ROI, as seen from above
    """

    rectangle = get_clockwise_sorted_points_in_rectangle(corner_points)

    (top_left, top_right, bottom_right, bottom_left) = rectangle

    width_top = get_euclidian_distance([top_left, top_right])
    width_bottom = get_euclidian_distance([bottom_left, bottom_right])

    max_width = max(int(width_top), int(width_bottom))

    height_left = get_euclidian_distance([top_left, bottom_left])
    height_right = get_euclidian_distance([top_right, bottom_right])

    max_height = max(int(height_left), int(height_right))

    destination_points = array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    transform = getPerspectiveTransform(
        src=rectangle,
        dst=destination_points
    )

    return warpPerspective(
        src=image,
        M=transform,
        dsize=(max_width, max_height)
    )
