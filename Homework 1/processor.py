from cv2 import cvtColor, boundingRect, erode, dilate, getStructuringElement, morphologyEx, countNonZero, \
    findContours, arcLength, bitwise_not, approxPolyDP, contourArea, adaptiveThreshold, threshold, drawContours, \
    GaussianBlur, THRESH_OTSU, bitwise_and, resize

from cv2 import MORPH_RECT, MORPH_OPEN, COLOR_BGR2GRAY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, THRESH_BINARY, \
    ADAPTIVE_THRESH_MEAN_C

from numpy import ndarray, zeros_like, ones, zeros, mean

from skimage.segmentation import clear_border

from imutils import grab_contours

from constants import sudoku_size
from utils import get_bird_eye_view, apply_pipeline


def is_color_sudoku(sudoku: ndarray):
    size = max(sudoku.shape)

    cell_size = int(size / 9)

    cell = sudoku[0: cell_size, 0: cell_size]

    b, g, r = mean(cell[:, :, 0]), mean(cell[:, :, 1]), mean(cell[:, :, 2])

    return abs(b - g) > 10 or abs(b - r) > 10 or abs(g - r) > 10


def convert_to_grayscale(image: ndarray):
    return cvtColor(src=image, code=COLOR_BGR2GRAY)


def invert(image: ndarray):
    return bitwise_not(src=image)


def find_contours(image: ndarray):
    contours = findContours(
        image=image,
        mode=RETR_EXTERNAL,
        method=CHAIN_APPROX_SIMPLE
    )

    contours = grab_contours(contours)

    contours = sorted(contours, key=contourArea, reverse=True)

    for contour in contours:
        perimeter = arcLength(curve=contour, closed=True)

        contour_approximation = approxPolyDP(
            curve=contour,
            epsilon=0.02 * perimeter,
            closed=True
        )

        if len(contour_approximation) == 4:
            return image, contour_approximation.reshape(4, 2)

    raise Exception("The contour couldn't be found.")


def _resize(image: ndarray):
    if image is None:
        return None

    return resize(
        src=image,
        dsize=(128, 128)
    )


def _resize_images(images: ndarray):
    return list(map(_resize, images))


def _add_gaussian_blur(image: ndarray):
    return GaussianBlur(
        src=image,
        ksize=(23, 23),
        sigmaX=5
    )


def _add_adaptive_threshold(image: ndarray):
    return adaptiveThreshold(
        src=image,
        maxValue=255,
        adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
        thresholdType=THRESH_BINARY,
        blockSize=25,
        C=2
    )


def _get_bird_eye_view(args: (ndarray, ndarray)):
    (image, contour_approximation) = args

    return get_bird_eye_view(
        image=image,
        corner_points=contour_approximation
    )


def _extract_digit(cell: ndarray):
    cell_without_border = clear_border(
        labels=cell,
        buffer_size=16
    )

    contours = findContours(
        image=cell_without_border,
        mode=RETR_EXTERNAL,
        method=CHAIN_APPROX_SIMPLE
    )

    contours = grab_contours(contours)

    if not len(contours):
        return None

    contour = max(contours, key=contourArea)

    mask = zeros_like(cell_without_border, dtype="uint8")

    drawContours(
        image=mask,
        contours=[contour],
        contourIdx=-1,
        color=255,
        thickness=-1
    )

    (height, width) = mask.shape

    percent_filled = countNonZero(mask) / float(width * height)

    if percent_filled < 0.02:
        return None

    digit = bitwise_and(
        src1=cell_without_border,
        src2=cell_without_border,
        mask=mask
    )

    x, y, w, h = boundingRect(contour)

    region_of_interest = digit[y: y + h, x: w + x]

    return region_of_interest


def _extract_digits(sudoku: ndarray):
    size = max(sudoku.shape)

    cell_size = int(size / 9)

    digit_images = []

    for row in range(9):
        for column in range(9):
            cell = sudoku[row * cell_size: (row + 1) * cell_size, column * cell_size: (column + 1) * cell_size]

            digit = _extract_digit(cell)

            digit_images.append(digit)

    return digit_images


def _keep_only_boundaries(sudoku: ndarray, is_color: bool):
    contours, _ = findContours(sudoku, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

    sudoku = drawContours(sudoku, [contours[0]], -1, (0, 0, 0), 10)

    eroded = erode(sudoku, ones((21, 21) if is_color else (15, 15), dtype="uint8"), iterations=1)

    eroded = erode(eroded, ones((5, 5), dtype="uint8"), iterations=1)

    dilated = dilate(eroded, ones((15, 15), dtype="uint8"), iterations=3)

    eroded = erode(dilated, ones((17, 17), dtype="uint8"), iterations=1)

    _, thresh = threshold(eroded, thresh=127, maxval=255, type=THRESH_BINARY)

    return thresh


def _get_vertical_and_horizontal_edges_matrix(sudoku: ndarray):
    size = max(sudoku.shape)

    cell_size = int(size / 9)

    half_cell_size = cell_size // 2

    edge_height = cell_size
    edge_width = 250

    vertical_edges_matrix = zeros(shape=(sudoku_size, sudoku_size - 1))
    horizontal_edges_matrix = zeros(shape=(sudoku_size - 1, sudoku_size))

    vertical_structure = getStructuringElement(MORPH_RECT, (1, edge_height))
    horizontal_structure = getStructuringElement(MORPH_RECT, (edge_width, 1))

    for row in range(sudoku_size):
        for column in range(sudoku_size - 1):
            patch = sudoku[row * cell_size:
                           (row + 1) * cell_size, column * cell_size + half_cell_size: (column + 1) * cell_size + half_cell_size]

            vertical_line = erode(patch, vertical_structure)

            vertical_edges_matrix[row][column] = bool(countNonZero(vertical_line))

    for row in range(sudoku_size - 1):
        for column in range(sudoku_size):
            patch = sudoku[row * cell_size + half_cell_size:
                           (row + 1) * cell_size + half_cell_size, column * cell_size: (column + 1) * cell_size]

            horizontal_line = erode(patch, horizontal_structure)

            horizontal_edges_matrix[row][column] = bool(countNonZero(horizontal_line))

    return vertical_edges_matrix, horizontal_edges_matrix


def process_image(image: ndarray):
    return apply_pipeline(image, [
        convert_to_grayscale,
        _add_gaussian_blur,
        _add_adaptive_threshold,
        invert,
    ])


def process_image_to_bird_eye_view(image: ndarray):
    return apply_pipeline(image, [
        process_image,
        find_contours,
        _get_bird_eye_view,
    ])


def get_vertical_and_horizontal_edges_matrix(bird_eye_view_image: ndarray, is_color: bool):
    only_with_boundaries = _keep_only_boundaries(bird_eye_view_image, is_color)

    return _get_vertical_and_horizontal_edges_matrix(only_with_boundaries)


def extract_digits(bird_eye_view_image: ndarray):
    return apply_pipeline(bird_eye_view_image, [
        _extract_digits,
        _resize_images
    ])
