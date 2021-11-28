from cv2 import matchTemplate
from cv2 import TM_CCOEFF
from numpy import ndarray, zeros, array
from constants import sudoku_size, SudokuType


def create_neighbourhood_matrix(vertical_edges: ndarray, horizontal_edges: ndarray):
    neighbourhood_matrix = zeros(shape=(sudoku_size, sudoku_size), dtype="object")

    for i in range(sudoku_size):
        for j in range(sudoku_size):
            top = False if i == 0 else not horizontal_edges[i - 1][j]
            right = False if j == sudoku_size - 1 else not vertical_edges[i][j]
            bottom = False if i == sudoku_size - 1 else not horizontal_edges[i][j]
            left = False if j == 0 else not vertical_edges[i][j - 1]

            neighbourhood_matrix[i][j] = array([top, right, bottom, left])

    return neighbourhood_matrix


def fill(result: ndarray, visited: ndarray, neighbourhood_matrix: ndarray, coordinate_changes: ndarray, row: int, column: int, index: int):
    if row >= sudoku_size or column >= sudoku_size or visited[row][column]:
        return

    visited[row][column] = True

    result[row][column] = index

    neighbours = neighbourhood_matrix[row][column]

    for neighbour_index, neighbour in enumerate(neighbours):
        if neighbour:
            changes = coordinate_changes[neighbour_index]

            new_row = row + changes[0]
            new_column = column + changes[1]

            if visited[new_row][new_column]:
                continue

            fill(result, visited, neighbourhood_matrix, coordinate_changes, new_row, new_column, index)


def get_boundaries_matrix(vertical_edges: ndarray, horizontal_edges: ndarray):
    result = zeros(shape=(sudoku_size, sudoku_size))
    visited = zeros(shape=(sudoku_size, sudoku_size))

    neighbourhood_matrix = create_neighbourhood_matrix(vertical_edges, horizontal_edges)
    coordinate_changes = array([(-1, 0), (0, 1), (1, 0), (0, -1)])

    index = 1

    for row in range(sudoku_size):
        for column in range(sudoku_size):
            if visited[row][column]:
                continue

            fill(result, visited, neighbourhood_matrix, coordinate_changes, row, column, index)

            index += 1

    return result


def create_digit_to_image_map(images: ndarray, answers: ndarray, sudoku_type: SudokuType):
    digit_to_image_map = {}

    exists = dict()

    is_classic_sudoku = sudoku_type == SudokuType.CLASSIC

    for i in range(len(images)):
        image = images[i]
        answer = answers[i if is_classic_sudoku else 2 * i + 1]

        if answer != 'o' and answer not in exists.values() and image is not None:
            exists[answer] = True
            digit_to_image_map[int(answer)] = image

        if len(exists.values()) == 9:
            break

    if len(exists.values()) < 9:
        raise Exception("Not enough digits were identified. Please use another image.")

    return digit_to_image_map


def _classify_image(image: ndarray, digit_to_image_map: dict = None):
    if image is None:
        return 'o'

    if digit_to_image_map is None:
        return 'x'

    digit_to_score_map = dict()

    for i in range(1, 10, 1):
        image_for_current_digit = digit_to_image_map[i]

        match = matchTemplate(
            image=image,
            templ=image_for_current_digit,
            method=TM_CCOEFF
        )

        digit_to_score_map[str(i)] = match[0]

    return max(digit_to_score_map, key=digit_to_score_map.get)


def classify_images(images: ndarray, digit_to_image_map: dict = None):
    return list(map(lambda img: _classify_image(img, digit_to_image_map), images))
