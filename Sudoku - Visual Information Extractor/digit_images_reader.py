from cv2 import imread

from constants import digit_images_jigsaw_sudoku_color_path, \
    digit_images_classic_sudoku_path, digit_images_jigsaw_sudoku_grayscale_path, SudokuType, sudoku_size
from processor import convert_to_grayscale


def get_digit_images_map(sudoku_type: SudokuType, is_color=False):
    assert(is_color is False or sudoku_type == SudokuType.JIGSAW and is_color)

    path = digit_images_classic_sudoku_path if sudoku_type == SudokuType.CLASSIC else \
        digit_images_jigsaw_sudoku_color_path if is_color else digit_images_jigsaw_sudoku_grayscale_path

    mp: dict = dict()

    for i in range(1, sudoku_size + 1):
        mp[i] = convert_to_grayscale(imread(f'{path}/{i}.jpg'))

    return mp
