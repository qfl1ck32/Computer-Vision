from numpy import ndarray
from os.path import exists
from pathlib import Path

from classifier import create_digit_to_image_map

from constants import train_path, SudokuType, digit_images_jigsaw_sudoku_color_path, \
    digit_images_classic_sudoku_path, digit_images_jigsaw_sudoku_grayscale_path, digit_images_path

from processor import extract_digits, process_image_to_bird_eye_view

from reader import get_files

from cv2 import imwrite


def handle_folders_if_not_exist():
    for path in [digit_images_classic_sudoku_path,
                 digit_images_jigsaw_sudoku_grayscale_path, digit_images_jigsaw_sudoku_color_path]:
        if not exists(path):
            Path(path).mkdir(parents=True)


def _write_digit_to_image_map(digit_to_image_map: dict, path: str):
    for key in digit_to_image_map.keys():
        imwrite(f'{path}/{key}.jpg', digit_to_image_map.get(key))


def _write_digit_images_for_sudoku_type(image: ndarray, answers: ndarray, sudoku_type: SudokuType, path: str):
    digits_images = extract_digits(process_image_to_bird_eye_view(image))

    _write_digit_to_image_map(
        create_digit_to_image_map(digits_images, answers, sudoku_type),
        path
    )


def generate_digit_images_files():
    classic_sudoku_files = get_files(
        path=train_path,
        sudoku_type=SudokuType.CLASSIC,
        answers_with_bonus=True,
        limit=1
    )

    jigsaw_sudoku_files = get_files(
        path=train_path,
        sudoku_type=SudokuType.JIGSAW,
        answers_with_bonus=True,
        limit=3
    )

    handle_folders_if_not_exist()

    _write_digit_images_for_sudoku_type(classic_sudoku_files.images[0], classic_sudoku_files.answers[0],
                                        SudokuType.CLASSIC, digit_images_classic_sudoku_path)

    _write_digit_images_for_sudoku_type(jigsaw_sudoku_files.images[0], jigsaw_sudoku_files.answers[0],
                                        SudokuType.JIGSAW, digit_images_jigsaw_sudoku_color_path)

    _write_digit_images_for_sudoku_type(jigsaw_sudoku_files.images[2], jigsaw_sudoku_files.answers[2],
                                        SudokuType.JIGSAW, digit_images_jigsaw_sudoku_grayscale_path)
