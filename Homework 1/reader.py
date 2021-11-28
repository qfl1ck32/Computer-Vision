from glob import glob
from os.path import join

from cv2 import imread
from numpy import array, loadtxt

from constants import SudokuFiles, SudokuType

Infinity = 0x40000

file_image_extension: str = '.jpg'
file_answer_extension: str = '.txt'


def _get_images_from_folder(path: str, limit: int = Infinity):
    file_names = glob(f'{path}/*{file_image_extension}')

    return list(map(lambda file_name: imread(file_name), file_names[:limit]))


def _get_answers_from_folder(path: str, bonus: bool = False, limit: int = Infinity):
    file_names = glob(f'{path}/*{"bonus*" if bonus else ""}{file_answer_extension}')

    files = list(map(lambda file_name: loadtxt(file_name, dtype=str), file_names[:limit]))

    return list(map(lambda file: array([letter for line in file for letter in line]), files))


def _get_path(directory: str, sudoku_type: SudokuType):
    return join(directory, "clasic" if sudoku_type == SudokuType.CLASSIC else "jigsaw")


def get_files(path: str, sudoku_type: SudokuType, answers_with_bonus=False, limit=None) -> SudokuFiles:
    return SudokuFiles(
        images=_get_images_from_folder(path=_get_path(path, sudoku_type), limit=limit),
        answers=_get_answers_from_folder(path=_get_path(path, sudoku_type), bonus=answers_with_bonus, limit=limit)
    )
