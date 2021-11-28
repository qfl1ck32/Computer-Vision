from enum import Enum
from numpy import ndarray

train_path: str = "./antrenare"
digit_images_path: str = "./digit_images"

digit_images_classic_sudoku_path: str = f'{digit_images_path}/classic'

digit_images_jigsaw_sudoku_color_path: str = f'{digit_images_path}/jigsaw/color'

digit_images_jigsaw_sudoku_grayscale_path: str = f'{digit_images_path}/jigsaw/grayscale'

solutions_path = "./evaluare/fisiere_solutie/Rusu_Andrei_331/"

sudoku_size = 9


class SudokuType(Enum):
    CLASSIC = "classic"
    JIGSAW = "jigsaw"


class SudokuFiles:
    images: list[ndarray]
    answers: list[ndarray]

    def __init__(self, images: list[ndarray], answers: list[ndarray]):
        self.images = images
        self.answers = answers


class Files:
    classic: SudokuFiles
    jigsaw: SudokuFiles

    def __init__(self, classic: SudokuFiles, jigsaw: SudokuFiles):
        self.classic = classic
        self.jigsaw = jigsaw


def get_write_path(sudoku_type: SudokuType):
    return f"{solutions_path}{'clasic' if sudoku_type == SudokuType.CLASSIC else 'jigsaw'}/"
