from constants import SudokuType
from pathlib import Path
from os.path import exists


def remove_predicted_digits_from_result(sudoku_type: SudokuType, result: str):
    if sudoku_type == SudokuType.CLASSIC:
        return ''.join(map(lambda char: char if char == 'o' else 'x', result))

    return ''.join(result[i: i + 2] if result[i + 1] == 'o' else result[i] + 'x' for i in range(0, 81 * 2 - 1, 2))


def write_solution(sudoku_type: SudokuType, results: [str], write_path: str):
    step = 9 if sudoku_type == SudokuType.CLASSIC else 9 * 2
    length = 81 if sudoku_type == SudokuType.CLASSIC else 81 * 2

    if not exists(write_path):
        Path(write_path).mkdir(parents=True)

    for index, result in enumerate(results):
        file_index = f"{index + 1}"

        path = f"{write_path}{file_index}"

        with open(f"{path}_bonus_predicted.txt", "w") as file:
            file.write('\n'.join(result[i: i + step] for i in range(0, length, step)))

        with open(f"{path}_predicted.txt", "w") as file:
            without_digit_prediction = remove_predicted_digits_from_result(sudoku_type, result)

            file.write('\n'.join(without_digit_prediction[i: i + step] for i in range(0, length, step)))
