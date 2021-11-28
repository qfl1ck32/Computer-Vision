from numpy import asarray, vstack

from argparse import ArgumentParser

from digit_images_generator import generate_digit_images_files
from digit_images_reader import get_digit_images_map

from processor import process_image, process_image_to_bird_eye_view, extract_digits, \
    get_vertical_and_horizontal_edges_matrix, \
    is_color_sudoku, find_contours

from utils import get_bird_eye_view

from reader import get_files

from constants import SudokuType, get_write_path, solutions_path

from classifier import classify_images, get_boundaries_matrix

from writer import write_solution

from logger import logging


def handle_classic_sudoku(path: str):
    sudoku_type: SudokuType = SudokuType.CLASSIC

    files = get_files(
        path=path,
        sudoku_type=sudoku_type,
        answers_with_bonus=True,
    )

    mp = get_digit_images_map(SudokuType.CLASSIC)

    results: [str] = []

    for index in range(len(files.images)):
        current_image = files.images[index]

        bird_eye_view_image = process_image_to_bird_eye_view(image=current_image)

        extracted_digits_images = extract_digits(bird_eye_view_image=bird_eye_view_image)

        result = ''.join(classify_images(images=extracted_digits_images, digit_to_image_map=mp))

        results.append(result)

    return results


def handle_jigsaw_sudoku(path: str):
    sudoku_type: SudokuType = SudokuType.JIGSAW

    files = get_files(
        path=path,
        sudoku_type=sudoku_type,
        answers_with_bonus=True,
    )

    color_sudoku_digit_map = get_digit_images_map(sudoku_type, True)
    grayscale_sudoku_digit_map = get_digit_images_map(sudoku_type, False)

    results: [str] = []

    for index in range(len(files.images)):
        current_image = files.images[index]

        processed = process_image(current_image)

        _, contours = find_contours(processed)

        color_bird_eye_view = get_bird_eye_view(current_image, contours)

        is_color = is_color_sudoku(color_bird_eye_view)

        bird_eye_view_image = get_bird_eye_view(processed, contours)

        vertical_edges, horizontal_edges = get_vertical_and_horizontal_edges_matrix(bird_eye_view_image, is_color)

        extracted_digits_images = extract_digits(bird_eye_view_image)

        digit_map = color_sudoku_digit_map if is_color else grayscale_sudoku_digit_map

        classification_result = classify_images(extracted_digits_images, digit_map)

        boundaries_matrix = get_boundaries_matrix(vertical_edges, horizontal_edges)

        boundaries_array = asarray(boundaries_matrix, dtype="uint8").reshape(-1)

        result = ''.join(vstack([boundaries_array, classification_result]).ravel('F'))

        results.append(result)

    return results


def main():
    # Use only if training files change
    # generate_digit_images_files()

    parser = ArgumentParser()

    parser.add_argument("path", help="The path to the images")

    parser.add_argument("sudoku-type", help="The type of sudoku", choices=['classic', 'jigsaw'])

    parser.add_argument("--output", help="Where to save the answers", required=False)

    args, _ = parser.parse_known_args()

    path: str = args.path
    sudoku_type = getattr(args, 'sudoku-type')
    output: str = args.output

    if output and output[-1] == "/":
        output = output[:-1]

    logging.info("Running...")

    sudoku_type = SudokuType.CLASSIC if sudoku_type == "classic" else SudokuType.JIGSAW

    if sudoku_type == SudokuType.CLASSIC:
        results = handle_classic_sudoku(path)
    else:
        results = handle_jigsaw_sudoku(path)

    logging.info("Done. Writing the solution...")

    write_solution(sudoku_type, results,
                   f"{output}//{'classic' if sudoku_type == SudokuType.CLASSIC else 'jigsaw'}/"
                   if output else get_write_path(sudoku_type))

    logging.info("Success!")


if __name__ == '__main__':
    main()
