import cv2

import numpy as np

from constants import character_names, image_height, image_width, train_directory, positive_examples_directory, \
    negative_examples_directory
from solution.utils import get_yellow_percentage, add_bounding_box
from utils import does_not_intersect, sliding_window


def generate_positive_examples():
    filename_indexes = {k: 0 for k in character_names}

    for name in character_names:
        with open(f"{train_directory}/{name}.txt") as annotations:
            last_file_name = ""
            current_image = []

            for line in annotations.readlines():
                file_name, x_min, y_min, x_max, y_max, character_name = line.split()

                if last_file_name != file_name:
                    current_image = cv2.imread(f"{train_directory}/{name}/{file_name}", cv2.IMREAD_COLOR)
                    last_file_name = file_name

                face = current_image[int(y_min): int(y_max), int(x_min): int(x_max)]

                face_warped = cv2.resize(face, (image_height, image_width))

                cv2.imwrite(f"{positive_examples_directory}/{character_name}_{filename_indexes[character_name]}.jpg",
                            face_warped)

                filename_indexes[character_name] += 1


def generate_negative_examples(count_per_image=10):
    filename_index = 0

    for name in character_names:
        if name == "unknown":
            break

        with open(f"{train_directory}/{name}.txt") as annotations:
            images = []
            all_boxes = []

            lines = annotations.readlines()

            line_index = 0

            while line_index < len(lines):
                file_name, x_min, y_min, x_max, y_max, character_name = lines[line_index].split()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                last_file_name = file_name

                current_image = cv2.imread(f"{train_directory}/{name}/{file_name}", cv2.IMREAD_COLOR)

                current_boxes = [(x_min, y_min, x_max, y_max)]

                while True:
                    if line_index + 1 == len(lines):
                        break

                    new_file_name, x_min, y_min, x_max, y_max, character_name = lines[line_index + 1].split()

                    if new_file_name != last_file_name:
                        break

                    line_index += 1

                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                    current_boxes.append((x_min, y_min, x_max, y_max))

                line_index += 1
                images.append(current_image)
                all_boxes.append(current_boxes)

            for index, (image, boxes) in enumerate(zip(images, all_boxes)):

                rows, columns, _ = image.shape

                tries = 0

                for _ in range(count_per_image):
                    while tries < 101:
                        x = np.random.randint(low=0, high=columns - image_width)
                        y = np.random.randint(low=0, high=rows - image_height)

                        does_not_have_intersections = does_not_intersect(boxes, x, y)

                        window = image[y: y + image_height, x: x + image_width]

                        if does_not_have_intersections or get_yellow_percentage(window) < 0.5:

                            cv2.imwrite(f"{negative_examples_directory}/{filename_index}.jpg", window)

                            filename_index += 1

                            break
                        else:
                            tries += 1
