import timeit
import torch
import os

from torchvision.models import resnet18
from torchvision.ops import nms
from torch import tensor

from NeuralNetwork import test_image
from utils import *
from constants import *
from Logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_face_detector(model_task1: resnet18, model_task2: resnet18):
    image_filenames = os.listdir(test_directory)

    final_filenames = []
    final_detections = []
    final_scores = []
    name_indexes = []

    color_images = list(map(lambda filename: cv2.imread(f"{test_directory}/{filename}",
                                                        cv2.IMREAD_COLOR), image_filenames))

    for file_index, color_image in enumerate(color_images):
        logger.info(f"Processing image {file_index + 1}.")

        start_time = timeit.default_timer()

        detections = []
        scores = []

        for window_size in [(100, 100), (90, 90), (80, 80), (70, 70), (40, 40), (30, 30)]:
            for (x, y, window) in sliding_window(color_image, 6, window_size):

                if window.shape != (window_size[0], window_size[1], image_number_of_channels):
                    continue

                score, is_detected = test_image(window, model_task1)

                if is_detected:
                    detections.append((x, y, x + window_size[0], y + window_size[1]))
                    scores.append(score)

        if len(detections) == 0:
            logger.info("No detection.")
            continue

        result = nms(tensor(detections, dtype=torch.float32), tensor(scores, dtype=torch.float32),
                     non_maximum_suppression_threshold)

        best_detections = result.numpy()

        for detection_index in best_detections:
            box = detections[detection_index]
            score = scores[detection_index]

            final_detections.append(box)
            final_scores.append(score)
            final_filenames.append(image_filenames[file_index])

            x_min, y_min, x_max, y_max = box

            window = color_image[y_min: y_max, x_min: x_max]

            _, name_index = test_image(window, model_task2)

            name_indexes.append(name_index)

            with_box = add_bounding_box(color_image.copy(), box)

        logger.info(f"Done. Time: {timeit.default_timer() - start_time}")

    return final_filenames, final_detections, final_scores, name_indexes
