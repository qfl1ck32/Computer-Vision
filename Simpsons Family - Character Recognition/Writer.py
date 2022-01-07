import numpy as np
from constants import *


def write_solution(file_names: np.array, detections: np.array, scores: np.array, name_indexes: np.array):
    with open(f"{solution_directory_task_1}/detections_all_faces.npy", "wb") as f:
        np.save(f, detections)

    with open(f"{solution_directory_task_1}/file_names_all_faces.npy", "wb") as f:
        np.save(f, file_names)

    with open(f"{solution_directory_task_1}/scores_all_faces.npy", "wb") as f:
        np.save(f, scores)

    detections_chars = {k: [] for k in character_names}
    scores_chars = {k: [] for k in character_names}
    file_names_chars = {k: [] for k in character_names}

    for file_name, detection, score, name_index in zip(file_names, detections, scores, name_indexes):
        name = character_names[name_index]

        if name == "unknown":
            continue

        detections_chars[name].append(detection)
        scores_chars[name].append(score)
        file_names_chars[name].append(file_name)

    for name_index in name_indexes:
        name = character_names[name_index]

        if name == "unknown":
            continue

        with open(f"{solution_directory_task_2}/detections_{name}.npy", "wb") as f:
            np.save(f, detections_chars[name])

        with open(f"{solution_directory_task_2}/file_names_{name}.npy", "wb") as f:
            np.save(f, file_names_chars[name])

        with open(f"{solution_directory_task_2}/scores_{name}.npy", "wb") as f:
            np.save(f, scores_chars[name])
