from ExamplesGenerator import generate_positive_examples, generate_negative_examples
from NeuralNetwork import *
from FaceDetector import run_face_detector
from Writer import write_solution
from Logger import logger


def main():
    if not os.listdir(positive_examples_directory):
        logger.info('Generating positive examples...')
        generate_positive_examples()
        logger.info('Done.')

    if not os.listdir(negative_examples_directory):
        logger.info('Generating negative examples...')
        generate_negative_examples()
        logger.info('Done.')

    logger.info('Training CNN #1...')
    model_task1 = train_neural_network(1)
    logger.info('Done.')

    logger.info('Training CNN #2...')
    model_task2 = train_neural_network(2)
    logger.info('Done.')

    logger.info('Running the face detector...')
    file_names, detections, scores, name_indexes = run_face_detector(model_task1, model_task2)
    logger.info('Done.')

    logger.info('Writing the solution...')
    write_solution(file_names, detections, scores, name_indexes)
    logger.info('Done.')


if __name__ == '__main__':
    main()
