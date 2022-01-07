train_directory = "../antrenare"
test_directory = "../validare/simpsons_validare"

data_task1_directory = "../data_task1"
data_task2_directory = "../data_task2"

positive_examples_directory = f"{data_task1_directory}/positive"
negative_examples_directory = f"{data_task1_directory}/negative"

solution_directory = "../evaluare/fisiere_solutie/Rusu_Andrei_331"

solution_directory_task_1 = f"{solution_directory}/task1"
solution_directory_task_2 = f"{solution_directory}/task2"

model_data_directory = "./models"

model_task1_path = f"{model_data_directory}/model_task1"
model_task2_path = f"{model_data_directory}/model_task2"

character_names = ["bart", "homer", "lisa", "marge", "unknown"]

image_width = 72
image_height = 72
image_number_of_channels = 3

non_maximum_suppression_threshold = 0.3

hyper_parameters = {
    'epochs': 10,
    'batch_size': 32,
    'momentum': 0.9,
    'learning_rate': 1e-2,
}
