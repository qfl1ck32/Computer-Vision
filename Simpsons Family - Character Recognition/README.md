# Character recognition (for Simpsons Family)

#### The app can be used to identify faces of characters in images from the Simpsons family and recognize their name.

#### The neural networks were trained to recognize Bart, Homer, Lisa and Marge.

# Dependencies

| **Dependency** | **Version** |
| -------------- | ----------- |
| numpy | 1.19.5 | 
| opencv-python | 4.5.3.56 |
| torch | 1.10.1+cu102 |
| torchvision | 0.11.2 |
| Pillow | 8.3.2 |

# How to run

### The neural networks have already been trained, and their weights are saved in ./models/.

### In order to run the application on a test dataset, just run `main.py`.

### In order to modify the `test_directory` or `solution_directory`, change the corresponding values in `constants.py`.
