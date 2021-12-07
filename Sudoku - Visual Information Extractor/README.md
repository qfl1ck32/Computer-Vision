# Visual information extraction from images with Sudoku games

#### The app can be used to extract data about a Sudoku game, given an image of the paper.
#### Digits will be identified, and empty spaces will be noted with 'o'.

### It works for two types of Sudokus:
  1. Classic
  2. Jigsaw (both color & grayscale)
    

# Dependencies


1. [numpy@1.19.5](https://numpy.org/install/)

2. [opencv-python@4.5.3.56](https://pypi.org/project/opencv-python/)

3. [scikit-image@0.18.3](https://pypi.org/project/scikit-image/)

4. [imutils@0.5.4](https://pypi.org/project/imutils/)


# How to run

### The application is using python's `ArgumentParser`, so it easy to understand the usage:

```
usage: main.py [-h] [--output OUTPUT] path {classic,jigsaw}

positional arguments:
  path              The path to the images
  {classic,jigsaw}  The type of sudoku

optional arguments:
  -h, --help        show this help message and exit
  --output OUTPUT   Where to save the answers
  ```
  
If there is no output specificied, by default, it'll be in `/evaluare/fisiere_solutie/Rusu_Andrei_331/`.
