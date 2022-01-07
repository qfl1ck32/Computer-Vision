import cv2
import torch
import os

import numpy as np

from PIL import Image

from torch.hub import load
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, Linear
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize, RandomRotation,\
    RandomAffine

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import save, load, no_grad, Tensor, max

from typing import List

from constants import *

from EarlyStopping import EarlyStopping

from Logger import logger


def train(**kwargs):
    """
    Function that trains a model on a dataset
    """

    # define the loss function (log loss)
    criterion: CrossEntropyLoss = CrossEntropyLoss()

    # create variables for the model and the data_size
    # in order not to access them via kwargs every other time
    model: resnet18 = kwargs['model']
    data_size: int = len(kwargs['data_loader'])

    # define the optimizer - stochastic gradient descent
    optimizer: SGD = SGD(params=model.parameters(),
                         lr=kwargs['learning_rate'],
                         momentum=kwargs['momentum'])

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1)

    early_stopping = EarlyStopping(
        patience=3,
    )

    # type hinting for future variables
    data: List[Tensor]
    images: Tensor
    labels: Tensor
    output: Tensor
    predicted: Tensor
    loss: Tensor

    best_accuracy = 0
    best_loss = float('+inf')

    last_loss = 0
    last_accuracy = 0

    # run a loop for each epoch
    for epoch in range(kwargs['epochs']):

        # define variables for metrics
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        # for every batch in the data loader
        for data in kwargs['data_loader']:
            # take the images, the labels, and send them to GPU
            images, labels = data[0].cuda(), data[1].cuda()

            # reset the gradients back to zero
            # since, by default, PyTorch accumulates them on every backward pass
            optimizer.zero_grad()

            # run the model on the current images (forward pass)
            output = model(images)

            # backward pass
            loss = criterion(output, labels)
            loss.backward()

            # optimize, by iterating over all the tensors
            # and use the internal "grad" parameter to update the values
            optimizer.step()

            # get the predicted labels
            _, predicted = max(output.data, 1)

            # accumulate to correct, total (for accuracy)
            correct += float((predicted == labels).sum())
            total += labels.size()[0]

            # accumulate to running_loss (which is the loss for the current epoch)
            running_loss += float(loss)

        # get loss and accuracy for the validation dataset
        val_loss, val_acc = get_validation_loss_acc(model=model,
                                                    data_loader=kwargs['validation_data'])

        if val_acc > best_accuracy:
            best_accuracy = val_acc

        if val_loss < best_loss:
            best_loss = val_loss

        last_loss = val_loss
        last_accuracy = val_acc

        logger.info('Epoch %d | loss: %.5f | acc: %.5f | val_loss: %.5f | val_acc: %.5f' % (
            epoch + 1, running_loss / data_size, correct / total, val_loss, val_acc))

        early_stopping(val_loss, model)

        # step on the scheduler
        scheduler.step(val_loss)

        if early_stopping.early_stop:
            logger.info('Early stopping...')
            break

        # save(model.state_dict(), f"{kwargs['save_model_path']}_{epoch}")
    return best_loss, best_accuracy, last_loss, last_accuracy


def get_validation_loss_acc(**kwargs):
    """
    Function that calculates the loss and accuracy on a dataset.
    Similar to train(**kwargs).
    """

    correct: int = 0
    total: int = 0

    loss: float = 0.0

    criterion: CrossEntropyLoss = CrossEntropyLoss()

    data: List[Tensor]
    images: Tensor
    labels: Tensor
    outputs: Tensor
    predicted: Tensor

    model: resnet18 = kwargs['model']

    # deactivate the autograd engine, since we won't use backpropagation.
    # it speeds up computations and reduces memory usage.
    with no_grad():
        for data in kwargs['data_loader']:
            images, labels = data[0].cuda(), data[1].cuda()

            outputs = model(images)

            _, predicted = max(outputs.data, 1)

            correct += int((predicted == labels).sum())
            total += labels.size()[0]

            current_loss = criterion(outputs, labels)

            loss += float(current_loss)

    return loss / len(kwargs['data_loader']), correct / total


def get_dataloader(dataset: ImageFolder, shuffle: bool = True) -> DataLoader:
    """
    Return a DataLoader object for a given ImageFolder instance.
    """

    return DataLoader(dataset=dataset,
                      batch_size=hyper_parameters['batch_size'],
                      shuffle=shuffle,
                      num_workers=4,
                      pin_memory=True)


def test_image(image: np.ndarray, model: resnet18):
    img = Image.fromarray(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB), mode='RGB')

    loader = Compose([
        Resize((image_height, image_width)),
        ToTensor()
    ])

    img = loader(img).float().unsqueeze_(0).to('cuda')

    with no_grad():
        outputs = model(img)

        score, prediction = torch.max(outputs, 1)

        return score.item(), prediction.item()


def train_neural_network(for_task: int):
    save_model_path = model_task1_path if for_task == 1 else model_task2_path

    # define the transformations that will occur on every other batch
    transform: Compose = Compose([
        Resize((image_height, image_width)),

        RandomHorizontalFlip(),
        RandomRotation(degrees=10),
        RandomAffine(degrees=0.2),

        ToTensor(),
    ])

    # load the dataset
    train_data: ImageFolder = ImageFolder(root=data_task1_directory if for_task == 1 else data_task2_directory,
                                          transform=transform)

    # get the DataLoader instance
    train_data_loader: DataLoader = get_dataloader(dataset=train_data)

    # declare the model
    model: resnet18 = resnet18(pretrained=False)

    # since the model is trained on Imagenet, it'll have an output of 1000 classes.
    # modify the fully connected layer to output 2 / 4 classes.
    model.fc = Linear(model.fc.in_features, 2 if for_task == 1 else 5)

    # send the model to the GPU
    model.cuda()

    if os.path.exists(save_model_path):
        model.load_state_dict(load(save_model_path))

        model.eval()

        return model

    # train the model
    train(
        data_loader=train_data_loader,
        epochs=hyper_parameters['epochs'],
        learning_rate=hyper_parameters['learning_rate'],
        momentum=hyper_parameters['momentum'],
        batch_size=hyper_parameters['batch_size'],
        validation_data=train_data_loader,
        model=model,
        save_model_path=save_model_path)

    save(model.state_dict(), save_model_path)

    model.eval()

    return model
