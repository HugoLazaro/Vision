import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchsummary import summary

import numpy as np
import time
import os
import random

import matplotlib.pyplot as plt
from matplotlib import colors

from nn.unet import UNet
from dataset.brain_image_set import BrainImageSet

# Set the directory paths for the training and test data
TRAIN_DATA_PATH = './Task01_BrainTumour_2D/training_images/'
TEST_DATA_PATH = './Task01_BrainTumour_2D/test_images/'
TRAIN_LABELS_PATH = './Task01_BrainTumour_2D/training_labels/'
TEST_LABELS_PATH = './Task01_BrainTumour_2D/test_labels/'

# # Download the dataset
# import tarfile
# # !wget https://www.dropbox.com/s/zmytk2yu284af6t/Task01_BrainTumour_2D.tar.gz

# # Unzip the '.tar.gz' file to the current directory
# datafile = tarfile.open('Task01_BrainTumour_2D.tar.gz')
# datafile.extractall()
# datafile.close()

# def visualize_dataset(num_images):
#     # Get a list of randomly selected training and test image filenames
#     train_image_filenames = random.sample(os.listdir(TRAIN_DATA_PATH), num_images)
#     test_image_filenames = random.sample(os.listdir(TEST_DATA_PATH), num_images)

#     # Plot the selected training and test images in a grid
#     fig, axs = plt.subplots(2, num_images, figsize=(num_images*3, 6))
#     for i in range(num_images):
#         # Plot the training image
#         train_img = plt.imread(TRAIN_DATA_PATH + train_image_filenames[i])
#         axs[0, i].imshow(train_img, cmap='gray')
#         axs[0, i].axis('off')
#         axs[0, i].set_title('Train Image {}'.format(i+1))
        
#         # Plot the test image
#         test_img = plt.imread(TEST_DATA_PATH + test_image_filenames[i])
#         axs[1, i].imshow(test_img, cmap='gray')
#         axs[1, i].axis('off')
#         axs[1, i].set_title('Test Image {}'.format(i+1))
        
#     plt.tight_layout()
#     plt.show()


# # Set the number of images you want to visualize
# num_images = 2

# visualize_dataset(num_images)

# CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))

# Build the model
"""
- 0: background
- 1: edema
- 2: non-enhancing tumour
- 3: enhancing tumour
"""
num_class = 4

model = UNet(input_channel=1, output_channel=num_class, num_filter=16)
model = model.to(device)
params = list(model.parameters())

summary(model,(1,128,128))

model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Adam Optimizer
### Insert your code ###
optimizer = optim.Adam(params, lr=0.001)
### End of your code ###

# Segmentation loss. Criterion: Cross Entropy Loss.
### Insert your code ###
criterion = nn.CrossEntropyLoss()
### End of your code ###

# Datasets
print("Creating training set")
train_set = BrainImageSet(TRAIN_DATA_PATH, TRAIN_LABELS_PATH)
print("Creating test set")
test_set = BrainImageSet(TEST_DATA_PATH, TEST_LABELS_PATH)

# Train the model
# Note: when you debug the model, you may reduce the number of iterations or batch size to save time.
num_iter = 10000
train_batch_size = 16
eval_batch_size = 16

start = time.time()
for it in range(1, 1 + num_iter):
    # Set the modules in training mode, which will have effects on certain modules, e.g. dropout or batchnorm.
    start_iter = time.time()
    model.train()

    # Get a random batch of images and labels
    ### Insert your code ###
    images, labels = train_set.get_random_batch(train_batch_size)
    ### End of your code ###
    
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
    logits = model(images)
    
    # Note that optimizer.zero_grad() is equivalent to net.zero_grad() if it optimises all the net parameters.
    optimizer.zero_grad()

    # Perform optimisation: compute the loss, backpropagation, and perform a step of your optimizer
    ### Insert your code ###
    output = criterion(logits, labels)
    output.backward()
    optimizer.step()

    ### End of your code ###

    print('--- Iteration {0}: Training loss = {1:.4f}, {2:.4f} s ---'.format(it, output.item(), time.time() - start_iter))

    # Evaluate
    if it % 10 == 0:
        model.eval()
        # Disabling gradient calculation during inference to reduce memory consumption
        with torch.no_grad():
            images, labels = test_set.get_random_batch(eval_batch_size)
            images, labels = torch.from_numpy(images), torch.from_numpy(labels)
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            logits = model(images)
            loss = criterion(logits, labels)
            print('--- Iteration {0}: Test loss = {1:.4f} ---\n'.format(it, loss.item()))

    # Save the model
    if it % 5000 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_{0}.pt'.format(it)))
        
print('Training took {:.3f}s in total.'.format(time.time() - start))