import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torchvision.models.swin_transformer import SwinTransformer

def plot_emotion_barchart(scores, title='Emotion Scores for a Given Image'):
    """
    Plots a bar chart of emotion scores.

    Parameters:
    - scores (list or array): A list of scores corresponding to the emotions.
    - emotions (list): A list of emotion names.
    - title (str, optional): Title for the bar chart. Default is 'Emotion Scores for a Given Image'.

    Returns:
    - None
    """
    emotion_labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'gray']

    plt.bar(emotion_labels, scores, color=colors)
    plt.ylabel('Scores')
    plt.title(title)
    plt.show()

scores = [0.012, 0.009, -0.007, -0.003, 0.021, 0.0005, -0.010, 0.006]
plot_emotion_barchart(scores)

sys.exit()


def plot_emotion_barchart_plus_image(image, scores, title='Emotion Scores for a Given Image'):
    """
    Plots an image and a bar chart of emotion scores.

    Parameters:
    - image (numpy array): The image to display.
    - scores (list or array): A list of scores corresponding to the emotions.
    - title (str, optional): Title for the bar chart. Default is 'Emotion Scores for a Given Image'.

    Returns:
    - None
    """
    emotion_labels = ["0", "1", "2", "3", "4", "5", "6", "7"]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'gray']

    # Set up the figure with two subplots: one for the image, one for the bar chart
    plt.figure(figsize=(10, 5))

    # First subplot: the image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))  # Assuming image is a PyTorch tensor
    plt.title('Input Image')
    plt.axis('off')  # Do not show axis value

    # Second subplot: the bar chart
    plt.subplot(1, 2, 2)
    plt.bar(emotion_labels, scores, color=colors)
    plt.ylabel('Scores')
    plt.title(title)

    plt.tight_layout()  # Ensure proper spacing between the subplots
    # plt.show()


def plot_emotion_scores(matrix):
    """
    Displays a plot of parallel coordinates for emotion scores

    Parametri:
        matrix (list of lists): A list of lists where each internal list has length 8
			        and represents the emotion scores for a sample.
    """

    def matrix_to_data(m):
       """Transform a matrix into a DataFrame with emotion labels as columns and scores as values."""
        return pd.DataFrame(m, columns=[f'Emotion {i}' for i in range(8)])

    df = matrix_to_data(matrix)

    # Add a 'class' column to color the lines in the parallel coordinate plot.
    df['class'] = range(len(matrix))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(matrix)))  # Assegna un colore diverso a ciascun campione
    pd.plotting.parallel_coordinates(df, class_column='class', color=colors)
    plt.title("Emotion Scores Parallel Coordinates")
    plt.ylabel("Score")
    plt.xlabel("Emotion")
    # plt.show()

########################################################################################################################

""""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Supponiamo che window_size sia la dimensione della finestra e shift_size sia la dimensione dello spostamento
window_size = 7  # Esempio di dimensione della finestra
shift_size = window_size // 2  # Esempio di dimensione dello spostamento

# Load image
image_path = 'data/AFFECT_NET_8_LABELS/trial/train/images/23.jpg'
img = Image.open(image_path)
fig,ax = plt.subplots(1)

# Show image
ax.imshow(img)

# Plot shifted window
for x in range(0, img.width, window_size):
    for y in range(0, img.height, window_size):
        # Finestra regolare
        rect = patches.Rectangle((x, y), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Finestra spostata
        shift_rect = patches.Rectangle((x+shift_size, y+shift_size), window_size, window_size, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(shift_rect)

# plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Load the image
image_path = 'data/AFFECT_NET_8_LABELS/trial/train/images/23.jpg'  # Sostituisci con il percorso della tua immagine
image = Image.open(image_path)
np_image = np.array(image)

# Show the image
fig, ax = plt.subplots()
ax.imshow(np_image)

# Define patches
def draw_patches(image, scale, color, ax):
    patch_size = image.shape[0] // scale  # Calculate the size of the patches
    for y in range(0, image.shape[0], patch_size):
        for x in range(0, image.shape[1], patch_size):
            rect = patches.Rectangle((x, y), patch_size, patch_size,
                                     linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)


# Draw and save the images for different scales
scales = [8, 16, 32]  # Replace with the scales you want
colors = ['red', 'green', 'blue']  # Colors for the different scales

# First, display the original image
fig, ax = plt.subplots()
ax.imshow(np_image)
ax.set_title('Original Image')
ax.axis('off')
plt.show()

for scale, color in zip(scales, colors):
    fig, ax = plt.subplots()
    ax.imshow(np_image)
    draw_patches(np_image, scale, color, ax)

    # Set axis limits to avoid extra borders
    ax.set_xlim(0, np_image.shape[1])
    ax.set_ylim(np_image.shape[0], 0)

    # Hide the axes and add title with patch size and batch size
    ax.axis('off')
    patch_size = np_image.shape[0] // scale
    batch_size = scale * scale
    ax.set_title(
        f'Scale: {scale}, Patch Size: {patch_size}x{patch_size}')

    plt.savefig(f'grid_{scale}_{color}.png')  # Save the image
    plt.show()  # Show the image

    # Print the patch size as output
    print(f'Patch Size for Scale {scale}: {patch_size}x{patch_size}, Batch Size: {batch_size}')

"""

########################################################################################################################

import matplotlib.pyplot as plt
epochs = range(1, 16)

# LOSS

train_loss = []

val_loss = []

fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.plot(epochs, train_loss, 'ro-', label='Training Loss', markersize=6)
ax1.tick_params(axis='y', labelcolor='tab:red')

# Crea un secondo asse y che condivide lo stesso asse x
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color='tab:blue')
ax2.plot(epochs, val_loss, 'bo-', label='Validation Loss', markersize=6)
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Titolo e legenda
fig.suptitle('Training and Validation Loss', fontsize=14)
# Posizionamento della legenda in alto a destra
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta il layout

# Aggiusta i margini per assicurarsi che tutto sia visibile
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiunge spazio per il titolo

# plt.show()


# ACC

train_acc = []

val_acc = []

fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Accuracy', color='tab:red')
ax1.plot(epochs, train_acc, 'ro-', label='Training Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy', color='tab:blue')
ax2.plot(epochs, val_acc, 'bo-', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Titolo e legenda
fig.suptitle('Training and Validation Accuracy', fontsize=14)

# Posizionamento della legenda in basso a destra
fig.legend(loc='lower right', bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes)

fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta il layout

# plt.show()

########################################################################################################################

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
original_image = Image.open("data/AFFECT_NET_8_LABELS/trial/train/images/18.jpg").convert('L')  # Convert to grayscale

# Define the size of each patch
patch_size = 16  # 16x16 pixels

# Calculate the number of patches in each dimension
num_patches_x = original_image.width // patch_size
num_patches_y = original_image.height // patch_size

# Create a new image to visualize patches with some space between them (for example, 1 pixel)
patched_image = Image.new('L', (num_patches_x * (patch_size + 1) - 1, num_patches_y * (patch_size + 1) - 1))

# Extract patches and place them into the patched_image with a space of 1 pixel
for i in range(num_patches_x):
    for j in range(num_patches_y):
        patch = original_image.crop((i * patch_size, j * patch_size, (i + 1) * patch_size, (j + 1) * patch_size))
        patched_image.paste(patch, (i * (patch_size + 1), j * (patch_size + 1)))

# Visualize the original and patched images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(patched_image, cmap='gray')
axes[1].set_title('Image Patches')
axes[1].axis('off')

# plt.show()

########################################################################################################################

from PIL import Image

# Paths to your images
image_paths = ['data/Celeb-A/Images/img_celeba/img_celeba/000001.jpg', 'data/Celeb-A/Images/img_celeba/img_celeba/000002.jpg', 'data/Celeb-A/Images/img_celeba/img_celeba/000330.jpg', "data/Celeb-A/Images/img_celeba/img_celeba/000004.jpg", "data/Celeb-A/Images/img_celeba/img_celeba/000550.jpg", "data/Celeb-A/Images/img_celeba/img_celeba/000006.jpg", "data/Celeb-A/Images/img_celeba/img_celeba/000007.jpg", "data/Celeb-A/Images/img_celeba/img_celeba/000080.jpg", "data/Celeb-A/Images/img_celeba/img_celeba/000009.jpg"]

image_size = (100, 100)  # Change this to the desired size

# Number of images per row
images_per_row = 3

# Function to resize image with maintained aspect ratio
def resize_and_pad(img, size, pad_color=(0, 0, 0)):
    # Resize the image so that the longest dimension matches the target size
    img.thumbnail((size[0], size[1]), Image.ANTIALIAS)
    # Calculate padding size
    x_left = (size[0] - img.width) // 2
    x_right = size[0] - img.width - x_left
    y_top = (size[1] - img.height) // 2
    y_bottom = size[1] - img.height - y_top
    # Add padding to the image
    return ImageOps.expand(img, border=(x_left, y_top, x_right, y_bottom), fill=pad_color)

# Create a new image for the collage
num_rows = len(image_paths) // images_per_row + (len(image_paths) % images_per_row > 0)
final_image = Image.new('RGB', (image_size[0] * images_per_row, image_size[1] * num_rows), color=(255, 255, 255))

# Load images, resize, pad, and paste them into the final image
for index, path in enumerate(image_paths):
    img = Image.open(path)
    img = resize_and_pad(img, image_size)
    x = index % images_per_row * image_size[0]
    y = index // images_per_row * image_size[1]
    final_image.paste(img, (x, y))

# Save the final image
# final_image.save('collage.jpg')
# final_image.show()
