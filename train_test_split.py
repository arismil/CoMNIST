import os
import shutil
import random
from pathlib import Path

# Set the seed for reproducibility
random.seed(123)

# Define the paths
data_dir = Path("images/Cyrillic")
train_dir = Path("images/Cyrillic_train")
test_dir = Path("images/Cyrillic_test")

# Create directories if they don't exist
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Define the split ratio
split_ratio = 0.7

# Iterate over each class directory
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        # Create class directories in train and test directories
        (train_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_dir.name).mkdir(parents=True, exist_ok=True)

        # Get all image files in the class directory
        images = list(class_dir.glob("*.png"))


        # Split the images
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Move the images to the respective directories
        for img in train_images:
            shutil.copy(str(img), str(train_dir / class_dir.name / img.name))

        for img in test_images:
            shutil.copy(str(img), str(test_dir / class_dir.name / img.name))

print("Images have been split into training and test sets.")
