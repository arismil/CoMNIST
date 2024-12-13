import tensorflow.keras.backend as K
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import api.model

SIZE = 32

INF = 10**9

# Load Dataset
data_dir = pathlib.Path("images/Cyrillic")
image_count = len(list(data_dir.glob("*/*.png")))
print(image_count)
batch_size = 32
img_height = 32
img_width = 32

cyrillic_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = cyrillic_dataset.class_names
print(class_names)

# # Load Model
# model = api.model.load_model(weight="weights/comnist_keras_ru.hdf5", nb_classes=34)

# LETTERS = "IАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

# print("learning rate:", K.eval(model.optimizer.learning_rate))
