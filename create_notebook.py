import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """\
# Image Classification Model Training
This notebook demonstrates the end-to-end process of training the Custom CNN for image classification."""

code1 = """\
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
"""

text2 = """\
## Dataset Preparation and Loading"""

code2 = """\
train_dir = '../dataset/train'
val_dir = '../dataset/val'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(32, 32), batch_size=64, class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(32, 32), batch_size=64, class_mode='categorical'
)
"""

text3 = """\
## Define and Train Model"""

code3 = """\
import sys
sys.path.append('../src')
from model import build_custom_cnn_a

model = build_custom_cnn_a()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train (Keep epochs low for demo)
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator
)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3)
]

with open('notebooks/Model_Training.ipynb', 'w') as f:
    nbf.write(nb, f)
