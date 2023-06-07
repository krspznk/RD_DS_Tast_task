import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import pandas as pd
import numpy as np


# Define the U-Net architecture
def unet_model():
    inputs = Input(shape=(768, 768, 3))

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = Conv2D(128, 2, activation='relu', padding='same')(up1)
    merge1 = Concatenate()([conv2, up1])

    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(64, 2, activation='relu', padding='same')(up2)
    merge2 = Concatenate()([conv1, up2])

    # Output
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define the dice loss function
def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_score


# Load and preprocess the training data
annotations_df = pd.read_csv('../data/train_ship_segmentations.csv')


def load_and_preprocess_data():
    # Load images from train_v2 folder
    image_ids = annotations_df['ImageId'].unique()
    train_images = []
    train_masks = []
    for image_id in image_ids:
        image_path = f'../data/train_v2/{image_id}'
        mask_path = f'../data/train_v2_masks/{image_id}'
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(768, 768))
        mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(768, 768), color_mode='grayscale')
        train_images.append(tf.keras.preprocessing.image.img_to_array(image))
        train_masks.append(tf.keras.preprocessing.image.img_to_array(mask) / 255.0)  # Normalize masks

    train_images = np.array(train_images)
    train_masks = np.array(train_masks)

    return train_images, train_masks


train_images, train_masks = load_and_preprocess_data()

# Create an instance of the U-Net model
model = unet_model()

# Compile the model with the dice loss and appropriate metrics
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Train the model
model.fit(train_images, train_masks, epochs=10, batch_size=32)

# Save the trained model weights for later use
model.save_weights('../models/trained_model_weights.h5')
