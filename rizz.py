import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def load_images_and_masks(root_folder, size=(256, 256)):
    images = []
    masks = []
    tile_folders = [f for f in os.listdir(root_folder) if f.startswith('Tile')]
    for tile_folder in tile_folders:
        images_folder = os.path.join(root_folder, tile_folder, 'images')
        masks_folder = os.path.join(root_folder, tile_folder, 'masks')
        
        for img_path in glob(os.path.join(images_folder, '*.jpg')):
            mask_path = img_path.replace(images_folder, masks_folder).replace('.jpg', '.png')
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    img = cv2.resize(img, size)
                    mask = cv2.resize(mask, size)
                    images.append(img)
                    masks.append(mask)
                else:
                    print(f'Failed to load image or mask for: {img_path}')
            else:
                print(f'File does not exist: {img_path} or {mask_path}')
                
    return np.array(images), np.array(masks)

def unet_model(input_size=(256, 256, 3), learning_rate=0.001):
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs, conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def extract_features(segmented_images):
    features = []
    for img in segmented_images:
        features.append(img.mean(axis=(0, 1))) 
    return np.array(features)

def build_coordinate_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='linear'))  

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


root_folder = '/project/workspace/mainfolder/Semantic segmentation dataset' 
images, masks = load_images_and_masks(root_folder)
images = images / 255.0  
masks = masks / 255.0  


print(f'Loaded {len(images)} images and {len(masks)} masks')


if len(images) > 0:
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)


    input_shape = (256, 256, 3)
    model = unet_model(input_shape, learning_rate=0.0001)
    model.summary()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))


    y_pred = model.predict(X_val)


    features = extract_features(y_pred)


    coordinate_model = build_coordinate_model((features.shape[1],))

    
    predicted_3d_coords = coordinate_model.predict(features)


    df = pd.DataFrame(predicted_3d_coords, columns=['x', 'y', 'z'])
    df.to_csv('predicted_3d_coords.csv', index=False)
else:
    print("No images or masks were loaded. Please check the directory structure and paths.")
