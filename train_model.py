from glob import glob
import multiprocessing as mp
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

def load_image(filename: str) -> np.ndarray:
    image = Image.open(filename)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    return image

def load_images(filenames: List[str]) -> np.ndarray:
    with mp.Pool() as pool:
        images = list(tqdm(pool.imap(load_image, filenames), total=len(filenames)))
    return np.array(images)

def load_model() -> tf.keras.Model:
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet")
    model.trainable = False
    return model

def add_top_to_model(model: tf.keras.Model) -> tf.keras.Model:
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs=model.input, outputs=x)

def main():
    model = load_model()
    model = add_top_to_model(model)
    model.build((None, 224, 224, 3))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    soccerball_filenames = glob("home/oem/soccer-balls-blender*")
    algae_images = load_images(soccerball_filenames)

    algae_labels = np.zeros(len(algae_images))

    images = np.concatenate([algae_images], axis=0)
    labels = np.concatenate([algae_labels], axis=0)

    print(images.shape, labels.shape)

    model.fit(images, labels, epochs=1, batch_size=128, shuffle=True, validation_split=0.1)
    model.save("model.h5")

if __name__ == "__main__":
    main()