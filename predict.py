import sys
import tensorflow as tf

from train_model import load_image

def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model('model.h5')

def predict(model: tf.keras.Model, image_path: str) -> None:
    image = load_image(image_path)[None]
    prediction = model.predict(image)[0]
    print("Algae confidence: {:.2f}%".format(prediction[0] * 100))
    print("Clean confidence: {:.2f}%".format(prediction[1] * 100))

if __name__ == '__main__':
    model = load_model()
    predict(model, sys.argv[1])