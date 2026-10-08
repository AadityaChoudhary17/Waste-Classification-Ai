import sys
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/waste_classifier.keras"
IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

if len(sys.argv) != 2:
    print("Usage: python src/predict.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(MODEL_PATH):
    print("Model not found. Train the model first using:")
    print("python src/train.py")
    sys.exit(1)

if not os.path.exists(image_path):
    print("Image not found:", image_path)
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

image = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)[0]
index = int(np.argmax(predictions))

print("Prediction:", CLASS_NAMES[index])
print("Confidence: {:.2f}%".format(predictions[index] * 100))
