import tensorflow as tf

DATASET_DIR = "dataset"
MODEL_PATH = "models/waste_classifier.keras"

ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

model = tf.keras.models.load_model(MODEL_PATH)
loss, accuracy = model.evaluate(ds)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
