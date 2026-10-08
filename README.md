# AI-Based Waste Classification using CNN

A beginner-friendly Deep Learning project that classifies waste images into categories using a Convolutional Neural Network (CNN).

## Classes
The project expects these folders inside `dataset/`:
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Requirements
Python 3.9+ is recommended.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
Use a labeled waste-image dataset such as Garbage Classification and arrange it like:

dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/

You can also use your own images.

## Train
```bash
python src/train.py
```

The trained model is saved as:
`models/waste_classifier.keras`

## Test a Single Image
```bash
python src/predict.py path/to/image.jpg
```

## Project Features
- Image preprocessing
- CNN model
- Data augmentation
- Training/validation split
- Accuracy and loss graphs
- Single-image prediction
- Saved trained model

## Technologies
Python, TensorFlow/Keras, NumPy, Matplotlib, Pillow.

## Note
The ZIP contains the complete project code but not the image dataset or trained model, because those files can be large.
