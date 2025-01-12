# Plant Recognition AI - CNN Model
This repository contains a convolutional neural network (CNN) model trained to recognize plants from images. The model can identify the plant species and detect if the plant is healthy or affected by a disease. The training was conducted using a Kaggle dataset.

## Project Structure
```
.
├── model_definition.py          # Model architecture definition
├── similarity_matrix.py         # Defined similarity lookup matrix
├── training_notebook.ipynb      # Jupyter notebook for model training
└── predict
    ├── labels.py                # Plant class labels and diseases
    ├── requirements.txt         # Required libraries to predict using the trained model
    ├── predict_using_model.py   # Script for predicting using the trained model
    ├── model.keras.part1        # Model (split part 1)
    ├── model.keras.part2        # Model (split part 2)
    └── model.keras.part3        # Model (split part 3)
```

## Description
- **Training**:
  - The CNN model is trained in the `training_notebook.ipynb` file.
  - The model's architecture is defined in `model_definition.py` and imported into the training notebook to keep the training environment clean and organized.

- **Prediction**:
  - The `predict` folder contains the final trained model, split into three parts to comply with GitHub's 100 MB file size limit.
  - `predict_using_model.py` provides a Python class with a `predict` function.
  - The `predict` function takes the path to an image (JPG format) as input and returns:
    - **Plant name**
    - **Boolean value** indicating if the plant is healthy
    - **Disease name** (if the plant is not healthy)

- **Labels**:
  - Class labels and disease names are stored in `labels.py` as a list of tuples. The model's output class ID can be matched to its corresponding label through this file.

- **Similarity**:
  - File `similarity_matrix.py` contains precalculated matrix (numpy array) which can be used as a lookup table to get similarity score between predicted class and actual class.
  - File `similarity_matrix.py` also contains code to generate (print) the matrix based on class labels, which can be used to easily change the similarity matrix values.

## Usage

### Training the Model
To retrain the model:
1. Open and run the `training_notebook.ipynb` file.
2. The model architecture is automatically imported from `model_definition.py`.

### Predicting with the Model
The `predict` folder contains all necessary files to make predictions.

```python
from predict_using_model import Model

model = Model()
result = model.predict("image_example.jpg")

# Example Output: ('Peach', False, 'Bacterial spot')
```

## Dependencies

### Full Dependencies (for Training)
To install all necessary packages for training the AI model, run the following command:
```bash
pip install kagglehub matplotlib tensorflow numpy pandas sklearn
```

### Minimal Dependencies (for Prediction Only)
If you only need to use the pre-trained model for predictions, install the following packages:
```bash
pip install numpy tensorflow
```
Alternatively, you can install dependencies from the `requirements.txt` file:
```bash
pip install -r predict\requirements.txt
```


## Acknowledgments
- Training dataset: [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)


