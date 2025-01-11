
from labels import labels
from typing import Tuple
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Disable oneDNN optimizations to avoid slight variations in numerical results
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class Model:
    def __init__(self):
        
        self.model_path = "model.keras"
        
        # If model does not exist, create it from parts
        if not os.path.exists(self.model_path):
            print("Creating model from parts...")
            self.merge_file(self.model_path, [
                ".\model_parts\model.keras.part1",
                ".\model_parts\model.keras.part2",
                ".\model_parts\model.keras.part3"
            ])
        
        # Load the trained model
        self.model = load_model(self.model_path)
    
    def predict(self, image_path) -> Tuple[str, bool, str]:
        """
        Args:
            image_path (str): The file path to the image in JPG format.

        Returns:
            Tuple[str, bool, str]: A tuple containing:
                - Plant name (str)
                - Boolean value indicating if the plant is healthy (bool)
                - Disease name (str) if the plant is not healthy, otherwise "healthy"
        """

        prepared_image = self.prepare_image(image_path) # Prepare the image for prediction
        # Přepněte model do inference režimu (pro jistotu deaktivace dropoutu a dalších náhodných operací)
        self.model.trainable = False  # Deaktivuje případné náhodné chování při inference
        predictions = self.model.predict(prepared_image) # Predict the label

        # Get the class with the highest probability
        predicted_class_number = np.argmax(predictions, axis=1)[0]
        predicted_class_info = labels[predicted_class_number]

        # Print the predicted class (you can map this to the actual class name if you have a label map)
        print(f"Predicted class {predicted_class_number}: {predicted_class_info}")
        return predicted_class_info


    # Merge multiple files into a single file (creates model file from parts)
    def merge_file(self, output_path, part_paths):
        with open(output_path, 'wb') as output:
            for part in part_paths:
                with open(part, 'rb') as part_file:
                    print("writing", part)
                    output.write(part_file.read())

    # Function to load and preprocess an image
    def prepare_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224)) # Load the image with target size (224, 224)
        img_array = image.img_to_array(img) # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0) # Add an extra dimension (batch size) to the image array (batch size of 1)
        img_array = img_array / 255.0  # Normalize the image to [0, 1]
        return img_array


# Provide the path to the image you want to predict
if __name__ == "__main__":
    model = Model()
    model.predict("image_example.jpg")
