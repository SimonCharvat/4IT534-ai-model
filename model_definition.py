
from typing import Literal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def get_model(model_name: Literal["batch32_dropout", "ResNet50_v2"], num_of_classes):

    model = None

    match model_name:

        case "batch32_dropout": # Simple CNN
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                
                layers.Dense(num_of_classes, activation='softmax')  # Number of classes
            ])

            # Compile the model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy', # Used sparse_categorical_crossentropy because there is large amount of classes
                metrics=['accuracy']
            )
    

        case "ResNet50_v2":
            # Load Pre-trained Model (ResNet50)
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            base_model.trainable = False  # Freeze the base model (initially)

            # Create New Model on Top
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(num_of_classes, activation='softmax')
            ])

            # Compile the Model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        case _:
            raise ValueError(f"Unknown model name: {model_name}")


    return model_name, model
