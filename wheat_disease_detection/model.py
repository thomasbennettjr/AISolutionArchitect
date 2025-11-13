"""
CNN Model for Wheat Disease Detection from Aerial Photos
Supports multiple disease classes including healthy wheat
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
import numpy as np


class WheatDiseaseCNN:
    """CNN Model for wheat disease classification"""

    def __init__(self, input_shape=(224, 224, 3), num_classes=5, model_type='custom'):
        """
        Initialize the wheat disease CNN model

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of disease classes to classify
            model_type: 'custom', 'resnet50', 'mobilenet', or 'efficientnet'
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None

    def build_custom_cnn(self):
        """Build a custom CNN architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def build_transfer_learning_model(self, base_model_name='resnet50'):
        """Build a model using transfer learning"""
        # Load pre-trained base model
        if base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False,
                                  input_shape=self.input_shape)
        elif base_model_name == 'mobilenet':
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                     input_shape=self.input_shape)
        elif base_model_name == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                        input_shape=self.input_shape)
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")

        # Freeze base model layers
        base_model.trainable = False

        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model, base_model

    def build_model(self):
        """Build the model based on specified type"""
        if self.model_type == 'custom':
            self.model = self.build_custom_cnn()
            print("Custom CNN model built successfully")
        else:
            self.model, self.base_model = self.build_transfer_learning_model(
                self.model_type)
            print(f"Transfer learning model built with {self.model_type} base")

        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer, loss, and metrics"""
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

        print("Model compiled successfully")
        return self.model

    def unfreeze_base_model(self, layers_to_unfreeze=30):
        """Unfreeze top layers of base model for fine-tuning"""
        if self.model_type == 'custom':
            print("Cannot unfreeze base model for custom CNN")
            return

        self.base_model.trainable = True

        # Freeze all layers except the top ones
        for layer in self.base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False

        print(f"Unfroze top {layers_to_unfreeze} layers for fine-tuning")

    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()

        return self.model.summary()

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError(
                "No model to save. Build and train the model first.")

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


if __name__ == "__main__":
    # Example usage
    print("Building Custom CNN model...")
    custom_model = WheatDiseaseCNN(num_classes=5, model_type='custom')
    custom_model.build_model()
    custom_model.get_model_summary()

    print("\n" + "="*80 + "\n")

    print("Building Transfer Learning model with ResNet50...")
    transfer_model = WheatDiseaseCNN(num_classes=5, model_type='resnet50')
    transfer_model.build_model()
    transfer_model.get_model_summary()
