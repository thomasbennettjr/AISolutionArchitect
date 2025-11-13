"""
Prediction/Inference script for Wheat Disease Detection
Load trained model and make predictions on new images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import json


class WheatDiseasePredictor:
    """Predictor class for wheat disease detection"""

    def __init__(self, model_path, class_names=None, img_size=(224, 224)):
        """
        Initialize the predictor

        Args:
            model_path: Path to the trained model file
            class_names: List of class names (optional, will try to load from JSON)
            img_size: Image size for preprocessing
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.class_names = class_names

        self.load_model()
        self.load_class_names()

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def load_class_names(self):
        """Load class names from JSON file if available"""
        if self.class_names is not None:
            return

        # Try to load from JSON file in the same directory as model
        model_dir = os.path.dirname(self.model_path)
        class_names_path = os.path.join(model_dir, 'class_names.json')

        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded class names: {self.class_names}")
        else:
            # Default class names for wheat diseases
            self.class_names = [
                'Healthy',
                'Leaf Rust',
                'Stem Rust',
                'Yellow Rust',
                'Septoria'
            ]
            print(f"Using default class names: {self.class_names}")

    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction

        Args:
            image_path: Path to image file or PIL Image object

        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.convert('RGB')

        # Resize
        img = img.resize(self.img_size)

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, img

    def predict(self, image_path, top_k=3):
        """
        Make prediction on a single image

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_classes = [self.class_names[i] for i in top_indices]
        top_probs = [float(predictions[i]) for i in top_indices]

        # Prepare results
        results = {
            'predicted_class': top_classes[0],
            'confidence': top_probs[0],
            'top_predictions': [
                {'class': cls, 'confidence': prob}
                for cls, prob in zip(top_classes, top_probs)
            ],
            'all_probabilities': {
                cls: float(predictions[i])
                for i, cls in enumerate(self.class_names)
            }
        }

        return results, original_img

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []
        for img_path in image_paths:
            try:
                result, _ = self.predict(img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })

        return results

    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and bar chart

        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        results, original_img = self.predict(image_path)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Display image
        axes[0].imshow(original_img)
        axes[0].axis('off')
        axes[0].set_title(f"Predicted: {results['predicted_class']}\n"
                          f"Confidence: {results['confidence']:.2%}",
                          fontsize=14, fontweight='bold')

        # Display probabilities
        classes = list(results['all_probabilities'].keys())
        probs = list(results['all_probabilities'].values())
        colors = ['green' if p == max(probs) else 'skyblue' for p in probs]

        axes[1].barh(classes, probs, color=colors)
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title('Disease Probabilities',
                          fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])

        # Add percentage labels
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            axes[1].text(prob + 0.02, i, f'{prob:.1%}',
                         va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

        return results

    def predict_from_directory(self, directory_path, recursive=False):
        """
        Make predictions on all images in a directory

        Args:
            directory_path: Path to directory containing images
            recursive: Whether to search subdirectories

        Returns:
            List of prediction results
        """
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        # Find all images
        image_paths = []
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(directory_path, file))

        print(f"Found {len(image_paths)} images")

        # Make predictions
        results = self.predict_batch(image_paths)

        return results

    def save_predictions_to_csv(self, results, output_path='predictions.csv'):
        """Save prediction results to CSV file"""
        import pandas as pd

        # Flatten results for CSV
        rows = []
        for result in results:
            if 'error' in result:
                rows.append({
                    'image_path': result['image_path'],
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'error': result['error']
                })
            else:
                row = {
                    'image_path': result.get('image_path', 'N/A'),
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence']
                }
                # Add all class probabilities
                for cls, prob in result['all_probabilities'].items():
                    row[f'prob_{cls}'] = prob
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


def save_class_names(class_names, output_path='models/class_names.json'):
    """Save class names to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to {output_path}")


if __name__ == "__main__":
    # Example usage

    # Path to your trained model
    MODEL_PATH = "models/best_wheat_disease_model.h5"

    # Class names (should match your training data)
    CLASS_NAMES = [
        'Healthy',
        'Leaf Rust',
        'Stem Rust',
        'Yellow Rust',
        'Septoria'
    ]

    # Save class names for future use
    save_class_names(CLASS_NAMES)

    # Initialize predictor
    predictor = WheatDiseasePredictor(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )

    # Single image prediction
    # image_path = "path/to/your/wheat/image.jpg"
    # results = predictor.visualize_prediction(image_path)
    # print(json.dumps(results, indent=2))

    # Batch prediction from directory
    # results = predictor.predict_from_directory("path/to/images/directory")
    # predictor.save_predictions_to_csv(results, "wheat_predictions.csv")

    print("Predictor ready! Update the paths above to make predictions.")
