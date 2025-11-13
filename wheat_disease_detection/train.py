"""
Training script for Wheat Disease Detection CNN
Handles data loading, augmentation, training, and evaluation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
from model import WheatDiseaseCNN

# Configure TensorFlow for better stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

# Configure threading
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Configure GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration warning: {e}")


class WheatDiseaseTrainer:
    """Trainer class for wheat disease detection model"""

    def __init__(self, data_dir, img_size=(224, 224), batch_size=32,
                 model_type='custom', num_classes=5):
        """
        Initialize the trainer

        Args:
            data_dir: Root directory containing train/val/test folders
            img_size: Size to resize images to
            batch_size: Training batch size
            model_type: Type of model to use
            num_classes: Number of disease classes
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_wrapper = None
        self.model = None
        self.history = None
        self.class_names = None

    def create_data_generators(self, validation_split=0.2):
        """Create data generators with augmentation"""

        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        # Validation/Test data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        # Check if separate train/val/test folders exist
        if os.path.exists(train_dir):
            self.train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )

            self.val_generator = val_datagen.flow_from_directory(
                train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
        else:
            # Use the main data directory
            self.train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )

            self.val_generator = val_datagen.flow_from_directory(
                self.data_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )

        # Test generator if test directory exists
        if os.path.exists(test_dir):
            self.test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        else:
            self.test_generator = None

        # Store class names
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        if self.test_generator:
            print(f"Test samples: {self.test_generator.samples}")

        return self.train_generator, self.val_generator, self.test_generator

    def setup_callbacks(self, model_save_path='models/best_model.h5'):
        """Setup training callbacks"""

        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]

        return callbacks

    def train(self, epochs=50, learning_rate=0.001,
              model_save_path='models/best_wheat_disease_model.h5'):
        """Train the model"""

        print(f"\n{'='*80}")
        print("Starting training...")
        print(f"{'='*80}\n")

        # Create data generators
        self.create_data_generators()

        # Build and compile model
        self.model_wrapper = WheatDiseaseCNN(
            input_shape=(*self.img_size, 3),
            num_classes=self.num_classes,
            model_type=self.model_type
        )
        self.model = self.model_wrapper.compile_model(
            learning_rate=learning_rate)

        # Setup callbacks
        callbacks = self.setup_callbacks(model_save_path)

        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )

        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"{'='*80}\n")

        return self.history

    def fine_tune(self, epochs=20, learning_rate=1e-5):
        """Fine-tune the model with lower learning rate"""

        if self.model is None:
            raise ValueError("Train the model first before fine-tuning")

        if self.model_type == 'custom':
            print("Fine-tuning not applicable for custom CNN")
            return

        print(f"\n{'='*80}")
        print("Starting fine-tuning...")
        print(f"{'='*80}\n")

        # Unfreeze base model layers
        self.model_wrapper.unfreeze_base_model(layers_to_unfreeze=30)

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )

        # Continue training
        fine_tune_history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=self.setup_callbacks('models/fine_tuned_model.h5'),
            verbose=1
        )

        # Append to history
        for key in self.history.history.keys():
            self.history.history[key].extend(fine_tune_history.history[key])

        print(f"\n{'='*80}")
        print("Fine-tuning completed!")
        print(f"{'='*80}\n")

        return fine_tune_history

    def evaluate(self):
        """Evaluate the model on test data"""

        if self.model is None:
            raise ValueError("No model to evaluate")

        generator = self.test_generator if self.test_generator else self.val_generator

        print(f"\n{'='*80}")
        print("Evaluating model...")
        print(f"{'='*80}\n")

        # Get predictions
        generator.reset()
        predictions = self.model.predict(generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = generator.classes

        # Calculate metrics
        print("\nClassification Report:")
        print("="*80)
        print(classification_report(true_classes, predicted_classes,
                                    target_names=self.class_names))

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self.plot_confusion_matrix(cm, self.class_names)

        # Overall accuracy
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(
            generator, verbose=1
        )
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        return predictions, predicted_classes, true_classes

    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""

        if self.history is None:
            raise ValueError("No training history to plot")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'],
                        label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'],
                        label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()

    def plot_confusion_matrix(self, cm, class_names, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    # Example usage
    DATA_DIR = "data/wheat_diseases"  # Change to your data directory

    # Initialize trainer
    trainer = WheatDiseaseTrainer(
        data_dir=DATA_DIR,
        img_size=(224, 224),
        batch_size=32,
        model_type='resnet50',  # Options: 'custom', 'resnet50', 'mobilenet', 'efficientnet'
        num_classes=5  # Update based on your dataset
    )

    # Train model
    history = trainer.train(epochs=50, learning_rate=0.001)

    # Plot training history
    trainer.plot_training_history()

    # Optional: Fine-tune
    # trainer.fine_tune(epochs=20, learning_rate=1e-5)

    # Evaluate
    trainer.evaluate()
