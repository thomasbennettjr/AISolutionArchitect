# Wheat Disease Detection Models

This directory will contain your trained models.

## Model Files

After training, you'll find:
- `best_wheat_disease_model.h5` - Best model based on validation accuracy
- `fine_tuned_model.h5` - Fine-tuned model (if applicable)
- `class_names.json` - Class names mapping

## Model Information

Default model configurations:
- **Input size**: 224x224x3
- **Output classes**: 5 (Healthy, Leaf Rust, Stem Rust, Yellow Rust, Septoria)
- **Architecture options**: 
  - Custom CNN
  - ResNet50 (transfer learning)
  - MobileNetV2 (transfer learning)
  - EfficientNetB0 (transfer learning)

## Loading Models

```python
from tensorflow import keras

# Load model
model = keras.models.load_model('models/best_wheat_disease_model.h5')

# Or use the predictor class
from predict import WheatDiseasePredictor
predictor = WheatDiseasePredictor('models/best_wheat_disease_model.h5')
```

## Model Performance

After training, document your model's performance:
- Training accuracy: ____%
- Validation accuracy: ____%
- Test accuracy: ____%
- Training time: ___ minutes
- Model size: ___ MB
