# 🌾 Wheat Disease Detection - CNN Classification Application

An AI-powered application for detecting diseases in wheat from aerial photographs using Convolutional Neural Networks (CNN).

## 🎯 Features

- **Multiple CNN Architectures**: Custom CNN, ResNet50, MobileNetV2, and EfficientNetB0
- **Transfer Learning Support**: Leverage pre-trained models for better accuracy
- **Data Augmentation**: Robust training with image augmentation techniques
- **Interactive Web App**: Easy-to-use Streamlit interface for image upload and prediction
- **Batch Processing**: Process multiple images at once
- **Detailed Analytics**: Confusion matrices, training history plots, and detailed metrics
- **Treatment Recommendations**: Disease-specific treatment suggestions

## 📋 Supported Disease Classes

1. **Healthy** - No disease detected
2. **Leaf Rust** - *Puccinia triticina*
3. **Stem Rust** - *Puccinia graminis*
4. **Yellow Rust** - *Puccinia striiformis*
5. **Septoria** - *Septoria tritici*

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd wheat_disease_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your wheat disease images in the following structure:

```
data/wheat_diseases/
├── train/
│   ├── Healthy/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── Leaf_Rust/
│   ├── Stem_Rust/
│   ├── Yellow_Rust/
│   └── Septoria/
├── test/
│   ├── Healthy/
│   ├── Leaf_Rust/
│   ├── Stem_Rust/
│   ├── Yellow_Rust/
│   └── Septoria/
```

**Note:** If you don't have a separate test folder, the training script will automatically split your training data into train/validation sets.

### 3. Train the Model

```python
# Option 1: Using the training script directly
python train.py

# Option 2: Custom training in Python
from train import WheatDiseaseTrainer

trainer = WheatDiseaseTrainer(
    data_dir="data/wheat_diseases",
    img_size=(224, 224),
    batch_size=32,
    model_type='resnet50',  # Options: 'custom', 'resnet50', 'mobilenet', 'efficientnet'
    num_classes=5
)

# Train the model
history = trainer.train(epochs=50, learning_rate=0.001)

# Plot training history
trainer.plot_training_history()

# Evaluate on test set
trainer.evaluate()

# Optional: Fine-tune for better accuracy
trainer.fine_tune(epochs=20, learning_rate=1e-5)
```

### 4. Make Predictions

```python
from predict import WheatDiseasePredictor

# Initialize predictor
predictor = WheatDiseasePredictor(
    model_path="models/best_wheat_disease_model.h5",
    class_names=['Healthy', 'Leaf Rust', 'Stem Rust', 'Yellow Rust', 'Septoria']
)

# Single image prediction
results = predictor.visualize_prediction("path/to/wheat/image.jpg")
print(results)

# Batch prediction
results = predictor.predict_from_directory("path/to/images/folder")
predictor.save_predictions_to_csv(results, "predictions.csv")
```

### 5. Run Web Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` to use the interactive web interface.

## 📁 Project Structure

```
wheat_disease_detection/
├── model.py              # CNN model architectures
├── train.py              # Training script
├── predict.py            # Prediction/inference script
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Dataset directory (create this)
├── models/              # Saved models directory
├── logs/                # TensorBoard logs (auto-created)
└── utils/               # Utility functions
```

## 🏗️ Model Architectures

### Custom CNN
- 4 convolutional blocks with batch normalization
- Max pooling and dropout layers
- Dense layers with 512 and 256 neurons
- Optimized for wheat disease detection

### Transfer Learning Options
- **ResNet50**: Deep residual network with 50 layers
- **MobileNetV2**: Lightweight model for mobile deployment
- **EfficientNetB0**: Balanced efficiency and accuracy

## 📊 Training Tips

1. **Data Quality**: Use clear, well-lit aerial photographs
2. **Data Quantity**: Aim for at least 500+ images per class
3. **Augmentation**: Enabled by default to prevent overfitting
4. **Learning Rate**: Start with 0.001, reduce if training is unstable
5. **Early Stopping**: Automatically stops training if no improvement
6. **Fine-tuning**: Unfreeze base model layers for extra performance

## 🎨 Web Application Features

- **Drag-and-drop** image upload
- **Real-time prediction** with confidence scores
- **Interactive charts** showing probability distribution
- **Treatment recommendations** for detected diseases
- **Download results** as JSON
- **Model configuration** from sidebar

## 📈 Performance Monitoring

The training script automatically generates:
- Training/validation accuracy and loss plots
- Precision and recall metrics
- Confusion matrices
- Classification reports
- TensorBoard logs for detailed analysis

View TensorBoard logs:
```bash
tensorboard --logdir logs/
```

## 🔧 Advanced Usage

### Custom Model Configuration

```python
from model import WheatDiseaseCNN

# Build custom architecture
model = WheatDiseaseCNN(
    input_shape=(256, 256, 3),
    num_classes=5,
    model_type='custom'
)
model.build_model()
model.compile_model(learning_rate=0.0001)
```

### Data Generator Customization

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

custom_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)
```

## 📝 Dataset Sources

Consider these public datasets for wheat disease images:
- [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
- [CGIAR Wheat Atlas](https://wheatatlas.org/)
- [Agricultural Image Database](https://www.aphis.usda.gov/)

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model accuracy is low
- **Solution**: Increase training data, try transfer learning, adjust augmentation

**Issue**: Out of memory errors
- **Solution**: Reduce batch size, use a smaller model, resize images to smaller dimensions

**Issue**: Model not loading in app
- **Solution**: Check model path, ensure model was saved correctly, verify TensorFlow version

**Issue**: Poor predictions on real photos
- **Solution**: Ensure training data matches real-world conditions, add more diverse training samples

## 🤝 Contributing

To add new disease classes:
1. Add images to `data/wheat_diseases/train/NewDisease/`
2. Update `num_classes` parameter
3. Update `class_names` list in all scripts
4. Retrain the model

## 📚 References

- [Deep Learning for Plant Disease Detection](https://arxiv.org/abs/1604.03169)
- [Transfer Learning for Agricultural Applications](https://www.nature.com/articles/s41598-019-42374-8)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

## ⚖️ License

This project is for educational and research purposes. Consult with agricultural experts before making treatment decisions based on model predictions.

## 🙏 Acknowledgments

- Pre-trained models from TensorFlow/Keras
- Streamlit for the web framework
- Agricultural researchers for disease classification knowledge

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs and error messages
3. Verify data format and model paths
4. Consult agricultural extension services for disease-specific questions

---

**Note**: This is a machine learning tool to assist in disease detection. Always verify results with agricultural experts before taking action.
