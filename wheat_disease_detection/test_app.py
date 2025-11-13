"""
Simple test script to verify TensorFlow configuration
Run this before starting the full app
"""

import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'


print("Testing TensorFlow Configuration...")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Configure threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
if gpus:
    for gpu in gpus:
        print(f"  - {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("    Memory growth enabled")
        except RuntimeError as e:
            print(f"    Warning: {e}")
else:
    print("  - Running on CPU")

# Test model creation
print("\nTesting model creation...")
try:
    test_model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(5, activation='softmax')
    ])
    print("✓ Model created successfully")

    # Test prediction
    print("\nTesting prediction...")
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = test_model.predict(test_input, verbose=0)
    print(f"✓ Prediction successful, output shape: {output.shape}")

    print("\n✅ All tests passed! Your TensorFlow setup is working correctly.")
    print("\nYou can now run:")
    print("  streamlit run app.py")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Try reinstalling TensorFlow: pip install --upgrade tensorflow")
    print("2. Check your Python version (3.8-3.11 recommended)")
    print("3. On macOS with M1/M2, use: pip install tensorflow-macos tensorflow-metal")
