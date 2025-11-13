# Troubleshooting: Mutex Lock Error

## Error Message
```
libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
```

## What Causes This Error?

This error occurs due to threading conflicts between TensorFlow and Streamlit. The issue happens when:
1. TensorFlow models are loaded multiple times due to Streamlit's rerun mechanism
2. Multiple threads try to access TensorFlow operations simultaneously
3. macOS-specific threading issues with TensorFlow

## Fixes Applied

### 1. Threading Configuration (app.py & train.py)
Added environment variables and TensorFlow threading limits:
```python
os.environ['OMP_NUM_THREADS'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
```

### 2. Model Caching (app.py)
Used Streamlit's `@st.cache_resource` to load models only once:
```python
@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path, compile=False)
    return model
```

### 3. GPU Memory Growth
Enabled dynamic GPU memory allocation:
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

## Testing the Fix

Run the test script first:
```bash
python test_app.py
```

If successful, run the app:
```bash
streamlit run app.py
```

## If Error Persists

### Option 1: Use CPU-only TensorFlow
```bash
pip uninstall tensorflow tensorflow-metal tensorflow-macos
pip install tensorflow-cpu
```

### Option 2: Reinstall TensorFlow
```bash
# For regular systems
pip install --upgrade tensorflow

# For macOS with M1/M2
pip install --upgrade tensorflow-macos tensorflow-metal
```

### Option 3: Use Different Python Version
TensorFlow works best with Python 3.8-3.11:
```bash
python --version  # Check your version
```

### Option 4: Run with Different Backend
Set environment variable before running:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
streamlit run app.py
```

## Additional macOS M1/M2 Notes

If on Apple Silicon:
1. Ensure you have the correct TensorFlow version:
   ```bash
   pip install tensorflow-macos tensorflow-metal
   ```

2. Set environment variables:
   ```bash
   export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
   export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
   ```

3. Use miniforge instead of regular conda if using conda

## Prevention Tips

1. Always load models using Streamlit caching
2. Avoid loading models in loops or callbacks
3. Set threading limits before importing TensorFlow
4. Use `compile=False` when loading models for inference only
5. Test with `test_app.py` before running full application

## Still Having Issues?

1. Check TensorFlow version compatibility:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

2. Clear Streamlit cache:
   ```bash
   streamlit cache clear
   ```

3. Run with verbose logging:
   ```bash
   TF_CPP_MIN_LOG_LEVEL=0 streamlit run app.py
   ```

4. Try running outside Streamlit:
   ```bash
   python predict.py  # Use the standalone prediction script
   ```
