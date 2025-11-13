#!/bin/bash

# Run script for Wheat Disease Detection App
# This script sets up the environment and runs the Streamlit app

echo "🌾 Wheat Disease Detection - Startup Script"
echo "============================================="
echo ""

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the wheat_disease_detection folder"
    exit 1
fi

# Run test first
echo "Step 1: Testing TensorFlow configuration..."
echo "-------------------------------------------"
python test_app.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Step 2: Starting Streamlit app..."
    echo "-------------------------------------------"
    streamlit run app.py
else
    echo ""
    echo "❌ Test failed. Please check the error messages above."
    echo "See TROUBLESHOOTING.md for help."
    exit 1
fi
