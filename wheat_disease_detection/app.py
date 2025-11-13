"""
Streamlit Web Application for Wheat Disease Detection
Upload aerial wheat photos and get instant disease predictions
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

# Configure TensorFlow to avoid threading issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass  # Ignore GPU errors in web app

# Page configuration
st.set_page_config(
    page_title="Wheat Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F1F8E9;
        border: 2px solid #AED581;
        margin: 10px 0;
    }
    .healthy {
        background-color: #C8E6C9;
        border-color: #4CAF50;
    }
    .diseased {
        background-color: #FFCDD2;
        border-color: #F44336;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load the trained model with caching to avoid reloading"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_class_names(class_names_path=None):
    """Load class names with caching"""
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            return json.load(f)
    else:
        # Default class names
        return [
            'Healthy',
            'Leaf Rust',
            'Stem Rust',
            'Yellow Rust',
            'Septoria'
        ]


class WheatDiseaseApp:
    """Streamlit app for wheat disease detection"""

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.img_size = (224, 224)
        self.model_loaded = model is not None

    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize
        img = image.resize(self.img_size)

        # Convert to array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image):
        """Make prediction on image"""
        if not self.model_loaded:
            st.error("Model not loaded!")
            return None

        # Preprocess
        img_array = self.preprocess_image(image)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # All probabilities
        all_probs = {
            cls: float(predictions[i])
            for i, cls in enumerate(self.class_names)
        }

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }

    def plot_probabilities(self, probabilities):
        """Create interactive bar chart of probabilities"""
        classes = list(probabilities.keys())
        probs = list(probabilities.values())

        # Sort by probability
        sorted_data = sorted(zip(classes, probs),
                             key=lambda x: x[1], reverse=True)
        classes, probs = zip(*sorted_data)

        # Color scheme
        colors = ['#4CAF50' if p == max(probs) else '#90CAF9' for p in probs]

        fig = go.Figure(data=[
            go.Bar(
                x=probs,
                y=classes,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{p:.1%}' for p in probs],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Disease Probability Distribution",
            xaxis_title="Probability",
            yaxis_title="Disease Class",
            xaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )

        return fig

    def display_prediction_card(self, result):
        """Display prediction result in a card"""
        predicted_class = result['predicted_class']
        confidence = result['confidence']

        # Determine if healthy or diseased
        is_healthy = predicted_class.lower() == 'healthy'
        card_class = 'healthy' if is_healthy else 'diseased'

        # Status icon
        icon = "✅" if is_healthy else "⚠️"

        st.markdown(f"""
            <div class="prediction-box {card_class}">
                <h2>{icon} Prediction: {predicted_class}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
            </div>
        """, unsafe_allow_html=True)

        # Recommendations
        if is_healthy:
            st.success(
                "🌾 **Status:** Your wheat appears healthy! Continue regular monitoring.")
        else:
            st.warning(
                f"⚠️ **Status:** {predicted_class} detected. Consider treatment options.")
            self.display_treatment_recommendations(predicted_class)

    def display_treatment_recommendations(self, disease):
        """Display treatment recommendations based on disease"""
        recommendations = {
            'Leaf Rust': [
                "Apply fungicides containing triazoles or strobilurins",
                "Remove and destroy infected plant debris",
                "Improve air circulation between plants",
                "Consider resistant wheat varieties for future planting"
            ],
            'Stem Rust': [
                "Apply fungicides immediately upon detection",
                "Plant resistant varieties in subsequent seasons",
                "Monitor nearby barberry plants (alternate host)",
                "Ensure adequate plant nutrition to boost resistance"
            ],
            'Yellow Rust': [
                "Apply foliar fungicides at early disease stages",
                "Use disease-resistant cultivars",
                "Avoid excessive nitrogen fertilization",
                "Monitor weather conditions (cool, moist weather favors development)"
            ],
            'Septoria': [
                "Apply fungicides during flag leaf emergence",
                "Practice crop rotation",
                "Remove crop residue after harvest",
                "Ensure good drainage and air circulation"
            ]
        }

        if disease in recommendations:
            st.subheader("💊 Treatment Recommendations")
            for i, rec in enumerate(recommendations[disease], 1):
                st.write(f"{i}. {rec}")


def main():
    """Main application function"""

    # Header
    st.markdown('<h1 class="main-header">🌾 Wheat Disease Detection System</h1>',
                unsafe_allow_html=True)
    st.markdown(
        "**AI-powered detection of wheat diseases from aerial photographs**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        model_path = st.text_input(
            "Model Path",
            value="models/best_wheat_disease_model.h5",
            help="Path to the trained model file"
        )

        class_names_path = st.text_input(
            "Class Names Path (optional)",
            value="models/class_names.json",
            help="Path to class names JSON file"
        )

        st.divider()

        # Information
        st.header("ℹ️ About")
        st.write("""
        This application uses a Convolutional Neural Network (CNN) 
        to detect diseases in wheat from aerial photographs.
        
        **Supported Diseases:**
        - Leaf Rust
        - Stem Rust
        - Yellow Rust
        - Septoria
        - Healthy (no disease)
        """)

    # Load model and class names (cached)
    model = None
    class_names = load_class_names(class_names_path)

    if os.path.exists(model_path):
        with st.spinner("Loading model..."):
            model = load_model(model_path)
            if model:
                st.sidebar.success("✅ Model loaded successfully!")
                st.sidebar.divider()
                st.sidebar.header("📊 Model Info")
                st.sidebar.metric("Input Size", "224x224")
                st.sidebar.metric("Classes", len(class_names))
                st.sidebar.info(f"Classes: {', '.join(class_names)}")
    else:
        st.sidebar.warning(f"⚠️ Model file not found: {model_path}")
        st.sidebar.info("Please train a model first or update the model path.")

    # Initialize app
    app = WheatDiseaseApp(model, class_names)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>',
                    unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a wheat image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an aerial photo of wheat field"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Predict button
            if st.button("🔍 Detect Disease", type="primary", use_container_width=True):
                if not app.model_loaded:
                    st.error("⚠️ Please load the model first!")
                else:
                    with st.spinner("Analyzing image..."):
                        # Make prediction
                        result = app.predict(image)

                        if result:
                            # Store result in session state
                            st.session_state['result'] = result
                            st.success("✅ Analysis complete!")

    with col2:
        st.markdown('<h2 class="sub-header">📊 Results</h2>',
                    unsafe_allow_html=True)

        # Display results if available
        if 'result' in st.session_state:
            result = st.session_state['result']

            # Display prediction card
            app.display_prediction_card(result)

            # Display probability chart
            fig = app.plot_probabilities(result['all_probabilities'])
            st.plotly_chart(fig, use_container_width=True)

            # Detailed probabilities
            with st.expander("📋 Detailed Probabilities"):
                for cls, prob in sorted(result['all_probabilities'].items(),
                                        key=lambda x: x[1], reverse=True):
                    st.write(f"**{cls}:** {prob:.4f} ({prob:.2%})")

            # Download results
            if st.button("💾 Download Results"):
                results_json = json.dumps(result, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"wheat_disease_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Upload an image and click 'Detect Disease' to see results")

    # Additional information
    st.divider()

    with st.expander("🔬 How It Works"):
        st.write("""
        ### Detection Process
        
        1. **Image Upload**: Upload an aerial photograph of wheat field
        2. **Preprocessing**: Image is resized and normalized
        3. **CNN Analysis**: Deep learning model analyzes visual patterns
        4. **Classification**: Model predicts the most likely disease class
        5. **Results**: Confidence scores and recommendations provided
        
        ### Model Architecture
        
        The model uses a Convolutional Neural Network (CNN) trained on thousands 
        of wheat disease images. It can detect:
        - Visual symptoms of rust diseases
        - Leaf spotting patterns
        - Color changes indicating disease
        - Overall plant health indicators
        
        ### Accuracy Notes
        
        - Best results with clear, well-lit images
        - Model confidence above 80% is generally reliable
        - Always consult an agronomist for treatment decisions
        - Early detection increases treatment effectiveness
        """)

    # Footer
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🌾 Wheat Disease Detection System | Built with Streamlit & TensorFlow</p>
            <p>For agricultural consultation, please contact your local extension office</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
