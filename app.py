import streamlit
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
@streamlit.cache_resource # Cache the model to avoid reloading
def load_model():
    model = tf.keras.models.load_model('model/breast_cancer_cnn_model.keras')
    return model

model = load_model()

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)             # Normalize pixel values
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

# Streamlit app UI
st.title("Breast Cancer Detector")
st.write("Upload a histopathological image of breast tissue to predict malignancy")

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    st.write("Processing the image...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Display the result
    if prediction[0] > 0.5:
        st.write("The model predicts: **malignant**")
    else:
        st.write("The model predicts: **Benign**")

st.write("Note: This model is trained for educational for educational purposes and  should not be used for medical diagnosis")
