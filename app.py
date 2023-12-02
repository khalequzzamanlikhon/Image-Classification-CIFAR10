import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the saved model
model = keras.models.load_model('cifar10_model.h5')  

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Streamlit app
st.title('Image Classification based on CIFAR-10 Dataset')

# Upload image through Streamlit

uploaded_file = st.file_uploader(f"Choose image from this list:{class_names}",type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for prediction
    image = np.array(image)
    image = tf.image.resize(image, (32, 32))  # Resize the image to match the model's expected sizing
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Display the result
    st.write("Prediction:", class_names[predicted_class])

    # Display the prediction probabilities
    st.write("Prediction Probabilities:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {prediction[0][i]:.4f}")
