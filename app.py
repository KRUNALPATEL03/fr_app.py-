import os
import tensorflow as tf
import numpy as np
import streamlit as st

# Define your model. Replace this with your actual model loading code.
def load_model():
    # Example model loading code
    return tf.keras.models.load_model('C:/Users/hp/OneDrive/Desktop/main flower/main/code/Flower_Recog_Model(123).h5')

def classify_images(image_path, model):
    # Your classification logic here

    # Example code for model usage
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

# Load your model
model = load_model()

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)

    # Call classify_images function with the loaded model
    result = classify_images(uploaded_file.name, model)
    st.markdown(f"The image belongs to **{result}**")



