import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st
class_names=['Heavy Traffic','Less Traffic', 'Moderate Traffic']
model = load_model('model.h5')
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    result = class_names[np.argmax(predictions)]

    if result is 'Less Traffic':
        st.success('Traffic is less')
    elif result is 'Moderate Traffic':
        st.info('Traffic is Moderate')
    else:
        st.warning('Traffic is heavy')
#Setting Title of App
image = Image.open('index.jpg')
st.image(image,use_column_width=True)
st.title("Parking Traffic Detection")
st.markdown("Upload images of Parking Spot location")

file = st.file_uploader("Choose an image...", type=['png', 'jpg' , 'jpeg'])
submit = st.button('Predict')


#On predict button click
if submit:

    if file is not None:
        
        images = Image.open(file)
        images = images.resize((200, 200))
        #images=images.convert('RGB')

        st.image(images)
        
        predicted_class= predict(images)

