"""
@author: Vineeth Kumar
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.write("""
          # Plant Classification
          """
          )
upload_file = st.sidebar.file_uploader("Upload Plant Leaf Images")
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('model.h5')
def import_n_pred(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size)
    image = ImageOps.grayscale(image)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred
if Generate_pred:
    image=Image.open(upload_file)
    with st.expander('Plant Leaf Image', expanded = True):
        st.image(image, width=350)#, use_column_width=True
    pred=import_n_pred(image, model)
    labels = ['Anhui Barberry', "Beale's Barberry", 'Big-fruited Holly', 
              'Camphortree', 'Canadian Poplar', 'Castor Aralia', 
              'Chinese Cinnamon', 'Chinese horse chestnut', 'Chinese redbud', 
              'Chinese Toon', 'Chinese Tulip Tree', 'Crape myrtle', 'Deodar', 
              'Ford Woodlotus', 'Glossy Privet', 'Goldenrain Tree', 
              'Japan Arrowwood', 'Japanese Cheesewood', 'Japanese Flowering Cherry',
              'Japanese maple', 'Maidenhair Tree', 'Nanmu', 'Oleander', 'Peach',
              'Pubescent Bamboo', 'Southern Magnolia', 'Sweet Osmanthus', 'Tangerine',
              'Trident Maple', 'True Indigo', 'Wintersweet', 'Yew Plum Pine']
    st.title("Prediction:{0} ({1})".format(labels[np.argmax(pred)], np.max(pred)))
