import cv2
import os
import numpy as np
import tempfile
import time
import streamlit as st
from PIL import Image 
from io import BytesIO
import torch
from ultralytics import YOLO
from utils import Ultimate_Lightning, age_lightning, gender_lightning, race_lightning
import pandas as pd
import mediapipe as mp
from torch.cuda import is_available as gpu_ready
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import gdown

torch.manual_seed(42)

if not os.path.exists('models/age_model.pth') and not os.path.exists('models/gender_model.pth') and not os.path.exists('models/race_model.pth') and not os.path.exists('models/joint_model.pth') :
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1cG_2D4za0Gm6nV0j9p5Vq_8IeTAg6Tzq", output = 'models/age_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1Hwmlu8ubfgz2LPOAwA8xE6EtWxP41tQe", output = 'models/gender_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1m-zXYZe-iYuhBOAKBBPOt7jmdsYAmJIp", output = 'models/race_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1V2fk-7JZYLVfqQ8Xh9gphEGNV5UILEG8", output = 'models/joint_model')
    

DEMO_IMAGE = 'demo.JPG'
GENDER_DICT = {
    1: 'Female',
    0: 'Male'
}
RACE_DICT = {
    0: 'White', 
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}
device = 'cuda' if gpu_ready() else 'cpu'

def load_face_detector():
  base_options = python.BaseOptions(model_asset_path='models/detector.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)
  return detector

@st.cache_data
def load_model():
  joint_model = Ultimate_Lightning()
  age_model = age_lightning()
  age_model.load_state_dict(torch.load('models/age_model.pth'))
  race_model = race_lightning()
  race_model.load_state_dict(torch.load('models/race_model.pth'))  
  gender_model = gender_lightning()
  gender_model.load_state_dict(torch.load('models/gender_model.pth'))
  joint_model.load_state_dict(torch.load('models/joint_model.pth'))

  return age_model, gender_model, race_model, joint_model

st.set_page_config(page_title="Face-Facts")
st.title('Face-Facts')

app_mode = st.sidebar.selectbox('Choose Page', ['About the App', 'Run Face Facts'])

st.markdown(
      """
      <style>
          [data-testid = 'stSidebar'][aria-expanded = 'true'] > div:first-child{
            width: 350px
          }
          [data-testid = 'stSidebar'][aria-expanded = 'false'] > div:first-child{
            width: 350px
            margin-left: -350px
          }
      </style>
      """, unsafe_allow_html = True
)

if app_mode == 'About the App':
  st.markdown('')

elif app_mode == 'Run Face Facts':
  
  age_model, gender_model, race_model, joint_model = load_model()
  detector = load_face_detector()

  st.sidebar.markdown('---')
  use_single_model = st.sidebar.checkbox('Use single model', value = False)
  
  kpi1, age_col, gender_col, race_col, kpi5 = st.columns(5)
  with age_col:
    st.markdown('**Age**')
    age_text = st.markdown('0')
  with gender_col:
    st.markdown('**Gender**')
    gender_text = st.markdown('0')
  with race_col:
    st.markdown('**Race**')
    race_text = st.markdown('0')

  img_file_buffer = st.sidebar.file_uploader('Upload an Image', type = ['jpg', 'png', 'jpeg'])
  if img_file_buffer:
    buffer = BytesIO(img_file_buffer.read())
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    image_orig = cv2.imdecode(data, cv2.IMREAD_COLOR)
  else:
    demo_image = DEMO_IMAGE
    image_orig = cv2.imread(demo_image, cv2.IMREAD_COLOR)
  
  st.sidebar.text('Original Image')
  st.sidebar.image(image_orig, channels = 'BGR')

  image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
  image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
  detection_result = detector.detect(image)
  if detection_result.detections != []:
    res = detection_result.detections[0].bounding_box
    x, y, w, h = res.origin_x, res.origin_y, res.width, res.height
    image = image.numpy_view()[y:y+h, x:x+w]
    image = cv2.resize(image, (200, 200))
    image = torch.from_numpy(image).permute(2, 0, 1) / 255.
    age_model.eval()
    gender_model.eval()
    race_model.eval()
    joint_model.eval()
    with torch.no_grad():
      if use_single_model:
        age_pred, gender_pred, race_pred = joint_model(image.unsqueeze(0))
      else:
        age_pred, gender_pred, race_pred = age_model(image.unsqueeze(0)), gender_model(image.unsqueeze(0)), race_model(image.unsqueeze(0))
    age = int(age_pred.item())
    gender = GENDER_DICT[int(gender_pred.item() > 0.5)]
    race = RACE_DICT[race_pred.argmax(dim = 1).item()] 
    gender_emoji = '♂️' if gender == 'Male' else '♀️'
    gender_color = 'blue' if gender == 'Male' else 'pink'
  else:
    age = '-'
    gender = '-'
    gender_emoji = '-'
    gender_color = ''  
    race = '-'
    st.error('No face detected in the image')
  age_text.write(f'<h1> {age} </h1>', unsafe_allow_html = True)
  gender_text.write(f"<h1 style='color: {gender_color};'>{gender_emoji}</h1>", unsafe_allow_html=True)
  race_text.write(f"<h1> {race} </h1>", unsafe_allow_html = True)

  st.markdown('---')

  if not age == '-':
    with st.expander('🔻 More Details 🔻'):
      gender_decimal = gender_pred.item() if gender == 'Female' else abs(1 - gender_pred.item())
      gender_precentage = f'{100 * gender_decimal:.2f}%'
      st.write('---')
      cols = st.columns(2)
      with cols[0]:
        st.markdown("<h3 style='color: gray;'>Face of Interest</h3>", unsafe_allow_html=True)
        st.image(image.permute(1, 2, 0).numpy(), channels = 'RGB', use_column_width = True)
      with cols[1]:
        st.write(f"<h3 style = 'color: {gender_color};'> {gender}; {gender_precentage} Probability </h3>", unsafe_allow_html = True)
        st.progress(gender_decimal)
        st.write('---')
        st.bar_chart(pd.DataFrame({'Probability': race_pred.squeeze().numpy(), 'Race': list(RACE_DICT.values())}), x = 'Race', y = 'Probability')
      st.write('---')
