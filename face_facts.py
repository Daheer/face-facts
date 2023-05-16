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
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1WPKUImAhEN-0leDbpuSu_DiHwXAF6vIN", output = 'models/age_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1b9gZzqwpHq7liWSVC6LxloFYkIEp20YV", output = 'models/gender_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1--9er5O6Hpe5Ete95lT50BPABTqzrWi2", output = 'models/race_model.pth')
    gdown.download(url = "https://drive.google.com/uc?export=download&id=1-8QdoEfxC6GIxfBM7P_Scx_vuv8XI5V_", output = 'models/joint_model.pth')
    

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
  st.markdown("""
Face Facts predicts age, race, and gender of individuals from their images using advanced machine learning techniques.

# Installation
- Clone this repo ` git clone https://github.com/Daheer/face-facts.git `
- Install requirements ` pip insatll requirements.txt `
- Launch streamlit app ` streamlit run face_facts.py `

# Usage

The 'Run Face Facts' section of the app lets you upload any image. It automatically retrieves one face and uses that face as "face of interest", the image of the selected face is run through the model for prediction. 

The app carries out prediction using two methods, the first employs a model that was trained to predict all three KPIs (age, gender and race) at once. The second uses three models, each trained for the individual task. 

Additionally, the app lets users access more in-depth details, including visualizing the selected face-of-interest, the confidence level of the gender prediction and the probability distribution of the person's race.

The app is available and can be accessed via two platforms
- [`Hugging Face Spaces`](https://huggingface.co/spaces/deedax/face-facts)
- [`Render`](https://face-facts.onrender.com/)

# Features

- Image upload
- Face detection
- Age, Gender and Race prediction
- Multiple prediction methods
- Seamless toggle between the prediction methods
- In-depth analysis

# Built Using
- [Python](https://python.org)
- [PyTorch](https://pytorch.org)
- [PyTorch Lightning Trainer](https://www.pytorchlightning.ai/index.html)
- [Mediapipe](https://developers.google.com/mediapipe)
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- [Streamlit](https://streamlit.io/)
    
# Details

- Dataset: [UTKFace](https://susanqq.github.io/UTKFace/) was used for face facts. It consists of over 20,000 facial images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc

- Data processing: The dataset exhibited a pronounced class imbalance in the age category, with a dominance of images of infants (0 - 4 years) as compared to other age ranges. This imbalance can adversely affect the performance of regression models that rely on accurate representation of all age groups. To address this issue, I employed a strategic approach that randomly discards 30% of examples containing images of individuals aged < 4. 

- Model selection and training details: For the standalone models, efficientnet_b0 was employed as the backbone architecture, with task-specific heads appended for each of the sub-tasks i.e. binary classification for gender prediction, regression for age prediction and multi-class classification for race prediction. The models were trained with separate pytorch lightning trainer modules for 25 epochs each. <br> As for the joint model, a single pytorch lightning trainer was used to train and optimize all three objectives.
After experimentation, I discovered the weighting scheme that performed well as follows: 
    * 0.001 * age loss
    * gender loss
    * race loss

# Performance

More details about performance can be seen here [notebook](training_face_facts.ipynb)
<br> Standalone Models

| Category | Validation Loss | Validation Accuracy |
|------------------|------------------|------------------|
| Age | 137.0 | - |
| Gender | - | 87.6% |
| Race | - | 77.5% |

Joint
| Category | Validation Loss | Validation Accuracy |
|------------------|------------------|------------------|
| Age | 174.0 | - |
| Gender | - | 88.4% |
| Race | - | 78.6% |
| Total | 0.639 | - |

# Contact

Dahir Ibrahim (Deedax Inc) <br>
Email - dahiru.ibrahim@outlook.com <br>
Twitter - https://twitter.com/DeedaxInc <br>
YouTube - https://www.youtube.com/@deedaxinc <br>
Project Link - https://github.com/Daheer/mask-check


  """)

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
