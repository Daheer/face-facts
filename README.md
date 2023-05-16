# Face Facts
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

