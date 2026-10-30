🩺 Skin Disease Classification Using Deep Learning

A deep learning-based web application that classifies skin lesions into seven categories using a Convolutional Neural Network (CNN) with MobileNetV2 architecture. The project combines image classification, Django web development, and cloud deployment to provide an accessible interface for skin lesion analysis.

📌 Project Overview

Skin diseases are among the most common health conditions worldwide. Early identification and classification of suspicious skin lesions can help people understand when further medical evaluation may be needed.

This project uses transfer learning with MobileNetV2, trained on the HAM10000 dataset, to classify dermoscopic images into seven different skin lesion categories.

The trained model is integrated into a Django web application where users can upload skin lesion images and receive a predicted classification.

🎯 Objectives

- Develop a deep learning model for skin lesion classification.
- Apply transfer learning using MobileNetV2.
- Train the model using the HAM10000 dataset.
- Integrate the trained model into a Django web application.
- Deploy the application on AWS EC2.
- Provide a simple interface for uploading and analyzing skin images.

🧠 Dataset

HAM10000 (Human Against Machine with 10000 Training Images)

The HAM10000 dataset contains dermatoscopic images of pigmented skin lesions collected from different populations and stored by multiple imaging systems.

Dataset: "HAM10000 – Kaggle" (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

Classification Categories

The model classifies images into seven categories:

Code| Disease
akiec| Actinic Keratoses and Intraepithelial Carcinoma
bcc| Basal Cell Carcinoma
bkl| Benign Keratosis-like Lesions
df| Dermatofibroma
mel| Melanoma
nv| Melanocytic Nevi
vasc| Vascular Lesions

⚙️ Technology Stack

Component| Technology
Programming Language| Python
Deep Learning| TensorFlow / Keras
Model Architecture| MobileNetV2
Dataset| HAM10000
Backend| Django
Web Server| Gunicorn
Frontend| HTML, CSS, JavaScript
Version Control| Git & GitHub
Cloud Deployment| AWS EC2
Operating System| Ubuntu Server

🏗️ Project Architecture

User
  |
  v
Django Web Application
  |
  v
Image Upload and Validation
  |
  v
Image Preprocessing
  |
  v
MobileNetV2 Model
  |
  v
Prediction
  |
  v
Classification Result

📂 Project Structure

Deep-Learning-Model-For-Skin-Disease-Classification/
│
├── classifier/
│   ├── migrations/
│   ├── templates/
│   ├── static/
│   ├── views.py
│   ├── models.py
│   └── ...
│
├── config/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── ...
│
├── ml_models/
│   └── ham10000_mobilenetv2.keras
│
├── templates/
├── static/
├── media/
│
├── skin_disease_classification.ipynb
├── manage.py
├── requirements.txt
├── Procfile
├── build.sh
├── render.yaml
├── .env.example
└── README.md

Note: The directory structure above represents the main project components; individual files may vary.

🚀 Installation and Setup

1. Clone the Repository

git clone https://github.com/RajeetMavi/Deep-Learning-Model-For-Skin-Disease-Classification.git

2. Navigate to the Project Directory

cd Deep-Learning-Model-For-Skin-Disease-Classification

3. Create a Virtual Environment

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Linux / Ubuntu:

source venv/bin/activate

4. Install Dependencies

pip install -r requirements.txt

5. Configure Environment Variables

Configure the required environment variables, including:

- "SECRET_KEY"
- "DEBUG"
- "ALLOWED_HOSTS"

Do not commit secret keys or sensitive credentials to GitHub.

6. Apply Database Migrations

python manage.py migrate

7. Collect Static Files

python manage.py collectstatic --noinput

8. Run the Development Server

python manage.py runserver

Open your browser:

http://127.0.0.1:8000/

☁️ AWS EC2 Deployment

The Django application has been deployed on an AWS EC2 instance running Ubuntu Server.

Deployment Configuration

- Cloud Provider: Amazon Web Services (AWS)
- Service: EC2
- Operating System: Ubuntu Server
- Python Environment: Virtual Environment
- Application Server: Gunicorn
- Application Port: 8000

Running with Gunicorn

gunicorn config.wsgi:application --bind 0.0.0.0:8000

The application can be accessed using the EC2 public IPv4 address:

http://YOUR-EC2-PUBLIC-IP:8000

Note: Nginx reverse proxy and HTTPS configuration are planned as further deployment improvements.

🔍 How It Works

1. The user opens the Django web application.
2. The user uploads a skin lesion image.
3. Django validates and processes the uploaded image.
4. The image is passed to the trained MobileNetV2 model.
5. The model generates a classification prediction.
6. The predicted class is displayed on the website.

🔮 Future Improvements

- Add support for healthy skin image classification.
- Expand the dataset to include additional skin conditions, including fungal infections.
- Improve model performance and generalization.
- Implement Grad-CAM visualizations for model interpretability.
- Add more robust image validation.
- Configure Nginx and HTTPS for production deployment.
- Optimize model inference and server resource consumption.

⚠️ Medical Disclaimer

This project is developed for educational and research purposes only.

The model is not a clinically validated diagnostic system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

Predictions may be inaccurate. Always consult a qualified healthcare professional for medical evaluation.

👨‍💻 Author

Rajeet Mavi

GitHub: "@RajeetMavi" (https://github.com/RajeetMavi)

---

⭐ If you find this project interesting, consider giving the repository a star!
