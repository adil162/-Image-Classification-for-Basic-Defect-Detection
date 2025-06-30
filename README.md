# -Image-Classification-for-Basic-Defect-Detection

This project focuses on using **Convolutional Neural Networks (CNNs)** to classify manufactured parts as **defective** or **non-defective** based on images. Built during my internship at **Mentoga** via **Skilled Score**, the model is designed for basic-level defect detection in quality control pipelines.

## 🎯 Objective

To develop a reliable image classification model that automates defect detection, improving quality assurance and reducing human error in manufacturing.

## 🔍 What It Does

- Classifies images into two categories: **Defective** and **Non-Defective**
- Preprocesses images (resizing, normalization, augmentation)
- Trains a CNN from scratch with TensorFlow/Keras
- Evaluates performance using accuracy, loss curves, and confusion matrix
- Supports test image prediction and visualization

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy & Matplotlib
- scikit-learn
- VS Code

## 🗂️ Dataset

- Custom dataset of labeled product images  
- Folder structure:
dataset/
├── defective/
└── non_defective/

bash
Copy
Edit

> *(Dataset used strictly for educational and research purposes.)*

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/defect-classification.git
cd defect-classification
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train the model:

bash
Copy
Edit
python train_model.py
Test predictions:

bash
Copy
Edit
python predict.py --image path_to_image.jpg
📊 Model Performance
Validation Accuracy: ~94%

Precision & Recall: High on both classes

Includes training loss/accuracy graphs

🧪 Features
CNN with Conv2D, MaxPooling, and Dropout

Data augmentation (rotation, flipping, zooming)

Visualization of test predictions

Easy model saving and loading

🧩 Future Work
Add Grad-CAM visualization for interpretability

Deploy as a web app using Flask or Streamlit

Extend to multi-class defect detection

🏅 Internship Acknowledgement
This project was built during my internship with Mentoga, organized by Skilled Score.

🙋‍♂️ About Me
Hi, I'm Adil Shah — a budding AI developer from Pakistan 🇵🇰
I’m passionate about solving real-world problems with deep learning and computer vision.
Linkedin Profile: www.linkedin.com/in/syed-adil-shah-8a1537365
GitHub Profile: https://github.com/adil162
