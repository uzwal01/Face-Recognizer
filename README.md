# Face-Recognizer
A real time Face Recognition System using python and openCV.


🎭 Face Recognizer

A Face Recognition application built with Python and OpenCV, capable of detecting and recognizing faces in real time. This project is designed as a step-by-step solution for recognizing faces by generating a dataset, training a classifier, and performing real-time detection with confidence levels.


🚀 Features

-Dataset Generation: Capture and store face images with unique labels.

-Model Training: Train a face recognition model using machine learning techniques.

-Real-Time Recognition: Detect and recognize faces from a live camera feed or video input.

-Accuracy and Confidence Metrics: Display real-time recognition confidence.


🛠️ Tech Stack

-Python

-OpenCV: For image processing and face detection.

-NumPy: For numerical computations.

-Machine Learning Libraries: For classifier training.


📂 Project Workflow

1. Dataset Generation:

-Capture images of individual faces and store them in a structured directory format.
-Each person's face images are saved in a unique folder for easy labeling.

2. Model Training:

-Train a machine learning classifier (e.g., LBPH, Haar Cascade, etc.) on the generated dataset.
-Save the trained model for future recognition.

3. Real-Time Recognition:

-Use the trained model to detect and recognize faces in real-time.
-Display the name and confidence score of the recognized faces.


🧑‍💻 How to Run the Project
**Run the Face Regonizer file in VS code and run all codes step by step to obtain appropriate result**

Prerequisites

1. Python 3.8+ installed on your system.

2. Required libraries installed:

  ---> pip install opencv-python opencv-contrib-python numpy matplotlib
  
  
Steps to Execute

1. Generate Dataset Run the dataset generator script to capture face images:

---> python src/dataset_generator.py

-This will save images in the dataset/ folder.

2. Train the Classifier Train the model using the generated dataset:

---> python src/train_classifier.py

-The trained model will be saved in the models/ directory.

3. Run Face Recognition Start real-time face recognition:

---> python src/face_recognition.py

-Use a webcam or video feed for live detection.


📊 Insights

-Accuracy Plotting: Visualize the model’s accuracy to evaluate its performance.

-Confidence Levels: Understand the model’s prediction confidence during real-time recognition.


📖 Usage Guide

1. Generate Data: Ensure faces are well-lit and clearly visible for optimal training.
 
2. Training the Model: Use a sufficiently large dataset for accurate predictions.
 
3. Recognition: The system highlights detected faces and displays recognition confidence.



