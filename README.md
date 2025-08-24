# Arabic Sign Language Recognition ✋🤖

## 📌 Overview
This project is an AI-based system for recognizing **Arabic Sign Language letters** using image classification.  
The goal is to support communication and provide tools that can help the community by leveraging **computer vision** and **machine learning** techniques.

## ⚙️ Features
- Detects and classifies Arabic sign language letters.
- Built using **Python**, **TensorFlow/Keras**, and **OpenCV**.
- Custom dataset collected manually for training.
- Lightweight model that can be extended with more letters and data.

 ## 📊 Dataset
- Images were collected manually for Arabic sign language letters.
- Each letter has 300+ images.

## 🔮 Future Improvements 
- Add more letters until full Arabic alphabet is covered.
- Increase dataset size for better accuracy.
- Deploy as a web or mobile application.

## 📊 Training Results

- Epochs: 15
- Batch size: (default by Keras)
- Image size: 64x64 (grayscale)

## 🔹 Accuracy Progression:

- **Epoch 1:** 34% training accuracy – 63% validation accuracy
- **Epoch 3:** 80% training accuracy – 92% validation accuracy
- **Epoch 5:** 90% training accuracy – 96% validation accuracy
- **Epoch 7:** 94% training accuracy – 98% validation accuracy
- **Epoch 15:** 97% training accuracy – 98% validation accuracy 

## 🔹 Loss Progression:

- Started with high loss (1.69)
- Dropped significantly to around 0.08 by the last epoch

## 🔹 Final Results:

- **Test Accuracy:** 98% 
- **Validation Loss:** stabilized around 0.11
- The model generalizes well, with minimal overfitting.


  **If you have questions or suggestions, feel free to reach out.**
