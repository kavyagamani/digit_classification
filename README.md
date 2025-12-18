Digit Classification (0–9)
Description

This project implements a digit classification system using Deep Learning. A Convolutional Neural Network (CNN) is trained to classify digit images from 0 to 9. The trained model is deployed using a Streamlit web application to provide real-time predictions through a simple user interface.

Technologies Used

Python

TensorFlow / Keras

Streamlit

NumPy, Pillow

Dataset

MNIST Dataset

70,000 digit images (0–9)

Image size: 28 × 28

Dataset Link:
http://yann.lecun.com/exdb/mnist/

How to Run
pip install -r requirements.txt
streamlit run streamlit_app.py

Output

Upload a digit image

Model predicts the digit (0–9)

Result displayed instantly

Result

Training Accuracy: ~99%

Testing Accuracy: ~98–99%
