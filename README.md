Fake News Detection using Machine Learning
Overview
This project detects fake news using machine learning and natural language processing techniques. It classifies news articles as real or fake based on their content and uses a web application for real-time predictions.

Problem Definition
We aim to identify news sources that spread fake news. By classifying these sources, we can predict that any future articles from them are likely fake. This model helps social networks make fake news less visible.

Project Structure
Images: Diagrams, reports, and screenshots.
Dataset: Training and test data from Kaggle.
Static: Web app assets (images).
Templates: HTML files for the web app.
app.py: Flask app for fake news detection.
model.pkl: Pre-trained machine learning model.
vector.pkl: Pre-trained vectorizer.
Datasets
train.csv: Includes article details with a label indicating reliability (1: fake, 0: real).
test.csv: Same as train.csv but without labels.
Model
The model used is the Passive Aggressive Classifier (PAC), which achieved 96% accuracy. It is efficient for real-time classification.

Prerequisites
Python 3.7+
Install dependencies from requirements.txt.