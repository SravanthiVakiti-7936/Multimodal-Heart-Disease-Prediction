Multimodal AI Analysis for Heart Disease Prediction
Project Overview

Heart disease prediction is an important problem in healthcare. Early detection of heart disease can help doctors provide timely treatment and reduce health risks. Traditional systems usually rely only on clinical data, which may not capture the complete condition of the heart.

This project proposes a Multimodal AI System that combines clinical patient data and ECG signal analysis to predict heart disease risk. The system uses Machine Learning and Deep Learning models to analyze patient data and generate predictions through a web interface.

Technologies Used

Python

Scikit-learn

TensorFlow

Keras

Pandas

NumPy

Streamlit

Models Used

Random Forest (Machine Learning)
Used for analyzing clinical data such as age, cholesterol, and blood pressure.

Convolutional Neural Network – CNN (Deep Learning)
Used for analyzing ECG signals and detecting abnormal heart patterns.

Decision-Level Fusion
Combines predictions from both models to generate the final heart disease risk prediction.

System Architecture
<img width="2385" height="1446" alt="system_architecture_heart_prediction" src="https://github.com/user-attachments/assets/9f4bcba8-d949-4bc0-96cf-858dc1c7d74a" />


The system processes clinical data and ECG signals, analyzes them using machine learning and deep learning models, and combines the predictions to produce the final result.

Methodology
<img width="2385" height="1446" alt="methodology_heart_prediction" src="https://github.com/user-attachments/assets/555fa8a5-7322-4c20-a4c3-c3d1cb78693a" />

The project follows these steps:

Data collection from clinical datasets and ECG signal datasets

Data preprocessing and cleaning

Training machine learning and deep learning models

Combining model predictions using multimodal fusion

Displaying results through a Streamlit web application

Application Screenshots
Clinical Data Input
<img width="1920" height="1080" alt="Screenshot 2026-03-18 071934" src="https://github.com/user-attachments/assets/6a1e2c17-8688-4802-9ea4-d08ec5ed0ef6" />


ECG Signal Upload
<img width="1920" height="1080" alt="Screenshot 2026-03-18 072007" src="https://github.com/user-attachments/assets/acdb6a10-328b-4312-b25b-e61893d34735" />


Final Prediction
<img width="1920" height="1080" alt="Screenshot 2026-03-18 072026" src="https://github.com/user-attachments/assets/b92cc307-e608-45e9-b6b3-07de3d933c51" />


Results

The system successfully predicts heart disease risk based on both clinical data and ECG signals.

Example prediction output:

Clinical Model Prediction: Low Risk

ECG Model Prediction: Normal ECG

Final Multimodal Prediction: Low Heart Disease Risk

This demonstrates how combining multiple data sources improves prediction reliability.

How to Run the Project
Step 1: Install required libraries
pip install -r requirements.txt
Step 2: Run the Streamlit application
python -m streamlit run app.py
Step 3: Open in browser
http://localhost:8501

Enter clinical data and upload an ECG file to view predictions.

Future Work

Improve prediction accuracy using larger medical datasets

Integrate additional medical data such as echocardiogram images

Implement Explainable AI techniques for better model transparency

Deploy the system as a cloud-based healthcare application

Author

Vakiti Sravanthi
