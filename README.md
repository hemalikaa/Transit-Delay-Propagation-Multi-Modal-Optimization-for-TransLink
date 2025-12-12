TransLink Transit Delay Prediction
A deep learning system for predicting transit delays in Vancouver's public transportation network, achieving 37.3% improvement over the target baseline.
🎯 Project Overview
This project develops a predictive model for TransLink Vancouver transit delays using deep neural networks and comprehensive feature engineering. The model processes real-time and historical transit data to predict delays with high accuracy, enabling better service planning and passenger communication.
Key Results:

Mean Absolute Error (MAE): 0.1442 minutes (Target: 0.229 minutes)
R² Score: 0.9783
Performance Improvement: 37.3% better than target baseline
Dataset Size: 4.78 million transit records

📁 Repository Structure
├── research/
│   ├── IE7615_project.ipynb    # Experimentation and model development
│   └── file                     # Supporting research files
├── streamlit_app/
│   ├── models/
│   │   ├── delay_prediction_final.h5    # Trained deep neural network
│   │   └── scaler_final.pkl              # Feature scaler for preprocessing
│   ├── app.py                   # Streamlit web application
│   └── requirements.txt         # App dependencies
├── final_result.ipynb           # Clean results and analysis
├── requirements.txt             # Project dependencies
├── .gitignore
└── README.md
🔑 Key Features
Feature Engineering (16 Features)
The model uses 16 carefully engineered features, with upstream delay emerging as the strongest predictor (98.9% correlation):

Temporal Features: Hour of day, day of week, time period classifications
Route Characteristics: Route type, direction, service patterns
Spatial Features: Stop sequence, geographic clustering
Historical Patterns: Average delays, schedule adherence
Real-time Context: Current trip status, upstream delays

Model Architecture

Deep Neural Network with multiple hidden layers
Optimized for regression tasks
Trained on 4.78 million transit events
Feature scaling for improved convergence

📊 Data Sources
Real-time Data Collection
Real-time transit data was collected using an automated system built with GitHub Actions:

Repository: translink-realtime-collector
Collection Method: Automated GTFS-RT API polling
Data Points: 4.78 million records
Time Period: [Specify your collection period]

Static Transit Data
Static schedule and route information from TransLink Vancouver:

Source: TransLink Open API
Format: GTFS (General Transit Feed Specification)
Includes: Routes, stops, schedules, and service patterns

Note: Due to size constraints, the processed dataset (4.78M records) is not included in this repository. The complete data collection and processing pipeline is documented in the notebooks.
🚀 Getting Started
Prerequisites
bashPython 3.8+
TensorFlow/Keras
Pandas, NumPy
Scikit-learn
Streamlit
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/transit-delay-prediction.git
cd transit-delay-prediction

Install dependencies:

bashpip install -r requirements.txt

Run the notebooks:

bashjupyter notebook final_result.ipynb
Running the Streamlit App
bashcd streamlit_app
streamlit run app.py
The web application will open in your browser, allowing you to:

Input transit parameters (route, time, stop sequence, etc.)
Get real-time delay predictions
View model performance metrics
Explore feature importance

📓 Notebooks
research/IE7615_project.ipynb
Contains the complete experimentation process:

Data exploration and preprocessing
Multiple model architectures tested
Hyperparameter tuning
Feature engineering iterations
Model evaluation and comparison

final_result.ipynb
Clean presentation of final results:

Best model architecture and parameters
Performance metrics and visualizations
Feature importance analysis
Error analysis and model insights
Production-ready code
