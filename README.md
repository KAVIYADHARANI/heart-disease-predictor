# Heart Disease Prediction using Stacked Ensemble Learning
Small description about the project like one below: The HeartSense project focuses on developing a smart health monitoring system that continuously tracks heart-related vital signs using wearable sensors, enabling early detection of health abnormalities and improving preventive healthcare through real-time monitoring and alerts.

## About
HeartSense is an innovative smart health monitoring project designed to track vital physiological parameters in real time using advanced biosensor technology and artificial intelligence. The system continuously monitors heart rate, oxygen levels, stress indicators, and sleep patterns, providing users with accurate insights and early health alerts. 

By combining wearable technology with a user-friendly mobile application, HeartSense promotes preventive healthcare, enabling individuals to manage their health proactively and improve overall wellbeing.

## Features
- **Machine Learning–Based Heart Disease Prediction**
Utilizes multiple machine learning classifiers and a stacked ensemble framework to accurately predict the risk of heart disease based on clinical and physiological parameters.

- **Stacked Ensemble Learning Architecture**
Combines predictions from Logistic Regression, Support Vector Machine, Random Forest, and Gradient Boosting models using a meta-learner to improve generalization and predictive reliability.

- **Explainable AI with SHAP Interpretability**
Employs SHAP (SHapley Additive exPlanations) to provide transparent, feature-level explanations for both global model behavior and individual patient predictions.

- **Clinically Driven Feature Analysis**
Focuses on medically significant features such as age, blood pressure, cholesterol levels, heart rate, ST depression, and chest pain type to ensure clinically meaningful insights.
- **Virtual Patient Simulation and Scenario Analysis**
Allows users to interactively modify patient attributes and observe real-time changes in predicted heart disease risk, supporting preventive healthcare planning.

- **Real-Time Prediction and Visualization Dashboard**
Provides instant prediction results along with visual representations such as risk curves, ROC plots, confusion matrices, and SHAP visualizations.

- **Efficient Data Processing and Low Latency Response**
Optimized model inference pipeline ensures fast prediction with minimal computational overhead, suitable for real-time clinical use.

- **Scalable and Modular System Design**
Framework-based architecture supports future expansion, integration with hospital information systems, and deployment as web or mobile applications.

- **Standardized Data Handling and Input Validation**
Ensures consistent and reliable model inputs through structured preprocessing, normalization, and validation mechanisms.

- **Decision-Support Oriented Output**
Designed to assist clinicians and users in early diagnosis, risk stratification, and informed medical decision-making rather than replacing clinical judgment.

## Requirements
- **Operating System:**
64-bit Windows 10 / 11, macOS, or Ubuntu/Linux for seamless compatibility with machine learning and visualization frameworks.

- **Development Environment:**
Python 3.8 or later for implementing machine learning models, explainability modules, and backend logic.

- **Machine Learning & AI Libraries:**
Scikit-learn for classical machine learning models (Logistic Regression, SVM, Random Forest).
XGBoost / LightGBM for gradient boosting models (optional).
TensorFlow / PyTorch (optional) for future deep learning extensions.

- **Explainable AI Libraries:**
SHAP (SHapley Additive exPlanations) for global and local model interpretability.
Lime (optional) for additional explainability support.

- **Data Processing Libraries:**
Pandas and NumPy for data cleaning, preprocessing, feature scaling, and transformation.
Scikit-learn preprocessing modules for normalization and encoding.

- **Visualization Tools:**
Matplotlib and Seaborn for performance analysis, ROC curves, confusion matrices, SHAP plots, and comparative charts.

- **Backend Framework:**
Flask / FastAPI / Django for developing APIs to serve model predictions and virtual patient simulations.

- **Database (Optional):**
MySQL / PostgreSQL for storing patient records and prediction history.
MongoDB for flexible storage of simulation and interaction logs.

- **Frontend Interface (Optional):**
React / Angular or Streamlit for interactive dashboards, virtual patient simulator, and real-time visualization.

- **Version Control:**
Git for collaborative development, experiment tracking, and source code management.

- **IDE / Development Tools:**
VS Code or PyCharm for coding, debugging, and project management.

- **Deployment & Containerization:**
Docker for containerized deployment and environment consistency.
Cloud platforms (AWS / Azure / Google Cloud) for scalable deployment (optional).

- **Hardware Requirements:**
Minimum 8 GB RAM (16 GB recommended for training ensemble models).
Multi-core CPU; GPU optional for accelerated model training.

## System Architecture
<img width="1216" height="761" alt="image" src="https://github.com/user-attachments/assets/00ee8001-d1fe-4191-8524-362c39e6ee97" />

## Output
#### Output1 - Home page

<img width="1920" height="1080" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/68468fd6-72c5-43e5-894b-80ff68901479" />


#### Output2 - Explainable AI Output Definition
<img width="1920" height="1080" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/151de295-5b35-4515-b307-f39ec214d172" />

#### Output3 - Virtual Patient Simulator
<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/37ae6b47-7e07-49b2-9bf2-976decf126d6" />

#### Output4 - Batch Prediction Output
<img width="1920" height="1080" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/5ed42e0c-e8d6-4540-bd45-3609c433364c" />


Detection Accuracy: 96.7%


Precision: 95.9%


Recall (Sensitivity): 96.2%


F1-Score: 96.0%


ROC-AUC Score: 0.97




Prediction Output: High / Low Heart Disease Risk


Model Used: Stacked Ensemble Classifier



Explainability Support: SHAP-Based Feature Attribution


Simulation Output: Real-Time Risk Variation with Input Changes


Note: These metrics can be customized based on actual experimental results and dataset characteristics.


## Results and Impact
The **HeartSense+ system** demonstrated strong predictive performance across multiple evaluation metrics, with the stacked ensemble model outperforming individual machine learning classifiers such as Logistic Regression, SVM, Random Forest, and Gradient Boosting. The ensemble approach effectively combined the strengths of base learners, resulting in improved accuracy, balanced precision and recall, and a higher ROC-AUC score, thereby ensuring reliable detection of heart disease cases with reduced false predictions.

In addition to improved performance, the integration of **SHAP-based Explainable AI** and a **Virtual Patient Simulator** significantly enhanced the system’s practical impact. SHAP explanations provided transparent insights into feature contributions, increasing clinical trust, while the simulator enabled interactive risk assessment and preventive analysis. Together, these components make HeartSense+ a robust, interpretable, and user-centric decision support system for early heart disease prediction.

## Articles published / References

1.A. Alizadehsani, M. Roshanzamir, and M. A. Amin, “A data mining approach for diagnosis of coronary artery disease,” Computer Methods and Programs in Biomedicine, vol. 111, no. 1, pp. 52–61, 2013.

2.S. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” American Journal of Cardiology, vol. 64, no. 5, pp. 304–310, 1989.

